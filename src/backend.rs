//! The device-independent half of `synth`: voice conditioning, the fragment
//! loop and the output file.
//!
//! Both backends -- `crate::pipeline::Pipeline` (wgpu) and
//! `crate::cpu::CpuPipeline` (hand-written host code) -- plug into
//! [`synthesize`] through [`Backend`], so this orchestration exists once. Only
//! the tensor maths, the decode loop and the VAE decode are per-device; keeping
//! them behind the trait is what stops the two paths from drifting apart (the
//! STYLE re-conditioning and the CFM history seed have both drifted before).

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};

use crate::audio;
use crate::pipeline::{Noise, SynthParams, LATENT_DIM};
use crate::qwen_lm::HIDDEN;
use crate::spk;
use crate::synth::Args;
use crate::tasks;
use crate::tok::{self, Tok};

/// Rate the VAE decodes at, and the rate the reference writes (44.1 kHz).
pub const SAMPLE_RATE: u32 = 44100;

/// What one decode loop produced.
pub struct Decoded {
    /// `[n, LATENT_DIM]` audio latents in decode order.
    pub latents: Vec<f32>,
    /// Step the stop head fired at (`None`: it never did, `max_decode_steps` won).
    pub stopped_at: Option<usize>,
}

/// Everything a synthesis run needs that does not depend on the device.
pub struct Job<'a> {
    pub model_dir: &'a Path,
    pub tok: &'a Tok,
    pub req: &'a tasks::SynthRequest,
    pub args: &'a Args,
    pub fragments: &'a [String],
    /// STYLE tasks with more than one fragment re-use chunk 0 as the prompt and
    /// the voice of every later fragment (ming_tts.synthesize).
    pub reference_first_fragment: bool,
}

/// The per-device operations [`synthesize`] needs. The GPU implementations are
/// the same calls the parity tests make, so a backend that satisfies the CLI
/// also satisfies the numeric tests.
pub trait Backend {
    /// Token ids -> `[t, HIDDEN]` input embeddings, before voice conditioning.
    fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>>;
    /// 192-dim CAMPlus embedding -> `HIDDEN`-wide LLM row (spk_head).
    fn project_speaker(&self, spk192: &[f32]) -> Result<Vec<f32>>;
    /// `n` latent patches -> `n * HIDDEN` rows (linear_proj_audio).
    fn project_latent_rows(&self, latent: &[f32], n: usize) -> Result<Vec<f32>>;
    /// 44.1 kHz waveform -> posterior-sampled prompt latent and its patch count.
    fn encode_prompt_latent(
        &self,
        rng: &mut dyn Noise,
        wav44k: &[f32],
    ) -> Result<(Vec<f32>, usize)>;
    /// The autoregressive loop: LLM step -> CFM sample -> aggregator.
    fn decode(
        &mut self,
        embeds: &[f32],
        t: usize,
        prompt_latent: Option<&[f32]>,
        prompt_frames: usize,
        noise: &mut dyn Noise,
        params: &SynthParams,
    ) -> Result<Decoded>;
    /// Latents `[n, LATENT_DIM]` -> waveform at [`SAMPLE_RATE`].
    fn vocode(&self, latents: &[f32], n: usize) -> Result<Vec<f32>>;
}

/// Condition the backend on the requested voice, run every fragment and write
/// the concatenated waveform to `--output`.
pub fn synthesize(job: &Job<'_>, be: &mut dyn Backend, rng: &mut dyn Noise) -> Result<()> {
    let a = job.args;
    let req = job.req;
    println!("fragments: {}", job.fragments.len());

    // ---- voice conditioning ------------------------------------------------
    // A zero speaker embedding wins over --spk_emb_path and is applied after the
    // projection in the reference (BailingMM.prepare_input_embed), so it is just
    // a zero row here.
    let mut spk_rows: Vec<Vec<f32>> = Vec::new();
    if a.use_zero_spk_emb {
        spk_rows.push(vec![0.0; HIDDEN]);
    } else if a.use_spk_emb {
        // One slot per precomputed .pt vector, exactly like
        // ming_tts._speaker_embeddings (last dim must be 192).
        for path in &a.spk_emb_path {
            let v = crate::pt::load_tensor(Path::new(path))?;
            if v.len() != 192 {
                bail!("{path} holds a {}-dim embedding, expected 192", v.len());
            }
            spk_rows.push(be.project_speaker(&v)?);
        }
    }

    let need_prompt = !a.prompt_wav_path.is_empty() && req.prompt_text.is_some();
    let mut prompt_text = req.prompt_text.clone();
    let mut prompt_latent: Option<(Vec<f32>, usize)> = None;
    if !a.prompt_wav_path.is_empty() {
        let (at_44k, for_spk) = read_prompt_wavs(&a.prompt_wav_path, a.use_spk_emb)?;
        if a.use_spk_emb && spk_rows.is_empty() {
            let onnx = job.model_dir.join("campplus.onnx");
            let emb = spk::embed_16k_with(&onnx, &for_spk).context("campplus speaker embedding")?;
            spk_rows.push(be.project_speaker(&emb)?);
        }
        if need_prompt {
            prompt_latent = Some(be.encode_prompt_latent(rng, &at_44k)?);
        }
    }
    if prompt_latent.is_none() {
        prompt_text = None;
    }

    // ---- project the prompt latents into LLM space -------------------------
    let mut latent_rows: Option<Vec<f32>> = match &prompt_latent {
        Some((lat, n)) => Some(be.project_latent_rows(lat, *n)?),
        None => None,
    };
    let mut latent_steps = prompt_latent.as_ref().map_or(0, |(_, n)| *n);

    let params = SynthParams {
        cfg: a.cfg as f32,
        sigma: a.sigma as f32,
        temperature: a.temperature as f32,
        max_decode_steps: a.max_decode_steps,
    };

    let mut out: Vec<f32> = Vec::new();
    for (index, fragment) in job.fragments.iter().enumerate() {
        let layout = tok::build_input_ids(
            job.tok,
            tok::DENSE,
            &tok::InputSeqArgs {
                prompt: &req.prompt,
                text: fragment,
                num_speakers: spk_rows.len(),
                instruction: req.instruction.as_deref(),
                prompt_text: if latent_rows.is_some() {
                    prompt_text.as_deref()
                } else {
                    None
                },
                prompt_latent_steps: latent_steps,
            },
        )?;
        let t = layout.input_ids.len();
        // MING_DUMP_IDS=<file> writes the token ids of each fragment, for
        // diffing the CLI against the reference pipeline.
        if let Ok(path) = std::env::var("MING_DUMP_IDS") {
            let ids: String = layout
                .input_ids
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let path = format!("{path}.{index}");
            std::fs::write(&path, ids)?;
            println!("ids -> {path} ({t} tokens)");
        }

        let mut embeds = be.embed_ids(&layout.input_ids)?;
        ensure!(
            embeds.len() == t * HIDDEN,
            "backend returned {} values for {t} tokens ({HIDDEN} wide)",
            embeds.len()
        );
        for (slot, se) in spk_rows.iter().enumerate() {
            let pos = layout.speaker_positions[slot];
            embeds[pos * HIDDEN..(pos + 1) * HIDDEN].copy_from_slice(se);
        }
        if let Some(rows) = &latent_rows {
            let start = layout.latent_position as usize;
            embeds[start * HIDDEN..(start + latent_steps) * HIDDEN].copy_from_slice(rows);
        }
        if let Ok(path) = std::env::var("MING_DUMP_EMBEDS") {
            let path = format!("{path}.{index}.f32");
            let mut bytes = Vec::with_capacity(embeds.len() * 4);
            for v in &embeds {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&path, &bytes)?;
            println!("embeds -> {path} ({t} x {HIDDEN})");
        }

        let t0 = Instant::now();
        let decoded = be.decode(
            &embeds,
            t,
            prompt_latent.as_ref().map(|(l, _)| l.as_slice()),
            latent_steps,
            rng,
            &params,
        )?;
        let decode_time = t0.elapsed();
        let n = decoded.latents.len() / LATENT_DIM;
        if let Ok(path) = std::env::var("MING_DUMP_LATENTS") {
            let mut bytes = Vec::with_capacity(decoded.latents.len() * 4);
            for v in &decoded.latents {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&path, &bytes)?;
            println!("latents -> {path} ({n} x {LATENT_DIM})");
        }

        let t1 = Instant::now();
        let mut wav = be.vocode(&decoded.latents, n)?;
        let vae_time = t1.elapsed();
        if !a.no_trim_silence {
            wav = audio::trim_trailing_silence(&wav, SAMPLE_RATE);
        }

        if job.reference_first_fragment && index == 0 {
            // ming_tts._prepare_generated_prompt: the chunk just generated
            // (trimmed, like the reference) becomes the acoustic prompt -- and
            // the voice -- of every later fragment.
            let spk16 = audio::resample(&wav, SAMPLE_RATE, 16000);
            let onnx = job.model_dir.join("campplus.onnx");
            let se = spk::embed_16k_with(&onnx, &spk16)
                .context("campplus speaker embedding of the generated prompt")?;
            spk_rows = vec![be.project_speaker(&se)?];
            let (lat, frames) = be.encode_prompt_latent(rng, &wav)?;
            latent_rows = Some(be.project_latent_rows(&lat, frames)?);
            latent_steps = frames;
            prompt_latent = Some((lat, frames));
            prompt_text = Some(fragment.clone());
            println!(
                "  re-conditioning on fragment 0: {frames} prompt patches, {} speaker samples",
                spk16.len()
            );
        }

        println!(
            "  fragment {index}: {n} latents in {:.2}s (stopped_at={:?}), vae in {:.2}s",
            decode_time.as_secs_f64(),
            decoded.stopped_at,
            vae_time.as_secs_f64()
        );
        out.extend_from_slice(&wav);
    }

    audio::write_wav(&a.output, &out, SAMPLE_RATE)
        .with_context(|| format!("write {}", a.output))?;
    println!(
        "wrote {} ({:.2}s @ {SAMPLE_RATE})",
        a.output,
        out.len() as f64 / SAMPLE_RATE as f64
    );
    Ok(())
}

/// Resample each prompt wav once, from its NATIVE rate: one branch goes to
/// 16 kHz for the speaker embedding, the other to 44.1 kHz for the VAE.
/// Chaining 48k -> 44.1k -> 16k is a different filter and measurably changes
/// the embedding.
fn read_prompt_wavs(paths: &[String], want_spk: bool) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut at_44k: Vec<f32> = Vec::new();
    let mut for_spk: Vec<f32> = Vec::new();
    for p in paths {
        let (x, r) = audio::read_wav(p).with_context(|| format!("read prompt {p}"))?;
        if want_spk {
            for_spk.extend_from_slice(&if r != 16000 {
                audio::resample(&x, r, 16000)
            } else {
                x.clone()
            });
        }
        at_44k.extend_from_slice(&if r != SAMPLE_RATE {
            audio::resample(&x, r, SAMPLE_RATE)
        } else {
            x
        });
    }
    Ok((at_44k, for_spk))
}
