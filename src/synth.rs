//! `synth` subcommand: the Rust equivalent of `python infer.py`, one task per
//! invocation, same flag surface (see main.rs).

use crate::audio;
use crate::pipeline::{Pipeline, SynthParams};
use crate::spk;
use crate::tasks;
use crate::text;
use crate::tok;
use crate::tok::Tok;
use anyhow::{bail, Context, Result};
use std::path::Path;

pub struct Args {
    pub model_path: String,
    pub task: String,
    pub text: String,
    pub output: String,
    pub seed: i64,
    pub language_id: Option<String>,
    pub instruction: Option<String>,
    pub instruction_schema: Option<String>,
    pub speed: Option<String>,
    pub pitch: Option<String>,
    pub volume: Option<String>,
    pub emotion: Option<String>,
    pub dialect: Option<String>,
    pub style: Option<String>,
    pub ip: Option<String>,
    pub cfg: f64,
    pub sigma: f64,
    pub temperature: f64,
    pub max_length: usize,
    pub max_decode_steps: usize,
    pub prompt_wav_path: Vec<String>,
    pub prompt_text: Option<String>,
    pub spk_emb_path: Vec<String>,
    pub use_spk_emb: bool,
    pub use_zero_spk_emb: bool,
    pub no_text_normalization: bool,
    pub raw_text: bool,
    pub no_trim_silence: bool,
}

/// Same split/normalise decision as ming_tts.prepare_text, minus the pynini TN
/// (which the port deliberately does not carry: see PROGRESS.md).
fn prepare_text(tok: &Tok, req: &tasks::SynthRequest, a: &Args) -> Result<Vec<String>> {
    let _ = tok;
    if a.raw_text {
        return Ok(vec![req.text.clone()]);
    }
    let multilingual = req
        .text_language
        .as_deref()
        .map(preserve_multilingual_text)
        .unwrap_or(false);
    let prepared_src = text::preprosess_text(&req.text);
    let base = if multilingual {
        prepared_src.trim().to_string()
    } else {
        text::add_period(&prepared_src)
    };
    let mut fragments = text::cut_text_by_semantic_length(&base, a.max_length);
    if !a.no_text_normalization && !multilingual {
        // The reference runs a pynini FST normaliser for non-multilingual
        // text. We only implement the pynini-free normalisation kit, which
        // is enough for text without digits/symbols.
        fragments = fragments
            .iter()
            .map(|f| text::normalize_text_kit(f))
            .filter(|f| !f.trim().is_empty())
            .collect();
    } else {
        fragments.retain(|f| !f.trim().is_empty());
    }
    if fragments.is_empty() {
        bail!("text produced no synthesizable fragments");
    }
    Ok(fragments)
}

/// Port of tts_tasks.preserve_multilingual_text: explicit non-zh/en languages
/// bypass Chinese/English text normalisation.
pub fn preserve_multilingual_text(language_id: &str) -> bool {
    if language_id.trim().is_empty() {
        return false;
    }
    let lang = language_id.trim().to_lowercase().replace('_', "-");
    !matches!(lang.as_str(), "zh" | "zh-cn" | "chinese" | "en" | "english")
}

pub fn run(a: Args) -> Result<()> {
    let model_dir = Path::new(&a.model_path);
    let tok = Tok::load(model_dir).context("load tokenizer")?;

    let controls = {
        let c = tasks::Controls {
            speed: a.speed.clone(),
            pitch: a.pitch.clone(),
            volume: a.volume.clone(),
            emotion: a.emotion.clone(),
            dialect: a.dialect.clone(),
            style: a.style.clone(),
            ip: a.ip.clone(),
        };
        if c.speed.is_none() && c.pitch.is_none() && c.volume.is_none()
            && c.emotion.is_none() && c.dialect.is_none() && c.style.is_none()
            && c.ip.is_none() {
            None
        } else {
            Some(c)
        }
    };

    let single_wav = if a.prompt_wav_path.len() == 1 {
        Some(a.prompt_wav_path[0].as_str())
    } else {
        None
    };
    let req = tasks::build_task_generation_kwargs(
        &a.task,
        &a.text,
        a.language_id.as_deref(),
        a.instruction.as_deref(),
        controls.as_ref(),
        a.instruction_schema.as_deref(),
        single_wav,
        a.prompt_text.as_deref(),
        a.spk_emb_path.first().map(|s| s.as_str()),
        a.use_spk_emb,
        a.use_zero_spk_emb,
        a.cfg,
        a.sigma,
        a.temperature,
    )?;

    // MING_NOISE_GOLDEN=<dir> replays the torch.randn draws recorded for this
    // utterance instead of the native RNG, which is how a full CLI run can be
    // compared sample-for-sample against the Python reference.
    let mut rng: Box<dyn crate::pipeline::Noise> = match std::env::var("MING_NOISE_GOLDEN") {
        Ok(dir) => {
            let mut draws = Vec::new();
            for i in 0.. {
                let p = Path::new(&dir).join(format!("noise_{i:03}.npy"));
                if !p.exists() {
                    break;
                }
                let npy = crate::golden::read_npy(&p)?;
                draws.push(npy.data);
            }
            println!("noise: {} golden draws from {dir}", draws.len());
            Box::new(crate::pipeline::GoldenNoise::new(draws))
        }
        Err(_) => Box::new(crate::pipeline::Torch::new(a.seed)),
    };

    // ---- voice conditioning ------------------------------------------------
    let mut spk_emb: Option<Vec<f32>> = None;
    let mut prompt_latent: Option<(Vec<f32>, usize)> = None;
    let mut prompt_text = req.prompt_text.clone();

    let zero_spk = a.use_zero_spk_emb;
    if a.use_zero_spk_emb {
        // The reference still allocates one speaker slot; it is just zeroed
        // after the projection.
        spk_emb = Some(vec![0.0; 192]);
    }

    if !a.spk_emb_path.is_empty() && a.use_spk_emb {
        bail!("--spk_emb_path needs a .pt reader, which this port does not ship yet; \
               use --prompt_wav_path with --use_spk_emb instead");
    }

    let need_prompt = !a.prompt_wav_path.is_empty() && prompt_text.is_some();
    let needs_encoder = need_prompt;
    let mut pl = Pipeline::load(model_dir, needs_encoder)?;

    if !a.prompt_wav_path.is_empty() {
        // The reference resamples each prompt wav once, from its NATIVE rate:
        // one branch goes to 16 kHz for the speaker embedding, the other to the
        // VAE's 44.1 kHz. Chaining 48k -> 44.1k -> 16k is a different filter and
        // measurably changes the embedding.
        let mut at_44k: Vec<f32> = Vec::new();
        let mut for_spk: Vec<f32> = Vec::new();
        for p in &a.prompt_wav_path {
            let (x, r) = audio::read_wav(p).with_context(|| format!("read prompt {p}"))?;
            if a.use_spk_emb {
                for_spk.extend_from_slice(&if r != 16000 {
                    audio::resample(&x, r, 16000)
                } else {
                    x.clone()
                });
            }
            at_44k.extend_from_slice(&if r != 44100 { audio::resample(&x, r, 44100) } else { x });
        }
        let mut all = at_44k;

        if a.use_spk_emb && spk_emb.is_none() {
            let onnx = model_dir.join("campplus.onnx");
            spk_emb = Some(
                spk::embed_16k_with(&onnx, &for_spk).context("campplus speaker embedding")?,
            );
        }

        if need_prompt {
            // pad at the FRONT to a multiple of hop * patch * dtar patch
            let align = 882 * 4 * 1;
            let rem = all.len() % align;
            if rem != 0 {
                let mut padded = vec![0f32; all.len() + align - rem];
                padded[align - rem..].copy_from_slice(&all);
                all = padded;
            }
            let enc = pl.enc.as_ref().expect("encoder loaded");
            let (mean, std, n_patch) = enc.encode_params(&all)?;
            // posterior.sample(): torch draws one tensor of shape [1, 64, T], so
            // the flat stream is (channel, frame) ordered while our mean/std are
            // (frame, channel); transpose to line the samples up.
            let frames = mean.len() / 64;
            let raw = rng.draw(64 * frames);
            let mut latent = vec![0f32; mean.len()];
            for t in 0..frames {
                for c in 0..64 {
                    let i = t * 64 + c;
                    latent[i] = mean[i] + std[i] * raw[c * frames + t];
                }
            }
            prompt_latent = Some((latent, n_patch));
        }
    }

    // spk_head: 192 -> 896. Applied before the zeroing, exactly as in
    // BailingMM.prepare_input_embed.
    let spk_row: Option<Vec<f32>> = match &spk_emb {
        None => None,
        Some(se) => {
            if zero_spk {
                Some(vec![0.0; 896])
            } else {
                Some(pl.project_speaker(se)?)
            }
        }
    };

    if prompt_latent.is_none() {
        prompt_text = None;
    }

    // ---- project prompt latents into LLM space -----------------------------
    let mut latent_rows: Option<Vec<f32>> = None;
    let mut latent_steps = 0usize;
    if let Some((pl_lat, n)) = &prompt_latent {
        let mut rows = Vec::with_capacity(n * 896);
        for i in 0..*n {
            let row = pl.project_latent(&pl_lat[i * 64..(i + 1) * 64])?;
            rows.extend_from_slice(&row);
        }
        latent_rows = Some(rows);
        latent_steps = *n;
    }

    let fragments = prepare_text(&tok, &req, &a)?;
    println!("fragments: {}", fragments.len());

    let params = SynthParams {
        cfg: a.cfg as f32,
        sigma: a.sigma as f32,
        temperature: a.temperature as f32,
        max_decode_steps: a.max_decode_steps,
    };

    let mut out: Vec<f32> = Vec::new();
    for (index, fragment) in fragments.iter().enumerate() {
        let layout = tok::build_input_ids(
            &tok,
            tok::DENSE,
            &tok::InputSeqArgs {
                prompt: &req.prompt,
                text: fragment,
                num_speakers: spk_row.is_some() as usize,
                instruction: req.instruction.as_deref(),
                prompt_text: if latent_rows.is_some() { prompt_text.as_deref() } else { None },
                prompt_latent_steps: latent_steps,
            },
        )?;
        // MING_DUMP_IDS=<file> writes the token ids and input embeddings of the
        // first fragment, for diffing the CLI against the reference pipeline.
        if let Ok(path) = std::env::var("MING_DUMP_IDS") {
            let ids: String = layout
                .input_ids
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            std::fs::write(&path, ids)?;
            println!("ids -> {path} ({} tokens)", layout.input_ids.len());
        }
        let mut embeds = pl.embed_ids(&layout.input_ids)?;
        if let Some(se) = &spk_row {
            let pos = layout.speaker_positions[0];
            embeds[pos * 896..(pos + 1) * 896].copy_from_slice(&se[..]);
        }
        if let Some(rows) = &latent_rows {
            let start = layout.latent_position as usize;
            embeds[start * 896..(start + latent_steps) * 896].copy_from_slice(rows);
        }

        let t = layout.input_ids.len();
        let t0 = std::time::Instant::now();
        let trace = pl.decode_loop(
            &embeds,
            t,
            prompt_latent.as_ref().map(|(l, _)| l.as_slice()),
            latent_steps,
            &mut *rng,
            &params,
        )?;
        println!(
            "  fragment {}: {} latents in {:?} (stopped_at={:?})",
            index,
            trace.n_latents,
            t0.elapsed(),
            trace.stopped_at
        );
        let mut wav = pl.vocode(&trace.latents, trace.n_latents)?;
        if !a.no_trim_silence {
            wav = Pipeline::trim_trailing_silence(&wav, 44100);
        }
        out.extend_from_slice(&wav);
    }

    audio::write_wav(&a.output, &out, 44100)
        .with_context(|| format!("write {}", a.output))?;
    println!(
        "wrote {} ({:.2}s @ 44100)",
        a.output,
        out.len() as f64 / 44100.0
    );
    Ok(())
}
