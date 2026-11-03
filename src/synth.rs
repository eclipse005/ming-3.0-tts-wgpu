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

#[derive(clap::Args)]
#[command(rename_all = "snake_case")]
pub struct Args {
    #[arg(long)]
    pub model_path: String,
    #[arg(long)]
    pub task: String,
    #[arg(long)]
    pub text: String,
    #[arg(long)]
    pub output: String,
    /// `cuda:0` (discrete), `igpu` (integrated) or `cpu` (WARP/lavapipe).
    #[arg(long, default_value = "cuda:0")]
    pub device: String,
    #[arg(long, default_value_t = 1895)]
    pub seed: i64,
    #[arg(long)]
    pub language_id: Option<String>,
    #[arg(long)]
    pub instruction: Option<String>,
    #[arg(long)]
    pub instruction_schema: Option<String>,
    #[arg(long)]
    pub speed: Option<String>,
    #[arg(long)]
    pub pitch: Option<String>,
    #[arg(long)]
    pub volume: Option<String>,
    #[arg(long)]
    pub emotion: Option<String>,
    #[arg(long)]
    pub dialect: Option<String>,
    #[arg(long)]
    pub style: Option<String>,
    #[arg(long)]
    pub ip: Option<String>,
    #[arg(long, default_value_t = 2.0)]
    pub cfg: f64,
    #[arg(long, default_value_t = 0.25)]
    pub sigma: f64,
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f64,
    #[arg(long, default_value_t = 24)]
    pub max_length: usize,
    #[arg(long, default_value_t = 1000)]
    pub max_decode_steps: usize,
    #[arg(long, num_args = 1..)]
    pub prompt_wav_path: Vec<String>,
    #[arg(long)]
    pub prompt_text: Option<String>,
    #[arg(long, num_args = 1..)]
    pub spk_emb_path: Vec<String>,
    #[arg(long)]
    pub use_spk_emb: bool,
    #[arg(long)]
    pub use_zero_spk_emb: bool,
    #[arg(long)]
    pub no_text_normalization: bool,
    #[arg(long)]
    pub raw_text: bool,
    #[arg(long)]
    pub no_trim_silence: bool,
}

/// Same split/normalise decision as ming_tts.prepare_text, minus the pynini TN
/// (which the port deliberately does not carry: see PROGRESS.md).
pub(crate) fn prepare_text(tok: &Tok, req: &tasks::SynthRequest, a: &Args) -> Result<Vec<String>> {
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
        // The reference runs TalkerTN (pynini FSTs + regex glue) and then the
        // pynini-free normalisation kit. crate::tn is a port of that chain; its
        // FSTs are the reference's own compiled automata.
        fragments = fragments
            .iter()
            .map(|f| -> Result<String> {
                let normalized = tn_normalize(f)?;
                Ok(text::normalize_text_kit(&normalized))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
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

/// Lazily loaded TalkerTN (the FSTs are ~8 MB; load once per process).
fn tn_normalize(text_in: &str) -> Result<String> {
    use std::sync::OnceLock;
    static TN: OnceLock<std::result::Result<crate::tn::TalkerTn, String>> = OnceLock::new();
    let tn = TN.get_or_init(|| {
        crate::tn::TalkerTn::load(&crate::tn::default_dir()).map_err(|e| e.to_string())
    });
    match tn {
        Ok(tn) => tn.normalize_fragment(text_in),
        Err(e) => anyhow::bail!(
            "text normalisation needs the FST data ({e}); pass --no_text_normalization \
             or set MING_TN_DIR to a directory holding *_tagger.fst / *_verbalizer.fst"
        ),
    }
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
    let mut spk_embs: Vec<Vec<f32>> = Vec::new();
    let mut prompt_latent: Option<(Vec<f32>, usize)> = None;
    let mut prompt_text = req.prompt_text.clone();

    let zero_spk = a.use_zero_spk_emb;
    if zero_spk {
        // The reference still allocates one speaker slot; it is zeroed after the
        // projection. It also takes precedence over --spk_emb_path.
        spk_embs.push(vec![0.0; 192]);
    }

    // Precomputed speaker vectors: one slot per .pt file, exactly like
    // ming_tts._speaker_embeddings (last dim must be 192).
    if a.use_spk_emb && !zero_spk && !a.spk_emb_path.is_empty() {
        for path in &a.spk_emb_path {
            let v = crate::pt::load_tensor(std::path::Path::new(path))?;
            if v.len() != 192 {
                bail!(
                    "{path} holds a {}-dim embedding, expected 192",
                    v.len()
                );
            }
            spk_embs.push(v);
        }
    }

    let need_prompt = !a.prompt_wav_path.is_empty() && prompt_text.is_some();
    // Voice-design (STYLE) tasks that get split re-use the first generated chunk
    // as the acoustic prompt for the remaining fragments (ming_tts.synthesize's
    // reference_first_fragment branch). That needs the VAE encoder even when no
    // --prompt_wav_path was given.
    let fragments = prepare_text(&tok, &req, &a)?;
    let reference_first_fragment = req.taskname == "STYLE" && fragments.len() > 1;
    let needs_encoder = need_prompt || reference_first_fragment;
    let mem = std::env::var("MING_MEM").is_ok();
    if crate::gpu::DevicePref::parse(&a.device)? == crate::gpu::DevicePref::Cpu {
        return crate::cpu::run(&a, model_dir);
    }
    let mut pl = Pipeline::load_pref(
        model_dir,
        needs_encoder,
        crate::gpu::DevicePref::parse(&a.device)?,
    )?;
    println!(
        "device: {} | large-t: {}",
        pl.gpu.describe(),
        if pl.ops.large_t_tile() { "48x64 tile" } else { "12-token window" }
    );
    if mem {
        let (cum, peak) = pl.gpu.mem_stats();
        println!("  mem load      : cumulative {:.1} MiB, peak window {:.1} MiB", cum as f64 / 1048576.0, peak as f64 / 1048576.0);
    }

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
        let all = at_44k;

        if a.use_spk_emb && spk_embs.is_empty() {
            let onnx = model_dir.join("campplus.onnx");
            spk_embs.push(
                spk::embed_16k_with(&onnx, &for_spk).context("campplus speaker embedding")?,
            );
        }

        if need_prompt {
            prompt_latent = Some(encode_prompt_latent(&pl, &mut *rng, &all)?);
        }
    }

    // spk_head: 192 -> 896. Applied before the zeroing, exactly as in
    // BailingMM.prepare_input_embed.
    let mut spk_rows: Vec<Vec<f32>> = if spk_embs.is_empty() {
        Vec::new()
    } else if zero_spk {
        spk_embs.iter().map(|_| vec![0.0; 896]).collect()
    } else {
        spk_embs
            .iter()
            .map(|se| pl.project_speaker(se))
            .collect::<Result<Vec<_>>>()?
    };

    if prompt_latent.is_none() {
        prompt_text = None;
    }

    // ---- project prompt latents into LLM space -----------------------------
    let mut latent_rows: Option<Vec<f32>> = match &prompt_latent {
        Some((lat, n)) => Some(project_latent_rows(&pl, lat, *n)?),
        None => None,
    };
    let mut latent_steps = prompt_latent.as_ref().map_or(0, |(_, n)| *n);

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
                num_speakers: spk_rows.len(),
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
            let path = format!("{path}.{index}");
            std::fs::write(&path, ids)?;
            println!("ids -> {path} ({} tokens)", layout.input_ids.len());
        }
        let mut embeds = pl.embed_ids(&layout.input_ids)?;
        for (slot, se) in spk_rows.iter().enumerate() {
            let pos = layout.speaker_positions[slot];
            embeds[pos * 896..(pos + 1) * 896].copy_from_slice(&se[..]);
        }
        if let Some(rows) = &latent_rows {
            let start = layout.latent_position as usize;
            embeds[start * 896..(start + latent_steps) * 896].copy_from_slice(rows);
        }

        let t = layout.input_ids.len();
        if let Ok(path) = std::env::var("MING_DUMP_EMBEDS") {
            let path = format!("{path}.{index}.f32");
            let mut bytes = Vec::with_capacity(embeds.len() * 4);
            for v in &embeds {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&path, &bytes)?;
            println!("embeds -> {path} ({t} x 896)");
        }
        let t0 = std::time::Instant::now();
        let before = pl.gpu.mem_stats().0;
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
        let (cum_loop, peak_loop) = pl.gpu.mem_stats();
        if mem {
            let (nb, nd) = pl.gpu.counters();
            println!(
                "  mem decode    : {:.1} MiB in {} buffers, {} dispatches for {} latents, peak window {:.1} MiB",
                (cum_loop - before) as f64 / 1048576.0,
                nb,
                nd,
                trace.n_latents,
                peak_loop as f64 / 1048576.0
            );
        }
        let before_vae = cum_loop;
        let mut wav = pl.vocode(&trace.latents, trace.n_latents)?;
        if mem {
            let (cum_vae, peak_vae) = pl.gpu.mem_stats();
            println!(
                "  mem vae-decode: {:.1} MiB allocated for {} frames, peak window {:.1} MiB",
                (cum_vae - before_vae) as f64 / 1048576.0,
                trace.n_latents * 4,
                peak_vae as f64 / 1048576.0
            );
        }
        if !a.no_trim_silence {
            wav = Pipeline::trim_trailing_silence(&wav, 44100);
        }

        if reference_first_fragment && index == 0 {
            // ming_tts._prepare_generated_prompt: take the chunk we just
            // generated (trimmed, like the reference) and use it as the acoustic
            // prompt for every later fragment -- including its speaker embedding.
            let spk16 = audio::resample(&wav, 44100, 16000);
            let onnx = model_dir.join("campplus.onnx");
            let se = spk::embed_16k_with(&onnx, &spk16)
                .context("campplus speaker embedding of the generated prompt")?;
            spk_rows = vec![pl.project_speaker(&se)?];
            let (lat, n) = encode_prompt_latent(&pl, &mut *rng, &wav)?;
            latent_rows = Some(project_latent_rows(&pl, &lat, n)?);
            latent_steps = n;
            prompt_latent = Some((lat, n));
            prompt_text = Some(fragment.clone());
            println!(
                "  re-conditioning on fragment 0: {} prompt patches, {} speaker dim",
                n,
                spk16.len()
            );
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

/// Encode a 44.1 kHz waveform into a prompt latent, exactly like
/// encode_prompt_latent: front-pad to a whole patch, run the VAE encoder, then
/// draw the posterior sample with the reference's RNG (torch draws [1,64,T] while
/// our mean/std are [T,64], so the flat draw is transposed).
fn encode_prompt_latent(
    pl: &Pipeline,
    rng: &mut dyn crate::pipeline::Noise,
    wav44k: &[f32],
) -> Result<(Vec<f32>, usize)> {
    let align = 882 * 4; // hop * encoder patch size * ditar patch size (1)
    let mut all = wav44k.to_vec();
    let rem = all.len() % align;
    if rem != 0 {
        let mut padded = vec![0f32; all.len() + align - rem];
        padded[align - rem..].copy_from_slice(&all);
        all = padded;
    }
    let enc = pl.enc.as_ref().context("pipeline loaded without the VAE encoder")?;
    let (mean, std, n_patch) = enc.encode_params(&all)?;
    let frames = mean.len() / 64;
    let raw = rng.draw(64 * frames);
    let mut latent = vec![0f32; mean.len()];
    for t in 0..frames {
        for c in 0..64 {
            let i = t * 64 + c;
            latent[i] = mean[i] + std[i] * raw[c * frames + t];
        }
    }
    Ok((latent, n_patch))
}

/// linear_proj_audio, patch by patch.
fn project_latent_rows(pl: &Pipeline, latent: &[f32], n: usize) -> Result<Vec<f32>> {
    let mut rows = Vec::with_capacity(n * 896);
    for i in 0..n {
        rows.extend_from_slice(&pl.project_latent(&latent[i * 64..(i + 1) * 64])?);
    }
    Ok(rows)
}
