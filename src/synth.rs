//! `synth` subcommand: the Rust equivalent of `python infer.py`, one task per
//! invocation, same flag surface (see main.rs).
//!
//! This module owns the CLI side only: turning flags into text fragments and
//! picking a backend. The fragment loop itself lives in `crate::backend`, which
//! every device shares.

use crate::backend::{self, Backend};
use crate::pipeline::{GoldenNoise, Noise, Pipeline, Torch};
use crate::tasks;
use crate::text;
use crate::tok::Tok;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// `serde::Deserialize` is for the persistent worker: the HTTP adapter sends
/// one JSON object per request with the same snake_case field names as the CLI.
/// `#[serde(default)]` + `Default` keep both surfaces in sync with clap's
/// defaults, and unknown keys are ignored so a client can send extra metadata.
#[derive(clap::Args, serde::Deserialize)]
#[serde(default)]
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
    /// `cuda:0` (discrete), `igpu` (integrated) or `cpu` (hand-written host
    /// backend, no wgpu at all).
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

/// Everything a single utterance needs before the device is touched.
pub struct Plan {
    req: tasks::SynthRequest,
    fragments: Vec<String>,
    reference_first_fragment: bool,
    /// A request built from a prompt wav + transcript (or a multi-fragment
    /// STYLE task) needs the VAE encoder to be resident on the device.
    pub needs_encoder: bool,
}

pub fn plan(a: &Args) -> Result<Plan> {
    let req = tasks::build_task_generation_kwargs(
        &a.task,
        &a.text,
        a.language_id.as_deref(),
        a.instruction.as_deref(),
        controls(a).as_ref(),
        a.instruction_schema.as_deref(),
        single_prompt_wav(a),
        a.prompt_text.as_deref(),
        a.spk_emb_path.first().map(String::as_str),
        a.use_spk_emb,
        a.use_zero_spk_emb,
        a.cfg,
        a.sigma,
        a.temperature,
    )?;

    // Same split/normalise decision as ming_tts.prepare_text.
    let fragments = prepare_text(&req, a)?;
    // Voice-design (STYLE) tasks that get split re-use the first generated chunk
    // as the acoustic prompt for the remaining fragments
    // (ming_tts.synthesize's reference_first_fragment branch). That needs the
    // VAE encoder even when no --prompt_wav_path was given.
    let reference_first_fragment = req.taskname == "STYLE" && fragments.len() > 1;
    let need_prompt = !a.prompt_wav_path.is_empty() && req.prompt_text.is_some();
    Ok(Plan {
        req,
        fragments,
        reference_first_fragment,
        needs_encoder: need_prompt || reference_first_fragment,
    })
}

/// A loaded model, reusable across utterances. The CLI loads one, runs one
/// utterance and exits; `worker` loads one and keeps it for the whole session.
pub struct Engine {
    model_dir: PathBuf,
    tok: Tok,
    be: Box<dyn Backend>,
}

impl Engine {
    /// `with_encoder` decides once whether the VAE encoder is resident. A
    /// worker passes `true` so a later request with a prompt transcript works.
    pub fn load(a: &Args, with_encoder: bool) -> Result<Self> {
        let model_dir = PathBuf::from(&a.model_path);
        let tok = Tok::load(&model_dir).context("load tokenizer")?;
        let be = load_backend(a, &model_dir, with_encoder)?;
        Ok(Self { model_dir, tok, be })
    }

    pub fn synthesize(&mut self, a: &Args, p: &Plan) -> Result<()> {
        let mut rng = noise_source(a)?;
        let job = backend::Job {
            model_dir: &self.model_dir,
            tok: &self.tok,
            req: &p.req,
            args: a,
            fragments: &p.fragments,
            reference_first_fragment: p.reference_first_fragment,
        };
        backend::synthesize(&job, self.be.as_mut(), rng.as_mut())
    }
}

pub fn run(a: Args) -> Result<()> {
    let p = plan(&a)?;
    let mut engine = Engine::load(&a, p.needs_encoder)?;
    engine.synthesize(&a, &p)
}

/// The style knobs, boxed up only when at least one of them is set.
fn controls(a: &Args) -> Option<tasks::Controls> {
    let c = tasks::Controls {
        speed: a.speed.clone(),
        pitch: a.pitch.clone(),
        volume: a.volume.clone(),
        emotion: a.emotion.clone(),
        dialect: a.dialect.clone(),
        style: a.style.clone(),
        ip: a.ip.clone(),
    };
    let any = c.speed.is_some()
        || c.pitch.is_some()
        || c.volume.is_some()
        || c.emotion.is_some()
        || c.dialect.is_some()
        || c.style.is_some()
        || c.ip.is_some();
    any.then_some(c)
}

/// The reference only forwards a prompt wav to the task builder when exactly one
/// was given; more than one is a cloning prompt that the CLI resamples itself.
fn single_prompt_wav(a: &Args) -> Option<&str> {
    if a.prompt_wav_path.len() == 1 {
        Some(&a.prompt_wav_path[0])
    } else {
        None
    }
}

/// MING_NOISE_GOLDEN=<dir> replays the torch.randn draws recorded for this
/// utterance instead of the native RNG, which is how a full CLI run can be
/// compared sample-for-sample against the Python reference.
fn noise_source(a: &Args) -> Result<Box<dyn Noise>> {
    match std::env::var("MING_NOISE_GOLDEN") {
        Ok(dir) => {
            let mut draws = Vec::new();
            for i in 0.. {
                let p = Path::new(&dir).join(format!("noise_{i:03}.npy"));
                if !p.exists() {
                    break;
                }
                draws.push(crate::golden::read_npy(&p)?.data);
            }
            println!("noise: {} golden draws from {dir}", draws.len());
            Ok(Box::new(GoldenNoise::new(draws)))
        }
        Err(_) => Ok(Box::new(Torch::new(a.seed))),
    }
}

/// `--device` -> backend. `cpu` is the hand-written host path; everything else
/// is wgpu, with the discrete/integrated choice resolved by `DevicePref`.
fn load_backend(a: &Args, model_dir: &Path, needs_encoder: bool) -> Result<Box<dyn Backend>> {
    let pref = crate::gpu::DevicePref::parse(&a.device)?;
    if pref == crate::gpu::DevicePref::Cpu {
        return Ok(Box::new(crate::cpu::load(model_dir)?));
    }
    let pl = Pipeline::load_pref(model_dir, needs_encoder, pref)?;
    println!(
        "device: {} | large-t: {}",
        pl.gpu.describe(),
        if pl.ops.large_t_tile() { "48x64 tile" } else { "12-token window" }
    );
    if std::env::var("MING_MEM").is_ok() {
        let (cum, peak) = pl.gpu.mem_stats();
        println!(
            "  mem load      : cumulative {:.1} MiB, peak window {:.1} MiB",
            cum as f64 / 1048576.0,
            peak as f64 / 1048576.0
        );
    }
    Ok(Box::new(pl))
}

/// Same split/normalise decision as ming_tts.prepare_text.
impl Default for Args {
    /// Mirrors the clap defaults above. The worker deserialises JSON into this
    /// struct, so the two must agree.
    fn default() -> Self {
        Self {
            model_path: String::new(),
            task: "TTS".to_string(),
            text: String::new(),
            output: String::new(),
            device: "cuda:0".to_string(),
            seed: 1895,
            language_id: None,
            instruction: None,
            instruction_schema: None,
            speed: None,
            pitch: None,
            volume: None,
            emotion: None,
            dialect: None,
            style: None,
            ip: None,
            cfg: 2.0,
            sigma: 0.25,
            temperature: 0.0,
            max_length: 24,
            max_decode_steps: 1000,
            prompt_wav_path: Vec::new(),
            prompt_text: None,
            spk_emb_path: Vec::new(),
            use_spk_emb: false,
            use_zero_spk_emb: false,
            no_text_normalization: false,
            raw_text: false,
            no_trim_silence: false,
        }
    }
}

fn prepare_text(req: &tasks::SynthRequest, a: &Args) -> Result<Vec<String>> {
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
fn preserve_multilingual_text(language_id: &str) -> bool {
    if language_id.trim().is_empty() {
        return false;
    }
    let lang = language_id.trim().to_lowercase().replace('_', "-");
    !matches!(lang.as_str(), "zh" | "zh-cn" | "chinese" | "en" | "english")
}
