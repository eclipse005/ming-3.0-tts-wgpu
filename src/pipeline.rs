//! End-to-end synthesis pipeline (M5): token ids -> LLM decode loop ->
//! flow-matching latent per step -> VAE decode -> waveform.
//!
//! Mirrors BailingMMNativeForConditionalGeneration.sample/generate, including
//! the ordering quirks: the stopping step's latent is still appended, the
//! latent history is seeded from the prompt latent at step 0, and the DiT
//! noise is drawn (2 draws per step) before each step's sampler runs.

use crate::dit::DitEngine;
use crate::gpu::Gpu;
use crate::qwen_lm::LlmEngine;
use crate::vae::VaeEncoder;
use crate::vae_dec::VaeDecoder;
use crate::weights::{bf16_widen, Weights};
use wgpu::Buffer;
use anyhow::{bail, Context, Result};
use std::path::Path;

pub const LATENT_DIM: usize = 64;
pub const PATCH: usize = 1;
pub const HISTORY: usize = 1;
pub const MAX_CACHE_LEN: usize = 512;
pub const CFM_STEPS: usize = 10;

/// Noise source for the sampler. One method call models one `torch.randn`
/// call, because the reference's generator advances its draw counter per call
/// (see rng.rs) -- splitting a draw into two halves would desynchronise it.
pub trait Noise {
    fn draw(&mut self, n: usize) -> Vec<f32>;
}

/// Replays recorded torch.randn draws (MING_NOISE_GOLDEN / parity runs).
pub struct GoldenNoise {
    draws: Vec<Vec<f32>>,
    cursor: usize,
}

impl GoldenNoise {
    pub fn new(draws: Vec<Vec<f32>>) -> Self {
        Self { draws, cursor: 0 }
    }
}

impl Noise for GoldenNoise {
    fn draw(&mut self, n: usize) -> Vec<f32> {
        match self.draws.get(self.cursor) {
            Some(d) => {
                assert_eq!(
                    d.len(),
                    n,
                    "golden draw {} has {} values, the pipeline asked for {n}: \
                     the runs diverged before this point",
                    self.cursor,
                    d.len()
                );
                self.cursor += 1;
                d.clone()
            }
            None => panic!("ran out of golden draws at draw {} (wanted {n} values)", self.cursor),
        }
    }
}

/// Production source: the reference's own generator (rng::Philox).
pub struct Torch { inner: crate::rng::Philox }

impl Torch {
    pub fn new(seed: i64) -> Self {
        Self { inner: crate::rng::Philox::new(seed) }
    }
}

impl Noise for Torch {
    fn draw(&mut self, n: usize) -> Vec<f32> {
        self.inner.draw(n)
    }
}


/// MING_NAN_TRACE=1: report NaN/Inf/absmax for a stage's values.
fn nan_report(label: &str, v: &[f32]) {
    if std::env::var("MING_NAN_TRACE").is_err() {
        return;
    }
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let inf = v.iter().filter(|x| x.is_infinite()).count();
    let max = v.iter().fold(0f32, |a, b| a.max(b.abs()));
    println!("    [nan] {label:14} n={} nan={nan} inf={inf} absmax={max:.4e}", v.len());
}

pub struct SynthParams {
    pub cfg: f32,
    pub sigma: f32,
    pub temperature: f32,
    pub max_decode_steps: usize,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self { cfg: 2.0, sigma: 0.25, temperature: 0.0, max_decode_steps: 1000 }
    }
}

pub struct Pipeline {
    pub gpu: Gpu,
    pub w: Weights,
    /// Kept so callers can report which large-t strategy the adapter picked.
    pub ops: std::rc::Rc<crate::ops::Ops>,
    pub llm: LlmEngine,
    pub dit: DitEngine,
    pub dec: VaeDecoder,
    pub enc: Option<VaeEncoder>,
    /// 192-dim CAMPlus space -> 896-dim LLM space (spk_head).
    spk_w: Buffer,
    spk_b: Buffer,
}

/// What one decode loop produced.
pub struct LoopTrace {
    pub latents: Vec<f32>,
    pub stopped_at: Option<usize>,
}

impl Pipeline {
    /// `pref` is the CLI's `--device` (cuda:0 / igpu / cpu) resolved to a class
    /// of adapter. Everything downstream shares the one device this picks.
    pub fn load_pref(
        model_dir: &Path,
        with_encoder: bool,
        pref: crate::gpu::DevicePref,
    ) -> Result<Self> {
        let w = Weights::load(&model_dir.join("model.safetensors"))?;
        let gpu = Gpu::with_pref(pref)?;
        // One compiled kernel set and one device for every stage.
        let ops = std::rc::Rc::new(crate::ops::Ops::new(gpu.clone()));
        let llm = LlmEngine::load(gpu.clone(), ops.clone(), &w)?;
        let dit = DitEngine::load(gpu.clone(), ops.clone(), &w)?;
        let dec = VaeDecoder::load(gpu.clone(), ops.clone(), &w)?;
        let enc = if with_encoder {
            Some(VaeEncoder::load(gpu.clone(), ops.clone(), &w)?)
        } else {
            None
        };
        let spk_w = gpu.upload_u32(
            &w.get("spk_head.weight")?.to_bf16_packed_u32()?,
            "spk.w",
        );
        let spk_b = gpu.upload_f32(&w.get("spk_head.bias")?.to_f32()?, "spk.b");
        Ok(Self { gpu, w, ops, llm, dit, dec, enc, spk_w, spk_b })
    }

    /// Token ids -> [T, 896] input embeddings (no speaker / latent rows).
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = self.w.get("model.model.embed_tokens.weight")?;
        if t.shape.len() != 2 {
            bail!("embed_tokens is not 2-D");
        }
        let (vocab, dim) = (t.shape[0], t.shape[1]);
        let mut out = vec![0f32; ids.len() * dim];
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= vocab {
                bail!("token id {id} >= vocab {vocab}");
            }
            let off = id * dim * 2;
            for j in 0..dim {
                let lo = t.data[off + 2 * j] as u32;
                let hi = t.data[off + 2 * j + 1] as u32;
                out[i * dim + j] = bf16_widen(lo | (hi << 8));
            }
        }
        Ok(out)
    }

    /// Project one latent patch [64] to a 896-wide LLM embedding (linear_proj_audio).
    pub fn project_latent(&self, latent: &[f32]) -> Result<Vec<f32>> {
        self.dit.agg_forward(latent)
    }

    /// Project a 192-dim CAMPlus embedding into the 896-dim LLM space
    /// (BailingMM.spk_head). A zero speaker embedding is applied AFTER the
    /// projection in the reference, so it is just a zero row here.
    pub fn project_speaker(&self, spk192: &[f32]) -> Result<Vec<f32>> {
        crate::nn::gemv_bf16w_buf(&self.gpu, &self.spk_w, &self.spk_b, spk192, 896, 192)
    }

    /// The autoregressive decode loop. `embeds` is [t, 896].
    #[allow(clippy::too_many_arguments)]
    pub fn decode_loop(
        &mut self,
        embeds: &[f32],
        t: usize,
        prompt_latent: Option<&[f32]>,
        prompt_latent_frames: usize,
        noise: &mut dyn Noise,
        p: &SynthParams,
    ) -> Result<LoopTrace> {
        let t_pre = std::time::Instant::now();
        let mem_before = self.gpu.mem_stats().0;
        let (mut z, mut prob) = self.llm.prefill_full(embeds, t)?;
        nan_report("prefill.z", &z);
        let prefill = t_pre.elapsed();
        let mut dit_time = std::time::Duration::ZERO;
        let mut llm_time = std::time::Duration::ZERO;
        let mut history = vec![0f32; HISTORY * LATENT_DIM];
        if let Some(pl) = prompt_latent {
            let n = prompt_latent_frames.min(HISTORY);
            let src = prompt_latent_frames - n;
            for i in 0..n {
                history[i * LATENT_DIM..(i + 1) * LATENT_DIM]
                    .copy_from_slice(&pl[(src + i) * LATENT_DIM..(src + i + 1) * LATENT_DIM]);
            }
        }

        // The reference decodes against a static 512-position cache:
        //   decode_budget = (MAX_CACHE_LEN - inputs_embeds.shape[1]) // patch
        // (modeling_bailingmm.py). With more prompt than cache it computes a
        // negative budget, so range() is empty and the utterance yields no
        // audio at all. The subtraction is done in usize here, so it has to be
        // saturating: --raw_text (no splitting) plus a long paragraph is enough
        // to reach t >= 512, and wrapping would hand the loop a giant budget
        // and let it write past the KV cache.
        let budget = MAX_CACHE_LEN.saturating_sub(t) / PATCH.max(1);
        if budget == 0 {
            bail!(
                "prompt is {t} tokens, which leaves no room in the {MAX_CACHE_LEN}-token \
                 decode cache: split the text (drop --raw_text / lower --max_length) or \
                 shorten it. The reference silently returns no audio for this input."
            );
        }
        let stop_trace = std::env::var("MING_STOP_TRACE").is_ok();
        let steps = p.max_decode_steps.min(budget);
        let mut latents: Vec<f32> = Vec::new();
        let mut stopped_at = None;
        let mut pos = t;

        for step in 0..steps {
            // The reference draws both up front, in this order, at every step.
            let nz = noise.draw(PATCH * LATENT_DIM);
            let sde = noise.draw(CFM_STEPS * PATCH * LATENT_DIM);
            let t_d = std::time::Instant::now();
            let lat = self.dit.flow_sample_stoch(
                &z, &history, &nz, &sde, p.cfg, p.sigma, p.temperature)?;
            dit_time += t_d.elapsed();
            nan_report("cfm.latent", &lat);
            if stop_trace {
                println!("    step {step:2}: stop_prob {:.6}", prob[1]);
            }
            latents.extend_from_slice(&lat);

            let done = prob[1] > 0.5 && step > 3;
            if done {
                stopped_at = Some(step);
                break;
            }

            let t_l = std::time::Instant::now();
            let emb = self.project_latent(&lat)?;
            nan_report("agg.emb", &emb);
            let (z2, p2) = self.llm.step(&emb, pos)?;
            nan_report("llm.z", &z2);
            llm_time += t_l.elapsed();
            z = z2;
            prob = p2;
            pos += PATCH;

            // shift the history left by PATCH and append the new patch
            let keep = HISTORY - PATCH;
            history.copy_within(PATCH * LATENT_DIM..HISTORY * LATENT_DIM, 0);
            history[keep * LATENT_DIM..].copy_from_slice(&lat);
        }

        let n_latents = latents.len() / LATENT_DIM;
        if std::env::var("MING_MEM").is_ok() {
            let (cum, peak) = self.gpu.mem_stats();
            let (nb, nd) = self.gpu.counters();
            println!(
                "  mem decode    : {:.1} MiB in {} buffers, {} dispatches for {} latents, \
                 peak window {:.1} MiB",
                (cum - mem_before) as f64 / 1048576.0,
                nb,
                nd,
                n_latents,
                peak as f64 / 1048576.0
            );
        }
        if std::env::var("MING_PROFILE").is_ok() {
            println!(
                "    profile: prefill {:?} (t={}) | dit {:?} | llm+agg {:?}",
                prefill, t, dit_time, llm_time
            );
        }
        Ok(LoopTrace { latents, stopped_at })
    }

    /// Latents [n, 64] -> waveform at 44.1 kHz.
    pub fn vocode(&self, latents: &[f32], n: usize) -> Result<Vec<f32>> {
        let before = self.gpu.mem_stats().0;
        let wav = self.dec.decode(latents, n)?;
        if std::env::var("MING_MEM").is_ok() {
            let (cum, peak) = self.gpu.mem_stats();
            println!(
                "  mem vae-decode: {:.1} MiB allocated for {} frames, peak window {:.1} MiB",
                (cum - before) as f64 / 1048576.0,
                n * PATCH,
                peak as f64 / 1048576.0
            );
        }
        Ok(wav)
    }

    /// Project a whole prompt latent, patch by patch (linear_proj_audio).
    pub fn project_latent_rows(&self, latent: &[f32], n: usize) -> Result<Vec<f32>> {
        let mut rows = Vec::with_capacity(n * LATENT_DIM);
        for i in 0..n {
            rows.extend_from_slice(&self.project_latent(&latent[i * LATENT_DIM..(i + 1) * LATENT_DIM])?);
        }
        Ok(rows)
    }

    /// Encode a 44.1 kHz waveform into a prompt latent, exactly like
    /// encode_prompt_latent: front-pad to a whole patch, run the VAE encoder,
    /// then draw the posterior sample with the reference's RNG (torch draws
    /// [1,64,T] while our mean/std are [T,64], so the flat draw is transposed).
    pub fn encode_prompt_latent(
        &self,
        rng: &mut dyn Noise,
        wav44k: &[f32],
    ) -> Result<(Vec<f32>, usize)> {
        let align = 882 * 4; // hop * encoder patch size * ditar patch size
        let mut all = wav44k.to_vec();
        let rem = all.len() % align;
        if rem != 0 {
            let mut padded = vec![0f32; all.len() + align - rem];
            padded[align - rem..].copy_from_slice(&all);
            all = padded;
        }
        let enc = self.enc.as_ref().context("pipeline loaded without the VAE encoder")?;
        let (mean, std, n_patch) = enc.encode_params(&all)?;
        let frames = mean.len() / LATENT_DIM;
        let raw = rng.draw(LATENT_DIM * frames);
        let mut latent = vec![0f32; mean.len()];
        for t in 0..frames {
            for c in 0..LATENT_DIM {
                let i = t * LATENT_DIM + c;
                latent[i] = mean[i] + std[i] * raw[c * frames + t];
            }
        }
        Ok((latent, n_patch))
    }
}

/// The CLI drives every device through this trait; `pipeline::Pipeline` is the
/// wgpu half (see `crate::backend`).
impl crate::backend::Backend for Pipeline {
    fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        Pipeline::embed_ids(self, ids)
    }

    fn project_speaker(&self, spk192: &[f32]) -> Result<Vec<f32>> {
        Pipeline::project_speaker(self, spk192)
    }

    fn project_latent_rows(&self, latent: &[f32], n: usize) -> Result<Vec<f32>> {
        Pipeline::project_latent_rows(self, latent, n)
    }

    fn encode_prompt_latent(
        &self,
        rng: &mut dyn Noise,
        wav44k: &[f32],
    ) -> Result<(Vec<f32>, usize)> {
        Pipeline::encode_prompt_latent(self, rng, wav44k)
    }

    fn decode(
        &mut self,
        embeds: &[f32],
        t: usize,
        prompt_latent: Option<&[f32]>,
        prompt_frames: usize,
        noise: &mut dyn Noise,
        params: &SynthParams,
    ) -> Result<crate::backend::Decoded> {
        let trace = Pipeline::decode_loop(self, embeds, t, prompt_latent, prompt_frames, noise, params)?;
        Ok(crate::backend::Decoded {
            latents: trace.latents,
            stopped_at: trace.stopped_at,
        })
    }

    fn vocode(&self, latents: &[f32], n: usize) -> Result<Vec<f32>> {
        Pipeline::vocode(self, latents, n)
    }
}
