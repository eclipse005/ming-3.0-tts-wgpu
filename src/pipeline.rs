//! End-to-end synthesis pipeline (M5): token ids -> LLM decode loop ->
//! flow-matching latent per step -> VAE decode -> waveform.
//!
//! Mirrors BailingMMNativeForConditionalGeneration.sample/generate, including
//! the ordering quirks: the stopping step's latent is still appended, the
//! latent history is seeded from the prompt latent at step 0, and the DiT
//! noise is drawn (2 draws per step) before each step's sampler runs.

use crate::dit::DitEngine;
use crate::gpu::Gpu;
use crate::qwen_lm::{LlmEngine, HIDDEN};
use crate::vae::VaeEncoder;
use crate::vae_dec::VaeDecoder;
use crate::weights::{bf16_widen, Weights};
use anyhow::{bail, Context, Result};
use std::path::Path;

pub const LATENT_DIM: usize = 64;
pub const PATCH: usize = 1;
pub const HISTORY: usize = 1;
pub const MAX_CACHE_LEN: usize = 512;
pub const CFM_STEPS: usize = 10;

/// Where the two per-step torch.randn draws come from.
pub trait Noise {
    /// (patch, latent) noise for the sampler start.
    fn noise(&mut self) -> Vec<f32>;
    /// (steps, patch, latent) SDE noise.
    fn sde(&mut self) -> Vec<f32>;
}

/// Records the draws so a parity run can assert on them.
pub struct GoldenNoise {
    pub draws: Vec<Vec<f32>>,
    pub cursor: usize,
}

impl GoldenNoise {
    pub fn new(draws: Vec<Vec<f32>>) -> Self {
        Self { draws, cursor: 0 }
    }
    fn take(&mut self) -> Vec<f32> {
        let d = self.draws.get(self.cursor).cloned().unwrap_or_default();
        self.cursor += 1;
        d
    }
}

impl Noise for GoldenNoise {
    fn noise(&mut self) -> Vec<f32> {
        self.take()
    }
    fn sde(&mut self) -> Vec<f32> {
        self.take()
    }
}

/// xoshiro256** + Box-Muller, the native (non-torch) RNG.
pub struct Xoshiro {
    s: [u64; 4],
    spare: Option<f32>,
}

impl Xoshiro {
    pub fn new(seed: u64) -> Self {
        // splitmix64 expansion so any seed works, including 0
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Self { s: [next(), next(), next(), next()], spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    fn uniform(&mut self) -> f32 {
        // (0, 1] like curand_uniform
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        if u == 0.0 { f32::EPSILON } else { u }
    }

    fn normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f32::consts::TAU * u2;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
}

impl Noise for Xoshiro {
    fn noise(&mut self) -> Vec<f32> {
        (0..PATCH * LATENT_DIM).map(|_| self.normal()).collect()
    }
    fn sde(&mut self) -> Vec<f32> {
        (0..CFM_STEPS * PATCH * LATENT_DIM).map(|_| self.normal()).collect()
    }
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
    pub llm: LlmEngine,
    pub dit: DitEngine,
    pub dec: VaeDecoder,
    pub enc: Option<VaeEncoder>,
}

/// Trace of one decode loop, kept for parity assertions.
pub struct LoopTrace {
    pub latents: Vec<f32>,
    pub n_latents: usize,
    pub z_steps: Vec<Vec<f32>>,
    pub stop_probs: Vec<f32>,
    pub stopped_at: Option<usize>,
}

impl Pipeline {
    pub fn load(model_dir: &Path, with_encoder: bool) -> Result<Self> {
        let w = Weights::load(&model_dir.join("model.safetensors"))?;
        let gpu = Gpu::new()?;
        let llm = LlmEngine::load(gpu.clone(), &w)?;
        let dit = DitEngine::load(gpu.clone(), &w)?;
        let dec = VaeDecoder::load(gpu.clone(), &w)?;
        let enc = if with_encoder {
            Some(VaeEncoder::load(gpu.clone(), &w)?)
        } else {
            None
        };
        Ok(Self { gpu, w, llm, dit, dec, enc })
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
        let (mut z, mut prob) = self.llm.prefill_full(embeds, t)?;
        let mut history = vec![0f32; HISTORY * LATENT_DIM];
        if let Some(pl) = prompt_latent {
            let n = prompt_latent_frames.min(HISTORY);
            let src = prompt_latent_frames - n;
            for i in 0..n {
                history[i * LATENT_DIM..(i + 1) * LATENT_DIM]
                    .copy_from_slice(&pl[(src + i) * LATENT_DIM..(src + i + 1) * LATENT_DIM]);
            }
        }

        let budget = (MAX_CACHE_LEN - t) / PATCH.max(1);
        let steps = p.max_decode_steps.min(budget);
        let mut latents: Vec<f32> = Vec::new();
        let mut z_steps = Vec::new();
        let mut probs = Vec::new();
        let mut stopped_at = None;
        let mut pos = t;

        for step in 0..steps {
            let nz = noise.noise();
            let sde = noise.sde();
            let lat = self.dit.flow_sample_stoch(
                &z, &history, &nz, &sde, p.cfg, p.sigma, p.temperature)?;
            z_steps.push(z.clone());
            probs.push(prob[1]);
            latents.extend_from_slice(&lat);

            let done = prob[1] > 0.5 && step > 3;
            if done {
                stopped_at = Some(step);
                break;
            }

            let emb = self.project_latent(&lat)?;
            let (z2, p2) = self.llm.step(&emb, pos)?;
            z = z2;
            prob = p2;
            pos += PATCH;

            // shift the history left by PATCH and append the new patch
            let keep = HISTORY - PATCH;
            history.copy_within(PATCH * LATENT_DIM..HISTORY * LATENT_DIM, 0);
            history[keep * LATENT_DIM..].copy_from_slice(&lat);
        }

        let n_latents = latents.len() / LATENT_DIM;
        Ok(LoopTrace {
            latents,
            n_latents,
            z_steps,
            stop_probs: probs,
            stopped_at,
        })
    }

    /// Latents [n, 64] -> waveform at 44.1 kHz.
    pub fn vocode(&self, latents: &[f32], n: usize) -> Result<Vec<f32>> {
        self.dec.decode(latents, n)
    }

    /// Trim trailing silence exactly like ming_tts.trim_trailing_silence.
    pub fn trim_trailing_silence(speech: &[f32], sample_rate: usize) -> Vec<f32> {
        let frame = sample_rate / 10;
        let tail = sample_rate * 3 / 10;
        if frame == 0 || speech.len() < frame {
            return speech[..speech.len().min(tail)].to_vec();
        }
        let num_frames = (speech.len() - frame) / frame + 1;
        let body = &speech[..num_frames * frame];
        let mut scores = vec![0f32; num_frames];
        for (i, s) in scores.iter_mut().enumerate() {
            let seg = &body[i * frame..(i + 1) * frame];
            *s = seg.iter().map(|v| v.abs()).sum::<f32>() / frame as f32;
        }
        let mut index = num_frames as isize - 1;
        while index >= 0 && scores[index as usize] <= 1e-3 {
            index -= 1;
        }
        if index < 0 {
            return speech[..speech.len().min(tail)].to_vec();
        }
        let end = (index as usize * frame + frame + tail).min(speech.len());
        speech[..end].to_vec()
    }
}

/// Convenience for callers that only have a model dir.
pub fn load(model_dir: &str, with_encoder: bool) -> Result<Pipeline> {
    Pipeline::load(Path::new(model_dir), with_encoder)
        .with_context(|| format!("load pipeline from {model_dir}"))
}
