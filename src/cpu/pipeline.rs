//! The CPU decode loop: LLM step, CFM sample, aggregator, and voice
//! conditioning (prompt encoding, speaker projection).

use anyhow::{bail, Result};

use super::dit::{CpuDit, DIT_STEPS};
use super::llm::CpuLlm;
use super::tensor::{CpuLin, Mat};
use super::vae::{CpuVaeDec, CpuVaeEnc, ENC_IN_DIM, ENC_PATCH};
use super::ZC;
use crate::qwen_lm::{HIDDEN, KV_CAP};
use crate::weights::Weights;

pub struct CpuPipeline {
    pub llm: CpuLlm,
    pub dit: CpuDit,
    pub dec: CpuVaeDec,
    pub enc: CpuVaeEnc,
    spk: CpuLin,
    embed: Mat,
}

impl CpuPipeline {
    pub fn load(w: &Weights) -> Result<Self> {
        Ok(Self {
            llm: CpuLlm::load(w)?,
            dit: CpuDit::load(w)?,
            dec: CpuVaeDec::load(w)?,
            enc: CpuVaeEnc::load(w)?,
            spk: CpuLin::load(w, "spk_head", HIDDEN, 192)?,
            embed: Mat::load(w, "model.model.embed_tokens.weight")?,
        })
    }

    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let dim = self.embed.cols;
        let mut out = vec![0.0f32; ids.len() * dim];
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.embed.rows {
                bail!("token id {id} >= vocab {}", self.embed.rows);
            }
            out[i * dim..(i + 1) * dim]
                .copy_from_slice(&self.embed.data[id * dim..(id + 1) * dim]);
        }
        Ok(out)
    }

    /// Prompt waveform -> posterior sample, exactly like synth.rs: pad to a whole
    /// encoder patch, encode, then draw eps transposed (torch draws [1,64,T]).
    pub fn encode_prompt_latent(
        &self,
        rng: &mut dyn crate::pipeline::Noise,
        wav44k: &[f32],
    ) -> Result<(Vec<f32>, usize)> {
        let align = ENC_IN_DIM * ENC_PATCH;
        let mut all = wav44k.to_vec();
        let rem = all.len() % align;
        if rem != 0 {
            let mut padded = vec![0f32; all.len() + align - rem];
            padded[align - rem..].copy_from_slice(&all);
            all = padded;
        }
        let (mean, std, n) = self.enc.encode_params(&all)?;
        let frames = mean.len() / ZC;
        let raw = rng.draw(ZC * frames);
        let mut latent = vec![0f32; mean.len()];
        for t in 0..frames {
            for c in 0..ZC {
                let i = t * ZC + c;
                latent[i] = mean[i] + std[i] * raw[c * frames + t];
            }
        }
        Ok((latent, n))
    }

    /// linear_proj_audio (the DiT aggregator), patch by patch.
    pub fn project_latent_rows(&self, latent: &[f32], n: usize) -> Vec<f32> {
        let mut rows = Vec::with_capacity(n * HIDDEN);
        for i in 0..n {
            rows.extend_from_slice(&self.dit.aggregate(&latent[i * ZC..(i + 1) * ZC]));
        }
        rows
    }

    /// 192-dim CAMPlus embedding -> 896-dim LLM row (spk_head).
    pub fn project_speaker(&self, spk: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; HIDDEN];
        self.spk.linear(spk, 1, &mut out);
        out
    }

    /// The autoregressive loop: LLM step -> CFM sample -> aggregator, until the
    /// stop head fires (same ordering and quirks as pipeline.rs).
    #[allow(clippy::too_many_arguments)] // numeric knobs mirror the wgpu call
    pub fn decode_loop(
        &mut self,
        embeds: &[f32],
        t: usize,
        prompt_latent: Option<&[f32]>,
        prompt_frames: usize,
        noise: &mut dyn crate::pipeline::Noise,
        cfg: f32,
        sigma: f32,
        temperature: f32,
        max_steps: usize,
    ) -> Result<Vec<f32>> {
        let budget = KV_CAP.saturating_sub(t);
        if budget == 0 {
            bail!("prompt is {t} tokens, no room in the {KV_CAP}-token decode cache");
        }
        let steps = max_steps.min(budget);
        let (mut z, mut prob) = self.llm.prefill(embeds, t)?;
        // Same conditioning as the wgpu path: the CFM history starts from the
        // prompt's last latent frame, not from silence.
        let mut history = vec![0.0f32; ZC];
        if let Some(pl) = prompt_latent {
            if prompt_frames > 0 {
                let start = (prompt_frames - 1) * ZC;
                history.copy_from_slice(&pl[start..start + ZC]);
            }
        }
        let mut latents: Vec<f32> = Vec::new();
        for (step, pos) in (t..).take(steps).enumerate() {
            let nz = noise.draw(ZC);
            let sde = noise.draw(DIT_STEPS * ZC);
            let lat = self
                .dit
                .sample(&z, &history, &nz, &sde, cfg, sigma, temperature);
            latents.extend_from_slice(&lat);
            if prob[1] > 0.5 && step > 3 {
                break;
            }
            let emb = self.dit.aggregate(&lat);
            let (z2, p2) = self.llm.step(&emb, pos)?;
            z = z2;
            prob = p2;
            history.copy_from_slice(&lat);
        }
        Ok(latents)
    }

    pub fn vocode(&self, latents: &[f32]) -> Vec<f32> {
        let n = latents.len() / ZC;
        self.dec.decode(latents, n)
    }
}
