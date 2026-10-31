//! Qwen2.5-0.5B LLM engine on wgpu: prefill + incremental GQA decode with a
//! GPU-resident f32 KV cache, driving the Ming-TTS stop head.
//!
//! Parity contract (M1): given the same prefill embeds and the same per-step
//! decode inputs as the Python pipeline, every step's post-final-norm hidden
//! and stop probability match (rel_l2 < 1e-5), and the stop step is identical.

use crate::gpu::Gpu;
use crate::ops::{AttendParams, Lin, Ops, RopeTables};
use crate::weights::Weights;
use anyhow::{bail, Result};
use std::sync::Arc;
use wgpu::Buffer;

pub const HIDDEN: usize = 896;
pub const LAYERS: usize = 24;
pub const Q_HEADS: usize = 14;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const KV_CAP: usize = 512;
pub const FF: usize = 4864;
pub const ROPE_THETA: f64 = 1_000_000.0;

struct Layer {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    gate: Lin,
    up: Lin,
    down: Lin,
    norm1: Buffer,
    norm2: Buffer,
    kc: Buffer,
    vc: Buffer,
}

pub struct LlmEngine {
    gpu: Gpu,
    layers: Vec<Layer>,
    norm_final: Buffer,
    stop_host_w: Vec<f32>,
    stop_host_b: Vec<f32>,
    rope: RopeTables,
    ops: Arc<Ops>,
}

impl LlmEngine {
    pub fn load(gpu: Gpu, ops: Arc<Ops>, w: &Weights) -> Result<Self> {
        let p = |name: &str| w.get(name);
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let base = format!("model.model.layers.{i}");
            let lin = |what: &str, rows: usize, inf: usize| -> Result<Lin> {
                Lin::load(&gpu, w, &format!("{base}.{what}"), rows, inf)
            };
            let norm = |what: &str| -> Result<Buffer> {
                let t = p(&format!("{base}.{what}.weight"))?;
                Ok(gpu.upload_f32(&t.to_f32()?, "norm"))
            };
            let kc = gpu.storage((KV_HEADS * KV_CAP * HEAD_DIM * 4) as u64, "kv.k");
            let vc = gpu.storage((KV_HEADS * KV_CAP * HEAD_DIM * 4) as u64, "kv.v");
            layers.push(Layer {
                q: lin("self_attn.q_proj", HIDDEN, HIDDEN)?,
                k: lin("self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN)?,
                v: lin("self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN)?,
                o: lin("self_attn.o_proj", HIDDEN, HIDDEN)?,
                gate: lin("mlp.gate_proj", FF, HIDDEN)?,
                up: lin("mlp.up_proj", FF, HIDDEN)?,
                down: lin("mlp.down_proj", HIDDEN, FF)?,
                norm1: norm("input_layernorm")?,
                norm2: norm("post_attention_layernorm")?,
                kc,
                vc,
            });
        }
        let nf = p("model.model.norm.weight")?.to_f32()?;
        let norm_final = gpu.upload_f32(&nf, "norm.final");
        let stop_host_w = p("stop_head.weight")?.to_f32()?;
        let stop_host_b = p("stop_head.bias")?.to_f32()?;
        let rope = RopeTables::new(&gpu, ROPE_THETA, HEAD_DIM, KV_CAP);

        Ok(Self {
            gpu,
            layers,
            norm_final,
            stop_host_w,
            stop_host_b,
            rope,
            ops,
        })
    }

    fn mrow(&self, lin: &Lin, x: &Buffer, t: usize) -> Result<Buffer> {
        self.ops.linear(lin, x, t)
    }

    fn rope_inplace(&self, q: &Buffer, n_heads: usize, tokens: usize, pos_base: usize) {
        // 0 = Qwen2 rotate-half
        self.ops.rope(&self.rope, q, n_heads, 0, pos_base, tokens);
    }

    fn kv_write(&self, src: &Buffer, cache: &Buffer, tokens: usize, pos_base: usize) {
        self.ops.kv_write(src, cache, tokens, pos_base, KV_CAP, KV_HEADS);
    }

    fn attend(
        &self,
        q: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        t: usize,
        q_pos_base: usize,
    ) -> Result<Buffer> {
        Ok(self.ops.attend(
            &AttendParams::causal(t, q_pos_base, Q_HEADS / KV_HEADS, HEAD_DIM, HIDDEN, KV_CAP),
            q,
            kc,
            vc,
            Q_HEADS,
        ))
    }

    fn silu_mul(&self, g: &Buffer, u: &Buffer, n: usize) -> Buffer {
        self.ops.silu_mul(g, u, n)
    }

    fn add(&self, a: &Buffer, b: &Buffer, n: usize) -> Buffer {
        self.ops.add(a, b, n)
    }

    fn rms(&self, x: &Buffer, w: &Buffer, tokens: usize) -> Buffer {
        self.ops.rms(x, w, tokens, HIDDEN)
    }

    fn transformer(&self, x: Buffer, t: usize, pos_base: usize) -> Result<Buffer> {
        let mut h = x;
        for layer in &self.layers {
            let a = self.rms(&h, &layer.norm1, t);
            let q = self.mrow(&layer.q, &a, t)?;
            let k = self.mrow(&layer.k, &a, t)?;
            let v = self.mrow(&layer.v, &a, t)?;
            self.rope_inplace(&q, Q_HEADS, t, pos_base);
            self.rope_inplace(&k, KV_HEADS, t, pos_base);
            self.kv_write(&k, &layer.kc, t, pos_base);
            self.kv_write(&v, &layer.vc, t, pos_base);
            let attn = self.attend(&q, &layer.kc, &layer.vc, t, pos_base)?;
            let o = self.mrow(&layer.o, &attn, t)?;
            h = self.add(&h, &o, t * HIDDEN);
            let a2 = self.rms(&h, &layer.norm2, t);
            let g = self.mrow(&layer.gate, &a2, t)?;
            let up = self.mrow(&layer.up, &a2, t)?;
            let sil = self.silu_mul(&g, &up, t * FF);
            let d = self.mrow(&layer.down, &sil, t)?;
            h = self.add(&h, &d, t * HIDDEN);
        }
        Ok(h)
    }

    fn last_hidden(&self, h: &Buffer, t: usize) -> Result<(Vec<f32>, [f32; 2])> {
        // read the last token's residual stream, apply final norm, stop head
        let x_last = self
            .gpu
            .read_f32_offset(h, HIDDEN, ((t - 1) * HIDDEN * 4) as u64)?;
        let xn = self.gpu.upload_f32(&x_last, "z.x");
        let z = self.rms(&xn, &self.norm_final, 1);
        let zc = self.gpu.read_f32(&z, HIDDEN)?;
        let mut logits = [0f32; 2];
        for r in 0..2 {
            let mut s = self.stop_host_b[r];
            for (i, &v) in zc.iter().enumerate() {
                s += self.stop_host_w[r * HIDDEN + i] * v;
            }
            logits[r] = s;
        }
        let mx = logits[0].max(logits[1]);
        let e0 = (logits[0] - mx).exp();
        let e1 = (logits[1] - mx).exp();
        Ok((zc, [e0 / (e0 + e1), e1 / (e0 + e1)]))
    }

    /// Prefill `t` tokens; returns the post-final-norm last hidden.
    #[cfg(test)]
    pub fn prefill(&mut self, embeds: &[f32], t: usize) -> Result<[f32; 2]> {
        Ok(self.prefill_full(embeds, t)?.1)
    }

    /// Prefill `t` tokens; returns (last hidden after the final norm, stop probs).
    pub fn prefill_full(&mut self, embeds: &[f32], t: usize) -> Result<(Vec<f32>, [f32; 2])> {
        if embeds.len() != t * HIDDEN {
            bail!("prefill embeds {} != t*{}", embeds.len(), HIDDEN);
        }
        let x = self.gpu.upload_f32(embeds, "prefill.x");
        let h = self.transformer(x, t, 0)?;
        self.last_hidden(&h, t)
    }

    /// One decode step at absolute position `pos`; returns (z, stop probs).
    pub fn step(&mut self, embed: &[f32], pos: usize) -> Result<(Vec<f32>, [f32; 2])> {
        if embed.len() != HIDDEN {
            bail!("step embed {} != {}", embed.len(), HIDDEN);
        }
        let x = self.gpu.upload_f32(embed, "step.x");
        let h = self.transformer(x, 1, pos)?;
        self.last_hidden(&h, 1)
    }
}