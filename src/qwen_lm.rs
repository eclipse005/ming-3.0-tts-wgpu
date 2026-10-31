//! Qwen2.5-0.5B LLM engine on wgpu: prefill + incremental GQA decode with a
//! GPU-resident f32 KV cache, driving the Ming-TTS stop head.
//!
//! Parity contract (M1): given the same prefill embeds and the same per-step
//! decode inputs as the Python pipeline, every step's post-final-norm hidden
//! and stop probability match (rel_l2 < 1e-5), and the stop step is identical.

use crate::gpu::Gpu;
use crate::nn::{kernel, Kernel};
use crate::shaders::RMSNORM;
use crate::shaders;
use crate::weights::Weights;
use anyhow::{bail, Context, Result};
use wgpu::Buffer;

pub const HIDDEN: usize = 896;
pub const LAYERS: usize = 24;
pub const Q_HEADS: usize = 14;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const KV_CAP: usize = 512;
pub const FF: usize = 4864;
pub const ROPE_THETA: f64 = 1_000_000.0;
const EPS: f32 = 1e-6;

struct Lin {
    w: Buffer,
    b: Buffer,
    rows: usize,
    in_features: usize,
}

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
    cos_buf: Buffer,
    sin_buf: Buffer,
    k_rms: Kernel,
    k_mrow: Kernel,
    k_rope: Kernel,
    k_attend: Kernel,
    k_silu: Kernel,
    k_add: Kernel,
    k_kvwrite: Kernel,
}

fn uni2(a: u32, b: u32) -> [u8; 16] {
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&a.to_le_bytes());
    u[4..8].copy_from_slice(&b.to_le_bytes());
    u
}

fn uni1(a: u32) -> [u8; 16] {
    uni2(a, 0)
}

fn rope_tables() -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; KV_CAP * 32];
    let mut sin = vec![0f32; KV_CAP * 32];
    for p in 0..KV_CAP {
        for i in 0..32 {
            let freq = ROPE_THETA.powf((2 * i) as f64 / (HEAD_DIM as f64));
            let ang = p as f64 * freq.recip();
            cos[p * 32 + i] = ang.cos() as f32;
            sin[p * 32 + i] = ang.sin() as f32;
        }
    }
    (cos, sin)
}

impl LlmEngine {
    pub fn load(gpu: Gpu, w: &Weights) -> Result<Self> {
        let p = |name: &str| w.get(name);
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let base = format!("model.model.layers.{i}");
            let lin = |what: &str, rows: usize, inf: usize| -> Result<Lin> {
                let wt = p(&format!("{base}.{what}.weight"))?;
                let packed = wt.to_bf16_packed_u32()?;
                let wbuf = gpu.upload_u32(&packed, "lin.w");
                let bbuf = match p(&format!("{base}.{what}.bias")) {
                    Ok(bt) => gpu.upload_f32(&bt.to_f32()?, "lin.b"),
                    Err(_) => gpu.upload_f32(&vec![0.0; rows], "lin.b0"),
                };
                Ok(Lin { w: wbuf, b: bbuf, rows, in_features: inf })
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
        let (cos, sin) = rope_tables();
        let cos_buf = gpu.upload_f32(&cos, "rope.cos");
        let sin_buf = gpu.upload_f32(&sin, "rope.sin");

        Ok(Self {
            k_rms: kernel(&gpu, RMSNORM, 2),
            k_mrow: kernel(&gpu, shaders::GEMV_MROW, 3),
            k_rope: kernel(&gpu, shaders::ROPE, 2),
            k_attend: kernel(&gpu, shaders::ATTEND, 3),
            k_silu: kernel(&gpu, shaders::SILU_MUL, 2),
            k_add: kernel(&gpu, shaders::ADD, 2),
            k_kvwrite: kernel(&gpu, shaders::KV_WRITE, 1),
            gpu,
            layers,
            norm_final,
            stop_host_w,
            stop_host_b,
            cos_buf,
            sin_buf,
        })
    }

    fn mrow(&self, lin: &Lin, x: &Buffer, t: usize) -> Result<Buffer> {
        let y = self.gpu.storage((t * lin.rows * 4) as u64, "mrow.y");
        self.k_mrow.run(
            &self.gpu,
            &uni2(lin.rows as u32, (lin.in_features / 2) as u32),
            &[&lin.w, x, &lin.b],
            &y,
            ((lin.rows + 7) / 8) as u32,
            t as u32,
        );
        Ok(y)
    }

    fn rope_inplace(&self, q: &Buffer, n_heads: usize, tokens: usize, pos_base: usize) {
        self.k_rope.run(
            &self.gpu,
            &uni2(n_heads as u32, pos_base as u32),
            &[&self.cos_buf, &self.sin_buf],
            q,
            tokens as u32,
            1,
        );
    }

    fn kv_write(&self, src: &Buffer, cache: &Buffer, tokens: usize, pos_base: usize) {
        self.k_kvwrite.run(
            &self.gpu,
            &uni1(pos_base as u32),
            &[src],
            cache,
            tokens as u32,
            1,
        );
    }

    fn attend(
        &self,
        q: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        t: usize,
        q_pos_base: usize,
    ) -> Result<Buffer> {
        let y = self.gpu.storage((t * HIDDEN * 4) as u64, "attn.y");
        let mut u = [0u8; 16];
        u[0..4].copy_from_slice(&(t as u32).to_le_bytes());
        u[4..8].copy_from_slice(&(q_pos_base as u32).to_le_bytes());
        u[8..12].copy_from_slice(&((Q_HEADS / KV_HEADS) as u32).to_le_bytes());
        u[12..16].copy_from_slice(&(1.0f32 / (HEAD_DIM as f32).sqrt()).to_le_bytes());
        self.k_attend.run(&self.gpu, &u, &[q, kc, vc], &y, Q_HEADS as u32, t as u32);
        Ok(y)
    }

    fn silu_mul(&self, g: &Buffer, u: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "silu.y");
        self.k_silu.run(&self.gpu, &uni1(n as u32), &[g, u], &y, ((n + 255) / 256) as u32, 1);
        y
    }

    fn add(&self, a: &Buffer, b: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "add.y");
        self.k_add.run(&self.gpu, &uni1(n as u32), &[a, b], &y, ((n + 255) / 256) as u32, 1);
        y
    }

    fn rms(&self, x: &Buffer, w: &Buffer, tokens: usize) -> Buffer {
        let y = self.gpu.storage((tokens * HIDDEN * 4) as u64, "rms.y");
        self.k_rms.run(
            &self.gpu,
            &uni2(HIDDEN as u32, EPS.to_bits()),
            &[x, w],
            &y,
            tokens as u32,
            1,
        );
        y
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
    pub fn prefill(&mut self, embeds: &[f32], t: usize) -> Result<[f32; 2]> {
        if embeds.len() != t * HIDDEN {
            bail!("prefill embeds {} != t*{}", embeds.len(), HIDDEN);
        }
        let x = self.gpu.upload_f32(embeds, "prefill.x");
        let h = self.transformer(x, t, 0)?;
        let (_, prob) = self.last_hidden(&h, t)?;
        Ok(prob)
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
