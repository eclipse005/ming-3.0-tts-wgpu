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

/// Persistent per-token scratch. Decode reuses the same shapes 24 times per
/// step and ~63 steps per utterance; keeping buffers and bind groups stable
/// removes the allocation/recording cost that CUDA graphs hide in Python.
struct LlmScratch {
    x0: Buffer,
    h0: Buffer,
    h1: Buffer,
    a: Buffer,
    a2: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    sil: Buffer,
}

pub struct LlmEngine {
    gpu: Gpu,
    layers: Vec<Layer>,
    scratch: LlmScratch,
    norm_final: Buffer,
    stop_host_w: Vec<f32>,
    stop_host_b: Vec<f32>,
    rope: RopeTables,
    ops: Arc<Ops>,
}


fn nan_probe(gpu: &Gpu, label: &str, buf: &Buffer, n: usize) {
    if std::env::var("MING_NAN_TRACE").is_err() {
        return;
    }
    if let Ok(v) = gpu.read_f32(buf, n) {
        let nan = v.iter().filter(|x| x.is_nan()).count();
        let inf = v.iter().filter(|x| x.is_infinite()).count();
        let max = v.iter().fold(0f32, |a, b| if b.is_nan() { a } else { a.max(b.abs()) });
        println!("    [nan] {label:16} nan={nan} inf={inf} absmax={max:.4e}");
    }
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
        let scratch = {
            let mk = |n: usize, label: &str| gpu.storage((n * 4) as u64, label);
            LlmScratch {
                x0: mk(HIDDEN, "lm.x0"),
                h0: mk(HIDDEN, "lm.h0"),
                h1: mk(HIDDEN, "lm.h1"),
                a: mk(HIDDEN, "lm.a"),
                a2: mk(HIDDEN, "lm.a2"),
                q: mk(Q_HEADS * HEAD_DIM, "lm.q"),
                k: mk(KV_HEADS * HEAD_DIM, "lm.k"),
                v: mk(KV_HEADS * HEAD_DIM, "lm.v"),
                attn: mk(HIDDEN, "lm.attn"),
                o: mk(HIDDEN, "lm.o"),
                gate: mk(FF, "lm.gate"),
                up: mk(FF, "lm.up"),
                sil: mk(FF, "lm.sil"),
            }
        };

        Ok(Self {
            gpu,
            layers,
            scratch,
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
        if t == 1 && std::env::var("MING_LM_FAST").map_or(true, |v| v != "0") {
            let s = &self.scratch;
            let mut h = x;
            for (li, l) in self.layers.iter().enumerate() {
                // DiT/aggregator and LLM share one Ops instance; give the LLM
                // its own non-weight cache namespace.
                let kb = 0x1_0000 | ((li as u64) << 8);
                self.ops
                    .rms_into_cached(&h, &l.norm1, &s.a, 1, HIDDEN, kb | 0x10);
                self.ops
                    .linear_into_cached(&l.q, &s.a, &s.q, 1, 1, 2);
                self.ops
                    .linear_into_cached(&l.k, &s.a, &s.k, 1, 1, 3);
                self.ops
                    .linear_into_cached(&l.v, &s.a, &s.v, 1, 1, 4);
                self.ops
                    .rope_cached(kb | 0x18, &self.rope, &s.q, Q_HEADS, 0, pos_base, 1, 0);
                self.ops
                    .rope_cached(kb | 0x19, &self.rope, &s.k, KV_HEADS, 0, pos_base, 1, 0);
                self.ops
                    .kv_write_cached(kb | 0x1A, &s.k, &l.kc, 1, pos_base, KV_CAP, KV_HEADS);
                self.ops
                    .kv_write_cached(kb | 0x1B, &s.v, &l.vc, 1, pos_base, KV_CAP, KV_HEADS);
                let ap =
                    AttendParams::causal(1, pos_base, Q_HEADS / KV_HEADS, HEAD_DIM, HIDDEN, KV_CAP);
                self.ops.attend_into_cached(
                    &ap,
                    &s.q,
                    &l.kc,
                    &l.vc,
                    &s.attn,
                    Q_HEADS,
                    kb | 0x1C,
                );
                self.ops
                    .linear_into_cached(&l.o, &s.attn, &s.o, 1, 5, 6);
                self.ops
                    .add_into_cached(&h, &s.o, &s.h1, HIDDEN, kb | 0x12);
                self.ops
                    .rms_into_cached(&s.h1, &l.norm2, &s.a2, 1, HIDDEN, kb | 0x11);
                self.ops
                    .linear_into_cached(&l.gate, &s.a2, &s.gate, 1, 7, 8);
                self.ops
                    .linear_into_cached(&l.up, &s.a2, &s.up, 1, 7, 9);
                self.ops
                    .silu_mul_into_cached(&s.gate, &s.up, &s.sil, FF, kb | 0x14);
                self.ops
                    .linear_into_cached(&l.down, &s.sil, &s.o, 1, 10, 6);
                self.ops
                    .add_into_cached(&s.h1, &s.o, &s.h0, HIDDEN, kb | 0x13);
                h = s.h0.clone();
            }
            return Ok(h);
        }
        let mut h = x;
        // MING_ACT_RANGE=1 prints the residual stream's magnitude per layer.
        // It is how we answer "can this checkpoint run in fp16": fp16 tops out
        // at 65504, so whichever layer crosses that is the one that would NaN.
        let probe = std::env::var("MING_ACT_RANGE").is_ok();
        static PREFILL_PROFILED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        let pprof = std::env::var_os("MING_LM_PREFILL_PROFILE").is_some()
            && !PREFILL_PROFILED.swap(true, std::sync::atomic::Ordering::Relaxed);
        let mut mark_at = std::time::Instant::now();
        for (li, layer) in self.layers.iter().enumerate() {
            macro_rules! mark {
                ($name:expr) => {
                    if pprof && li == 0 {
                        self.gpu.flush();
                        let now = std::time::Instant::now();
                        println!("      lm t={t} {:?} {:?}", $name, now - mark_at);
                        mark_at = now;
                    }
                };
            }
            let a = self.rms(&h, &layer.norm1, t);
            if li == 0 { nan_probe(&self.gpu, "l0.norm1", &a, t * HIDDEN); }
            let q = self.mrow(&layer.q, &a, t)?;
            if li == 0 { nan_probe(&self.gpu, "l0.q", &q, t * HIDDEN); }
            let k = self.mrow(&layer.k, &a, t)?;
            let v = self.mrow(&layer.v, &a, t)?;
            if li == 0 {
                nan_probe(&self.gpu, "l0.k", &k, t * KV_HEADS * HEAD_DIM);
                nan_probe(&self.gpu, "l0.v", &v, t * KV_HEADS * HEAD_DIM);
            }
            mark!("qkv");
            self.rope_inplace(&q, Q_HEADS, t, pos_base);
            self.rope_inplace(&k, KV_HEADS, t, pos_base);
            self.kv_write(&k, &layer.kc, t, pos_base);
            self.kv_write(&v, &layer.vc, t, pos_base);
            let attn = self.attend(&q, &layer.kc, &layer.vc, t, pos_base)?;
            mark!("rope+kv+attn");
            if li == 0 { nan_probe(&self.gpu, "l0.attn", &attn, t * HIDDEN); }
            let o = self.mrow(&layer.o, &attn, t)?;
            if li == 0 { nan_probe(&self.gpu, "l0.o", &o, t * HIDDEN); }
            h = self.add(&h, &o, t * HIDDEN);
            if li < 2 { nan_probe(&self.gpu, "h_after_attn", &h, t * HIDDEN); }
            mark!("o+add");
            let a2 = self.rms(&h, &layer.norm2, t);
            let g = self.mrow(&layer.gate, &a2, t)?;
            let up = self.mrow(&layer.up, &a2, t)?;
            mark!("gate+up");
            let sil = self.silu_mul(&g, &up, t * FF);
            let d = self.mrow(&layer.down, &sil, t)?;
            h = self.add(&h, &d, t * HIDDEN);
            mark!("silu+down+add");
            if probe {
                if let Ok(v) = self.gpu.read_f32(&h, t * HIDDEN) {
                    let m = v.iter().fold(0f32, |a, b| a.max(b.abs()));
                    println!("  layer {li:2}: residual absmax {m:.4e}");
                }
            }
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
        let fast = std::env::var("MING_LM_FAST").map_or(true, |v| v != "0");
        let x = if fast {
            self.gpu.write_f32(&self.scratch.x0, embed);
            self.scratch.x0.clone()
        } else {
            self.gpu.upload_f32(embed, "step.x")
        };
        let h = self.transformer(x, 1, pos)?;
        self.last_hidden(&h, 1)
    }
}