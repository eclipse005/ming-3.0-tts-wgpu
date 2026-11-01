//! VAE audio encoder on wgpu: frame frontend + 24-layer causal Qwen2 stack +
//! patch/cls frontend + 4-layer aggregator + diagonal-gaussian parameters.
//! Same geometry as the LLM (896, GQA 14/2, rope theta 1e6) so the M1 kernel
//! family is reused directly.

// Parts of this module are exercised only by the parity tests (they read
// goldens) or kept as public API; the release binary does not call them.
#![allow(dead_code)]

use crate::gpu::Gpu;
use crate::ops::{AttendParams, KvCache, Lin, Ops, QwenLayer};
use crate::qwen_lm::{HEAD_DIM, HIDDEN, KV_CAP, KV_HEADS, Q_HEADS, ROPE_THETA};
use crate::weights::Weights;
use anyhow::Result;
use std::cell::RefCell;
use std::sync::Arc;
use wgpu::Buffer;

const IN_DIM: usize = 882;
const HOP: usize = 882;
const ENC_LAYERS: usize = 24;
const AGG_LAYERS: usize = 4;
const PATCH: usize = 4;

pub struct VaeEncoder {
    gpu: Gpu,
    fc1: Lin,
    fc2: Lin,
    enc: Vec<QwenLayer>,
    agg: Vec<QwenLayer>,
    cls: Buffer,
    fc3: Lin,
    enc_norm: Buffer,
    agg_norm: Buffer,
    kv: RefCell<KvCache>,
    ops: Arc<Ops>,
}

impl VaeEncoder {
    pub fn load(gpu: Gpu, ops: Arc<Ops>, w: &Weights) -> Result<Self> {
        let p = |name: &str| w.get(name);
        let lin = |base: &str, rows: usize, inf: usize| -> Result<Lin> {
            let wt = p(&format!("{base}.weight"))?;
            let packed = wt.to_bf16_packed_u32()?;
            let wbuf = gpu.upload_u32(&packed, "lin.w");
            let bbuf = match p(&format!("{base}.bias")) {
                Ok(bt) => gpu.upload_f32(&bt.to_f32()?, "lin.b"),
                Err(_) => gpu.upload_f32(&vec![0.0; rows], "lin.b0"),
            };
            Ok(Lin { w: wbuf, b: bbuf, rows, in_features: inf })
        };
        let qlayer = |base: &str| -> Result<QwenLayer> {
            Ok(QwenLayer {
                q: lin(&format!("{base}.self_attn.q_proj"), HIDDEN, HIDDEN)?,
                k: lin(&format!("{base}.self_attn.k_proj"), KV_HEADS * HEAD_DIM, HIDDEN)?,
                v: lin(&format!("{base}.self_attn.v_proj"), KV_HEADS * HEAD_DIM, HIDDEN)?,
                o: lin(&format!("{base}.self_attn.o_proj"), HIDDEN, HIDDEN)?,
                gate: lin(&format!("{base}.mlp.gate_proj"), 4864, HIDDEN)?,
                up: lin(&format!("{base}.mlp.up_proj"), 4864, HIDDEN)?,
                down: lin(&format!("{base}.mlp.down_proj"), HIDDEN, 4864)?,
                n1: gpu.upload_f32(&p(&format!("{base}.input_layernorm.weight"))?.to_f32()?, "n1"),
                n2: gpu.upload_f32(&p(&format!("{base}.post_attention_layernorm.weight"))?.to_f32()?, "n2"),
            })
        };
        let enc = (0..ENC_LAYERS)
            .map(|i| qlayer(&format!("audio.encoder.encoder.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let agg = (0..AGG_LAYERS)
            .map(|i| qlayer(&format!("audio.encoder.aggregator.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let kv = RefCell::new(KvCache::new(&gpu, KV_HEADS, HEAD_DIM, ROPE_THETA, KV_CAP));
        Ok(Self {
            fc1: lin("audio.encoder.fc1", HIDDEN, IN_DIM)?,
            fc2: lin("audio.encoder.fc2", HIDDEN, HIDDEN)?,
            enc,
            agg,
            cls: gpu.upload_f32(&p("audio.encoder.cls_embed")?.to_f32()?, "cls"),
            enc_norm: gpu.upload_f32(&p("audio.encoder.encoder.norm.weight")?.to_f32()?, "enorm"),
            agg_norm: gpu.upload_f32(&p("audio.encoder.aggregator.norm.weight")?.to_f32()?, "agnorm"),
            fc3: lin("audio.encoder.fc3", 128, HIDDEN)?,
            kv,
            ops,
            gpu,
        })
    }

    fn rms(&self, x: &Buffer, w: &Buffer, rows: usize) -> Buffer {
        self.ops.rms(x, w, rows, HIDDEN)
    }

    fn mrow(&self, lin: &Lin, x: &Buffer, t: usize) -> Result<Buffer> {
        self.ops.linear(lin, x, t)
    }

    fn silu_mul(&self, g: &Buffer, u: &Buffer, n: usize) -> Buffer {
        self.ops.silu_mul(g, u, n)
    }

    fn add(&self, a: &Buffer, b: &Buffer, n: usize) -> Buffer {
        self.ops.add(a, b, n)
    }

    /// Causal Qwen2 stack over `t` tokens at positions 0..t.
    fn stack(&self, layers: &[QwenLayer], x: Buffer, t: usize) -> Result<Buffer> {
        let mut h = x;
        let mut kv = self.kv.borrow_mut();
        kv.ensure(&self.gpu, KV_HEADS, HEAD_DIM, ROPE_THETA, t, KV_CAP);
        let cap = kv.cap;
        for layer in layers {
            let a = self.rms(&h, &layer.n1, t);
            let q = self.mrow(&layer.q, &a, t)?;
            let k = self.mrow(&layer.k, &a, t)?;
            let v = self.mrow(&layer.v, &a, t)?;
            // Qwen2 rotate-half; the audio encoder runs causal + a 64-token window
            self.ops.rope(&kv.rope, &q, Q_HEADS, 0, 0, t);
            self.ops.rope(&kv.rope, &k, KV_HEADS, 0, 0, t);
            let kc = &kv.k;
            let vc = &kv.v;
            self.ops.kv_write(&k, kc, t, 0, cap, KV_HEADS);
            self.ops.kv_write(&v, vc, t, 0, cap, KV_HEADS);
            let attn = self.ops.attend(
                &AttendParams::causal(t, 0, Q_HEADS / KV_HEADS, HEAD_DIM, HIDDEN, cap)
                    .with_window(64),
                &q,
                kc,
                vc,
                Q_HEADS,
            );
            let o = self.mrow(&layer.o, &attn, t)?;
            h = self.add(&h, &o, t * HIDDEN);
            let a2 = self.rms(&h, &layer.n2, t);
            let g = self.mrow(&layer.gate, &a2, t)?;
            let up = self.mrow(&layer.up, &a2, t)?;
            let sil = self.silu_mul(&g, &up, t * 4864);
            let d = self.mrow(&layer.down, &sil, t)?;
            h = self.add(&h, &d, t * HIDDEN);
        }
        Ok(h)
    }

    // ---- bisect helpers (parity diagnostics) ----
    pub fn embed_check(&self, frames: &[f32], t: usize) -> Result<Vec<f32>> {
        let x = self.gpu.upload_f32(frames, "v.frames");
        let e1 = self.mrow(&self.fc1, &x, t)?;
        self.gpu.read_f32(&self.mrow(&self.fc2, &e1, t)?, t * HIDDEN)
    }

    pub fn encode_stack_check(&self, embeds: &[f32], t: usize) -> Result<Vec<f32>> {
        let x = self.gpu.upload_f32(embeds, "v.e2");
        let h = self.stack(&self.enc, x, t)?;
        let z = self.rms(&h, &self.enc_norm, t);
        self.gpu.read_f32(&z, t * HIDDEN)
    }

    pub fn stack_range_check(&self, embeds: &[f32], t: usize, r: std::ops::Range<usize>) -> Result<Vec<f32>> {
        let x = self.gpu.upload_f32(embeds, "v.e2");
        let h = self.stack(&self.enc[r], x, t)?;
        self.gpu.read_f32(&h, t * HIDDEN)
    }

    pub fn agg_check(&self, enc_out: &[f32], t_frames: usize) -> Result<Vec<f32>> {
        let p_count = (t_frames + PATCH - 1) / PATCH;
        let padded = p_count * PATCH;
        let mut seq = vec![0f32; padded * 5 * HIDDEN];
        for p in 0..p_count {
            for f in 0..PATCH {
                let src = p * PATCH + f;
                if src < t_frames {
                    seq[(p * 5 + f) * HIDDEN..(p * 5 + f) * HIDDEN + HIDDEN]
                        .copy_from_slice(&enc_out[src * HIDDEN..(src + 1) * HIDDEN]);
                }
            }
        }
        let cls_host = self.gpu.read_f32(&self.cls, HIDDEN)?;
        for p in 0..p_count {
            seq[(p * 5 + 4) * HIDDEN..(p * 5 + 4) * HIDDEN + HIDDEN].copy_from_slice(&cls_host);
        }
        let seq_buf = self.gpu.upload_f32(&seq, "v.seq");
        let a_raw = self.stack(&self.agg, seq_buf, p_count * 5)?;
        let a = self.rms(&a_raw, &self.agg_norm, padded * 5);
        self.gpu.read_f32(&a, p_count * 5 * HIDDEN)
    }

    /// Layer-0 sub-stage bisect (parity diagnostics).
    pub fn layer0_parts_check(&self, embeds: &[f32], t: usize) -> Result<Vec<(String, Vec<f32>)>> {
        let l = &self.enc[0];
        let xb = self.gpu.upload_f32(embeds, "v.e0");
        let a = self.rms(&xb, &l.n1, t);
        let mut out = vec![("l0_attnln".into(), self.gpu.read_f32(&a, t * HIDDEN)?)];
        let q = self.mrow(&l.q, &a, t)?;
        let k = self.mrow(&l.k, &a, t)?;
        let v = self.mrow(&l.v, &a, t)?;
        out.push(("l0_q_proj".into(), self.gpu.read_f32(&q, t * HIDDEN)?));
        out.push(("l0_k_proj".into(), self.gpu.read_f32(&k, t * (KV_HEADS * HEAD_DIM))?));
        out.push(("l0_v_proj".into(), self.gpu.read_f32(&v, t * (KV_HEADS * HEAD_DIM))?));
        // Qwen2 rotate-half; the audio encoder runs causal + a 64-token window
        let mut kv = self.kv.borrow_mut();
        kv.ensure(&self.gpu, KV_HEADS, HEAD_DIM, ROPE_THETA, t, KV_CAP);
        let cap = kv.cap;
        self.ops.rope(&kv.rope, &q, Q_HEADS, 0, 0, t);
        self.ops.rope(&kv.rope, &k, KV_HEADS, 0, 0, t);
        let kc = &kv.k;
        let vc = &kv.v;
        self.ops.kv_write(&k, kc, t, 0, cap, KV_HEADS);
        self.ops.kv_write(&v, vc, t, 0, cap, KV_HEADS);
        if std::env::var("MING_KV_TRACE").is_ok() {
            let kch = self.gpu.read_f32(kc, KV_HEADS * cap * HEAD_DIM)?;
            let mut kk = vec![0f32; t * 128];
            for pos in 0..t {
                for h in 0..KV_HEADS {
                    for d in 0..64 {
                        kk[pos * 128 + h * 64 + d] = kch[(h * cap + pos) * 64 + d];
                    }
                }
            }
            out.push(("l0_kcache".into(), kk));
        }
        let attn = self.ops.attend(
            &AttendParams::causal(t, 0, Q_HEADS / KV_HEADS, HEAD_DIM, HIDDEN, cap)
                .with_window(64),
            &q,
            kc,
            vc,
            Q_HEADS,
        );
        let attn_host = self.gpu.read_f32(&attn, t * HIDDEN)?;
        out.push(("l0_attn".into(), attn_host.clone()));
        let o = self.mrow(&l.o, &attn, t)?;
        out.push(("l0_o_proj".into(), self.gpu.read_f32(&o, t * HIDDEN)?));
        let h = self.add(&xb, &o, t * HIDDEN);
        let a2 = self.rms(&h, &l.n2, t);
        let gat = self.mrow(&l.gate, &a2, t)?;
        let up = self.mrow(&l.up, &a2, t)?;
        out.push(("l0_gate_proj".into(), self.gpu.read_f32(&gat, t * 4864)?));
        out.push(("l0_up_proj".into(), self.gpu.read_f32(&up, t * 4864)?));
        let sil = self.silu_mul(&gat, &up, t * 4864);
        let d = self.mrow(&l.down, &sil, t)?;
        out.push(("l0_down_proj".into(), self.gpu.read_f32(&d, t * HIDDEN)?));
        let hf = self.add(&h, &d, t * HIDDEN);
        out.push(("layer0".into(), self.gpu.read_f32(&hf, t * HIDDEN)?));
        Ok(out)
    }

    /// waveform -> (mean, std) gaussian parameters, shape [T_patch, 64] each.
    pub fn encode_params(&self, waveform: &[f32]) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let mut out_vec: Vec<(String, Vec<f32>)> = Vec::new();
        let _ = &out_vec;
        let n = waveform.len();
        let t_frames = (n + HOP - 1) / HOP;
        let mut frames = vec![0f32; t_frames * IN_DIM];
        for t in 0..t_frames {
            let start = t * HOP;
            let end = (start + IN_DIM).min(n);
            frames[t * IN_DIM..t * IN_DIM + (end - start)]
                .copy_from_slice(&waveform[start..end]);
        }
        let x = self.gpu.upload_f32(&frames, "v.frames");
        let e1 = self.mrow(&self.fc1, &x, t_frames)?;
        let e2 = self.mrow(&self.fc2, &e1, t_frames)?;
        let h_raw = self.stack(&self.enc, e2, t_frames)?;
        let h = self.rms(&h_raw, &self.enc_norm, t_frames);

        // patch + cls frontend: [P*5, 896]
        let h_host = self.gpu.read_f32(&h, t_frames * HIDDEN)?;
        let p_count = (t_frames + PATCH - 1) / PATCH;
        let padded = p_count * PATCH;
        let mut seq = vec![0f32; padded * 5 * HIDDEN]; // row 5p+4 = cls
        for p in 0..p_count {
            for f in 0..PATCH {
                let src = p * PATCH + f;
                if src < t_frames {
                    seq[(p * 5 + f) * HIDDEN..(p * 5 + f) * HIDDEN + HIDDEN]
                        .copy_from_slice(&h_host[src * HIDDEN..(src + 1) * HIDDEN]);
                }
            }
        }
        // cls rows come from the learned embedding (broadcast)
        let cls_host = self.gpu.read_f32(&self.cls, HIDDEN)?;
        for p in 0..p_count {
            seq[(p * 5 + 4) * HIDDEN..(p * 5 + 4) * HIDDEN + HIDDEN].copy_from_slice(&cls_host);
        }
        let seq_buf = self.gpu.upload_f32(&seq, "v.seq");
        let a_raw = self.stack(&self.agg, seq_buf, p_count * 5)?;
        let a = self.rms(&a_raw, &self.agg_norm, padded * 5);

        // extract cls positions (every 5th row, offset 4)
        let a_host = self.gpu.read_f32(&a, p_count * 5 * HIDDEN)?;
        let mut cls_rows = vec![0f32; p_count * HIDDEN];
        for p in 0..p_count {
            cls_rows[p * HIDDEN..(p + 1) * HIDDEN].copy_from_slice(
                &a_host[(p * 5 + 4) * HIDDEN..(p * 5 + 4) * HIDDEN + HIDDEN]);
        }
        let cr = self.gpu.upload_f32(&cls_rows, "v.clsrows");
        out_vec.push(("fc3_in".into(), cls_rows.clone()));
        let out = self.mrow(&self.fc3, &cr, p_count)?;
        let flat = self.gpu.read_f32(&out, p_count * 128)?;
        out_vec.push(("fc3_out".into(), flat.clone()));
        // diffusers OobleckDiagonalGaussianDistribution: mean = chunk 0,
        // std = softplus(chunk 1) + 1e-4 -- NOT exp(0.5 * logvar).
        // PyTorch softplus switches to the identity above threshold = 20.
        let mut mean = vec![0f32; p_count * 64];
        let mut std = vec![0f32; p_count * 64];
        for i in 0..p_count * 64 {
            let row = i / 64;
            let ch = i % 64;
            mean[i] = flat[row * 128 + ch];
            let scale = flat[row * 128 + 64 + ch];
            let sp = if scale > 20.0 { scale } else { (scale.exp() + 1.0).ln() };
            std[i] = sp + 1e-4;
        }
        out_vec.push(("mean".into(), mean.clone()));
        out_vec.push(("std".into(), std.clone()));
        BISECT_OUT.with(|c| *c.borrow_mut() = Some(out_vec.clone()));
        Ok((mean, std, p_count))
    }
}

thread_local! {
    static BISECT_OUT: std::cell::RefCell<Option<Vec<(String, Vec<f32>)>>> =
        const { std::cell::RefCell::new(None) };
}
impl VaeEncoder {
    pub fn take_bisect(&self) -> Vec<(String, Vec<f32>)> {
        BISECT_OUT.with(|c| c.borrow_mut().take()).unwrap_or_default()
    }
}