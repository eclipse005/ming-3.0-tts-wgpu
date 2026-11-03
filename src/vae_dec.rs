//! VAE audio decoder on wgpu (M4): latent -> fc1 -> x4 linear upsample ->
//! 24-layer causal Qwen2 stack (sliding window 64) -> final RMSNorm ->
//! ISTFT head (896 -> 3530 = 2 x 1765) -> CPU iSTFT -> waveform.
//!
//! Same geometry as the encoder, so the M1/M3 kernel family is reused; the
//! only new piece is the CPU iSTFT (istft.rs).

// Parts of this module are exercised only by the parity tests (they read
// goldens) or kept as public API; the release binary does not call them.
#![allow(dead_code)]

use crate::gpu::Gpu;
use crate::istft::Istft;
use crate::ops::{AttendParams, KvCache, Lin, Ops, QwenLayer};
use crate::qwen_lm::{HEAD_DIM, HIDDEN, KV_CAP, KV_HEADS, Q_HEADS, ROPE_THETA};
use crate::weights::Weights;
use anyhow::Result;
use rustfft::num_complex::Complex32;
use std::cell::RefCell;
use std::sync::Arc;
use wgpu::Buffer;

const DEC_LAYERS: usize = 24;
const LATENT_DIM: usize = 64;
const UPSAMPLE: usize = 4;
const MLP_DIM: usize = 4864;
const WINDOW: u32 = 64;

pub struct VaeDecoder {
    gpu: Gpu,
    fc1: Lin,
    dec: Vec<QwenLayer>,
    dec_norm: Buffer,
    head: Lin,
    istft: Istft,
    kv: RefCell<KvCache>,
    ops: Arc<Ops>,
}

impl VaeDecoder {
    pub fn load(gpu: Gpu, ops: Arc<Ops>, w: &Weights) -> Result<Self> {
        let p = |name: &str| w.get(name);
        let lin = |base: &str, rows: usize, inf: usize| -> Result<Lin> {
            Lin::load(&gpu, w, base, rows, inf)
        };
        let qlayer = |base: &str| -> Result<QwenLayer> {
            Ok(QwenLayer {
                q: lin(&format!("{base}.self_attn.q_proj"), HIDDEN, HIDDEN)?,
                k: lin(&format!("{base}.self_attn.k_proj"), KV_HEADS * HEAD_DIM, HIDDEN)?,
                v: lin(&format!("{base}.self_attn.v_proj"), KV_HEADS * HEAD_DIM, HIDDEN)?,
                o: lin(&format!("{base}.self_attn.o_proj"), HIDDEN, HIDDEN)?,
                gate: lin(&format!("{base}.mlp.gate_proj"), MLP_DIM, HIDDEN)?,
                up: lin(&format!("{base}.mlp.up_proj"), MLP_DIM, HIDDEN)?,
                down: lin(&format!("{base}.mlp.down_proj"), HIDDEN, MLP_DIM)?,
                n1: gpu.upload_f32(&p(&format!("{base}.input_layernorm.weight"))?.to_f32()?, "dec.n1"),
                n2: gpu.upload_f32(&p(&format!("{base}.post_attention_layernorm.weight"))?.to_f32()?, "dec.n2"),
            })
        };
        let dec = (0..DEC_LAYERS)
            .map(|i| qlayer(&format!("audio.decoder.decoder.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let kv = RefCell::new(KvCache::new(&gpu, KV_HEADS, HEAD_DIM, ROPE_THETA, KV_CAP));
        let n_fft = 4 * 882;
        let win = p("audio.decoder.head.istft.window")?.to_f32()?;
        Ok(Self {
            fc1: lin("audio.decoder.fc1", HIDDEN, LATENT_DIM)?,
            dec,
            dec_norm: gpu.upload_f32(&p("audio.decoder.decoder.norm.weight")?.to_f32()?, "dec.norm"),
            head: lin("audio.decoder.head.out", n_fft + 2, HIDDEN)?,
            istft: Istft::new(n_fft, 882, win),
            kv,
            ops,
            gpu,
        })
    }

    pub fn sample_rate(&self) -> usize {
        44100
    }

    /// Hop in samples (50 frames per second).
    pub fn hop(&self) -> usize {
        882
    }

    fn rms(&self, x: &Buffer, w: &Buffer, rows: usize) -> Buffer {
        self.ops.rms(x, w, rows, HIDDEN)
    }

    fn mrow(&self, lin: &Lin, x: &Buffer, t: usize) -> Result<Buffer> {
        // Decoder output is the waveform: the tiled GEMM is safe here and does
        // not influence the token trajectory (see Ops::linear_big).
        self.ops.linear_big(lin, x, t)
    }

    fn silu_mul(&self, g: &Buffer, u: &Buffer, n: usize) -> Buffer {
        self.ops.silu_mul(g, u, n)
    }

    fn add(&self, a: &Buffer, b: &Buffer, n: usize) -> Buffer {
        self.ops.add(a, b, n)
    }

    /// Causal Qwen2 stack with sliding window 64 over t tokens at positions 0..t.
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
            self.ops.rope(&kv.rope, &q, Q_HEADS, 0, 0, t);
            self.ops.rope(&kv.rope, &k, KV_HEADS, 0, 0, t);
            let kc = &kv.k;
            let vc = &kv.v;
            self.ops.kv_write(&k, kc, t, 0, cap, KV_HEADS);
            self.ops.kv_write(&v, vc, t, 0, cap, KV_HEADS);
            let attn = self.ops.attend(
                &AttendParams::causal(t, 0, Q_HEADS / KV_HEADS, HEAD_DIM, HIDDEN, cap)
                .with_window(WINDOW as usize),
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
            let sil = self.silu_mul(&g, &up, t * MLP_DIM);
            let d = self.mrow(&layer.down, &sil, t)?;
            h = self.add(&h, &d, t * HIDDEN);
        }
        Ok(h)
    }

    fn fc1_out(&self, latent: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        let x = self.gpu.upload_f32(latent, "dec.latent");
        let e = self.mrow(&self.fc1, &x, t_lat)?;
        self.gpu.read_f32(&e, t_lat * HIDDEN)
    }

    /// Latent [t_lat, 64] -> waveform.
    pub fn decode(&self, latent: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        let e = self.fc1_out(latent, t_lat)?;
        let up = linear_upsample(&e, t_lat, HIDDEN, UPSAMPLE);
        let t = t_lat * UPSAMPLE;
        let ub = self.gpu.upload_f32(&up, "dec.up");
        let h = self.stack(&self.dec, ub, t)?;
        let z = self.rms(&h, &self.dec_norm, t);
        let head = self.mrow(&self.head, &z, t)?;
        let spec = self.gpu.read_f32(&head, t * (self.head.rows))?;
        let bins = self.istft.n_bins();
        let mut cplx = Vec::with_capacity(t * bins);
        for row in 0..t {
            let base = row * self.head.rows;
            for k in 0..bins {
                let m = spec[base + k];
                let mag = if m > 4.6051702 { 100.0 } else { m.exp().min(100.0) };
                let ph = spec[base + bins + k];
                cplx.push(Complex32::new(mag * ph.cos(), mag * ph.sin()));
            }
        }
        Ok(self.istft.forward(&cplx, t))
    }

    // ---- bisect helpers (parity diagnostics) ----
    pub fn fc1_check(&self, latent: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        self.fc1_out(latent, t_lat)
    }

    pub fn upsample_check(&self, latent: &[f32], t_lat: usize) -> Result<Vec<f32>> {
        let e = self.fc1_out(latent, t_lat)?;
        Ok(linear_upsample(&e, t_lat, HIDDEN, UPSAMPLE))
    }

    pub fn stack_check(&self, up: &[f32], t: usize) -> Result<Vec<f32>> {
        let x = self.gpu.upload_f32(up, "dec.up2");
        let h = self.stack(&self.dec, x, t)?;
        let z = self.rms(&h, &self.dec_norm, t);
        self.gpu.read_f32(&z, t * HIDDEN)
    }

    pub fn head_check(&self, up: &[f32], t: usize) -> Result<Vec<f32>> {
        let x = self.gpu.upload_f32(up, "dec.up3");
        let h = self.stack(&self.dec, x, t)?;
        let z = self.rms(&h, &self.dec_norm, t);
        let o = self.mrow(&self.head, &z, t)?;
        self.gpu.read_f32(&o, t * self.head.rows)
    }

    /// torch.nn.Upsample(scale_factor, mode="linear", align_corners=False)
    /// along the time axis of a [t, ch] row-major tensor.
    pub fn upsample_from(enc: &[f32], t_in: usize, ch: usize) -> Vec<f32> {
        linear_upsample(enc, t_in, ch, UPSAMPLE)
    }
}

fn linear_upsample(x: &[f32], t_in: usize, ch: usize, scale: usize) -> Vec<f32> {
    let t_out = t_in * scale;
    let mut y = vec![0f32; t_out * ch];
    let ratio = t_in as f64 / t_out as f64;
    for d in 0..t_out {
        let mut src = ratio * (d as f64 + 0.5) - 0.5;
        if src < 0.0 {
            src = 0.0;
        }
        let i0 = (src.floor() as usize).min(t_in - 1);
        let i1 = if i0 + 1 < t_in { i0 + 1 } else { i0 };
        let l1 = (src - i0 as f64) as f32;
        let l0 = 1.0 - l1;
        for c in 0..ch {
            y[d * ch + c] = x[i0 * ch + c] * l0 + x[i1 * ch + c] * l1;
        }
    }
    y
}