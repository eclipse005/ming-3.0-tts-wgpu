//! VAE audio decoder on wgpu (M4): latent -> fc1 -> x4 linear upsample ->
//! 24-layer causal Qwen2 stack (sliding window 64) -> final RMSNorm ->
//! ISTFT head (896 -> 3530 = 2 x 1765) -> CPU iSTFT -> waveform.
//!
//! Same geometry as the encoder, so the M1/M3 kernel family is reused; the
//! only new piece is the CPU iSTFT (istft.rs).

use crate::gpu::Gpu;
use crate::istft::Istft;
use crate::nn::{kernel, Kernel};
use crate::qwen_lm::{HEAD_DIM, HIDDEN, KV_CAP, KV_HEADS, Q_HEADS, ROPE_THETA};
use crate::shaders;
use crate::vae::{uni2, Lin, QLayer};
use crate::weights::Weights;
use anyhow::Result;
use rustfft::num_complex::Complex32;
use wgpu::Buffer;

const DEC_LAYERS: usize = 24;
const LATENT_DIM: usize = 64;
const UPSAMPLE: usize = 4;
const MLP_DIM: usize = 4864;
const EPS: f32 = 1e-6;
const WINDOW: u32 = 64;

pub struct VaeDecoder {
    gpu: Gpu,
    fc1: Lin,
    dec: Vec<QLayer>,
    dec_norm: Buffer,
    head: Lin,
    istft: Istft,
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

impl VaeDecoder {
    pub fn load(gpu: Gpu, w: &Weights) -> Result<Self> {
        let p = |name: &str| w.get(name);
        let lin = |base: &str, rows: usize, inf: usize| -> Result<Lin> {
            let wt = p(&format!("{base}.weight"))?;
            let packed = wt.to_bf16_packed_u32()?;
            let wbuf = gpu.upload_u32(&packed, "dec.w");
            let bbuf = match p(&format!("{base}.bias")) {
                Ok(bt) => gpu.upload_f32(&bt.to_f32()?, "dec.b"),
                Err(_) => gpu.upload_f32(&vec![0.0; rows], "dec.b0"),
            };
            Ok(Lin { w: wbuf, b: bbuf, rows, in_features: inf })
        };
        let qlayer = |base: &str| -> Result<QLayer> {
            Ok(QLayer {
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
        let (mut cos, mut sin) = (vec![0f32; KV_CAP * 32], vec![0f32; KV_CAP * 32]);
        for pos in 0..KV_CAP {
            for i in 0..32 {
                let freq = ROPE_THETA.powf((2 * i) as f64 / 64.0).recip();
                cos[pos * 32 + i] = (pos as f64 * freq).cos() as f32;
                sin[pos * 32 + i] = (pos as f64 * freq).sin() as f32;
            }
        }
        let n_fft = 4 * 882;
        let win = p("audio.decoder.head.istft.window")?.to_f32()?;
        Ok(Self {
            fc1: lin("audio.decoder.fc1", HIDDEN, LATENT_DIM)?,
            dec,
            dec_norm: gpu.upload_f32(&p("audio.decoder.decoder.norm.weight")?.to_f32()?, "dec.norm"),
            head: lin("audio.decoder.head.out", n_fft + 2, HIDDEN)?,
            istft: Istft::new(n_fft, 882, win),
            cos_buf: gpu.upload_f32(&cos, "dec.cos"),
            sin_buf: gpu.upload_f32(&sin, "dec.sin"),
            k_rms: kernel(&gpu, shaders::RMSNORM, 2),
            k_mrow: kernel(&gpu, shaders::GEMV_MROW, 3),
            k_rope: kernel(&gpu, shaders::ROPE, 2),
            k_attend: kernel(&gpu, shaders::ATTEND, 3),
            k_silu: kernel(&gpu, shaders::SILU_MUL, 2),
            k_add: kernel(&gpu, shaders::ADD, 2),
            k_kvwrite: kernel(&gpu, shaders::KV_WRITE, 1),
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
        let y = self.gpu.storage((rows * HIDDEN * 4) as u64, "dec.rms");
        self.k_rms.run(&self.gpu, &uni2(HIDDEN as u32, EPS.to_bits()), &[x, w], &y,
                       rows as u32, 1);
        y
    }

    fn mrow(&self, lin: &Lin, x: &Buffer, t: usize) -> Result<Buffer> {
        let y = self.gpu.storage((t * lin.rows * 4) as u64, "dec.mrow");
        self.k_mrow.run(&self.gpu,
                        &uni2(lin.rows as u32, (lin.in_features / 2) as u32),
                        &[&lin.w, x, &lin.b], &y,
                        ((lin.rows + 7) / 8) as u32, t as u32);
        Ok(y)
    }

    fn silu_mul(&self, g: &Buffer, u: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "dec.silu");
        self.k_silu.run(&self.gpu, &crate::vae::uni1(n as u32), &[g, u], &y,
                        ((n + 255) / 256) as u32, 1);
        y
    }

    fn add(&self, a: &Buffer, b: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "dec.add");
        self.k_add.run(&self.gpu, &crate::vae::uni1(n as u32), &[a, b], &y,
                       ((n + 255) / 256) as u32, 1);
        y
    }

    /// Causal Qwen2 stack with sliding window 64 over t tokens at positions 0..t.
    fn stack(&self, layers: &[QLayer], x: Buffer, t: usize) -> Result<Buffer> {
        let mut h = x;
        for layer in layers {
            let a = self.rms(&h, &layer.n1, t);
            let q = self.mrow(&layer.q, &a, t)?;
            let k = self.mrow(&layer.k, &a, t)?;
            let v = self.mrow(&layer.v, &a, t)?;
            self.k_rope.run(&self.gpu, &uni2(Q_HEADS as u32, 0),
                            &[&self.cos_buf, &self.sin_buf], &q, t as u32, 1);
            self.k_rope.run(&self.gpu, &uni2(KV_HEADS as u32, 0),
                            &[&self.cos_buf, &self.sin_buf], &k, t as u32, 1);
            let kc = self.gpu.storage((KV_HEADS * KV_CAP * HEAD_DIM * 4) as u64, "dec.kc");
            let vc = self.gpu.storage((KV_HEADS * KV_CAP * HEAD_DIM * 4) as u64, "dec.vc");
            let mut ku = [0u8; 16];
            ku[0..4].copy_from_slice(&0u32.to_le_bytes());
            ku[4..8].copy_from_slice(&(KV_CAP as u32).to_le_bytes());
            ku[8..12].copy_from_slice(&(KV_HEADS as u32).to_le_bytes());
            self.k_kvwrite.run(&self.gpu, &ku, &[&k], &kc, t as u32, 1);
            self.k_kvwrite.run(&self.gpu, &ku, &[&v], &vc, t as u32, 1);
            let mut au = [0u8; 32];
            au[0..4].copy_from_slice(&(t as u32).to_le_bytes());
            au[4..8].copy_from_slice(&0u32.to_le_bytes());
            au[8..12].copy_from_slice(&((Q_HEADS / KV_HEADS) as u32).to_le_bytes());
            au[12..16].copy_from_slice(&(1.0f32 / (HEAD_DIM as f32).sqrt()).to_le_bytes());
            au[16..20].copy_from_slice(&0u32.to_le_bytes());
            au[20..24].copy_from_slice(&0u32.to_le_bytes());
            au[24..28].copy_from_slice(&(HIDDEN as u32).to_le_bytes());
            au[28..32].copy_from_slice(&WINDOW.to_le_bytes());
            let attn = self.gpu.storage((t * HIDDEN * 4) as u64, "dec.attn");
            self.k_attend.run(&self.gpu, &au, &[&q, &kc, &vc], &attn, Q_HEADS as u32, t as u32);
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
