//! VAE encoder and decoder on the host: the 24-layer causal stacks with a
//! 64-token window, the patch/cls frontend and the ISTFT head.

use anyhow::Result;

use super::tensor::{
    add_into, attention_window, rmsnorm, rope_split_half, silu_mul, CpuLin,
};
use super::ZC;
use crate::qwen_lm::{HEAD_DIM, HIDDEN, KV_HEADS, Q_HEADS, ROPE_THETA};
use crate::weights::Weights;

pub const DEC_LAYERS: usize = 24;

pub const UPSAMPLE: usize = 4;

pub const MLP_DIM: usize = 4864;

pub const VAE_WINDOW: usize = 64;

struct CpuQwenLayer {
    q: CpuLin,
    k: CpuLin,
    v: CpuLin,
    o: CpuLin,
    gate: CpuLin,
    up: CpuLin,
    down: CpuLin,
    n1: Vec<f32>,
    n2: Vec<f32>,
}

impl CpuQwenLayer {
    fn load(w: &Weights, base: &str) -> Result<Self> {
        Ok(Self {
            q: CpuLin::load(w, &format!("{base}.self_attn.q_proj"), HIDDEN, HIDDEN)?,
            k: CpuLin::load(w, &format!("{base}.self_attn.k_proj"), KV_HEADS * HEAD_DIM, HIDDEN)?,
            v: CpuLin::load(w, &format!("{base}.self_attn.v_proj"), KV_HEADS * HEAD_DIM, HIDDEN)?,
            o: CpuLin::load(w, &format!("{base}.self_attn.o_proj"), HIDDEN, HIDDEN)?,
            gate: CpuLin::load(w, &format!("{base}.mlp.gate_proj"), MLP_DIM, HIDDEN)?,
            up: CpuLin::load(w, &format!("{base}.mlp.up_proj"), MLP_DIM, HIDDEN)?,
            down: CpuLin::load(w, &format!("{base}.mlp.down_proj"), HIDDEN, MLP_DIM)?,
            n1: w.get(&format!("{base}.input_layernorm.weight"))?.to_f32()?,
            n2: w.get(&format!("{base}.post_attention_layernorm.weight"))?.to_f32()?,
        })
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

pub struct CpuVaeDec {
    fc1: CpuLin,
    dec: Vec<CpuQwenLayer>,
    dec_norm: Vec<f32>,
    head: CpuLin,
    istft: crate::istft::Istft,
}

impl CpuVaeDec {
    pub fn load(w: &Weights) -> Result<Self> {
        let dec = (0..DEC_LAYERS)
            .map(|i| CpuQwenLayer::load(w, &format!("audio.decoder.decoder.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let n_fft = 4 * 882;
        let win = w.get("audio.decoder.head.istft.window")?.to_f32()?;
        Ok(Self {
            fc1: CpuLin::load(w, "audio.decoder.fc1", HIDDEN, ZC)?,
            dec,
            dec_norm: w.get("audio.decoder.decoder.norm.weight")?.to_f32()?,
            head: CpuLin::load(w, "audio.decoder.head.out", n_fft + 2, HIDDEN)?,
            istft: crate::istft::Istft::new(n_fft, 882, win),
        })
    }

    fn stack(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut h = x.to_vec();
        let mut a = vec![0.0f32; t * HIDDEN];
        let mut q = vec![0.0f32; t * HIDDEN];
        let mut k = vec![0.0f32; t * KV_HEADS * HEAD_DIM];
        let mut v = vec![0.0f32; t * KV_HEADS * HEAD_DIM];
        let mut attn = vec![0.0f32; t * HIDDEN];
        let mut o = vec![0.0f32; t * HIDDEN];
        let mut gate = vec![0.0f32; t * MLP_DIM];
        let mut up = vec![0.0f32; t * MLP_DIM];
        let mut sil = vec![0.0f32; t * MLP_DIM];
        let mut a2 = vec![0.0f32; t * HIDDEN];
        let mut kc = vec![0.0f32; KV_HEADS * t * HEAD_DIM];
        let mut vc = vec![0.0f32; KV_HEADS * t * HEAD_DIM];
        let mut inv_freq = vec![0.0f32; HEAD_DIM / 2];
        for (i, f) in inv_freq.iter_mut().enumerate() {
            *f = ROPE_THETA.powf((2 * i) as f64 / HEAD_DIM as f64).recip() as f32;
        }
        for l in &self.dec {
            rmsnorm(&h, &l.n1, t, HIDDEN, &mut a);
            l.q.linear(&a, t, &mut q);
            l.k.linear(&a, t, &mut k);
            l.v.linear(&a, t, &mut v);
            rope_split_half(&mut q, &inv_freq, Q_HEADS, 0, t);
            rope_split_half(&mut k, &inv_freq, KV_HEADS, 0, t);
            for tt in 0..t {
                for hh in 0..KV_HEADS {
                    let dst = hh * t * HEAD_DIM + tt * HEAD_DIM;
                    let src = (tt * KV_HEADS + hh) * HEAD_DIM;
                    kc[dst..dst + HEAD_DIM].copy_from_slice(&k[src..src + HEAD_DIM]);
                    vc[dst..dst + HEAD_DIM].copy_from_slice(&v[src..src + HEAD_DIM]);
                }
            }
            attention_window(&q, &kc, &vc, t, 0, Q_HEADS, KV_HEADS, VAE_WINDOW, t, &mut attn);
            l.o.linear(&attn, t, &mut o);
            add_into(&mut h, &o);
            rmsnorm(&h, &l.n2, t, HIDDEN, &mut a2);
            l.gate.linear(&a2, t, &mut gate);
            l.up.linear(&a2, t, &mut up);
            silu_mul(&gate, &up, &mut sil);
            l.down.linear(&sil, t, &mut o);
            add_into(&mut h, &o);
        }
        let mut out = vec![0.0f32; t * HIDDEN];
        rmsnorm(&h, &self.dec_norm, t, HIDDEN, &mut out);
        out
    }

    /// Decode [t_lat, 64] latents to a waveform.
    pub fn decode(&self, latent: &[f32], t_lat: usize) -> Vec<f32> {
        let (wav, _) = self.decode_parts(latent, t_lat);
        wav
    }

    /// Same, also returning every intermediate for parity tests.
    pub fn decode_parts(&self, latent: &[f32], t_lat: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut fc1 = vec![0.0f32; t_lat * HIDDEN];
        self.fc1.linear(latent, t_lat, &mut fc1);
        let up = linear_upsample(&fc1, t_lat, HIDDEN, UPSAMPLE);
        let t = t_lat * UPSAMPLE;
        let dec = self.stack(&up, t);
        let mut spec = vec![0.0f32; t * self.head.w.rows];
        self.head.linear(&dec, t, &mut spec);
        let bins = self.istft.n_bins();
        let mut cplx = Vec::with_capacity(t * bins);
        for row in 0..t {
            let base = row * self.head.w.rows;
            for k in 0..bins {
                let m = spec[base + k];
                let mag = if m > 4.6051702 { 100.0 } else { m.exp().min(100.0) };
                let ph = spec[base + bins + k];
                cplx.push(rustfft::num_complex::Complex32::new(mag * ph.cos(), mag * ph.sin()));
            }
        }
        let wav = self.istft.forward(&cplx, t);
        (wav, vec![fc1, up, dec, spec])
    }
}


// ---------------------------------------------------------------------------
// CPU pipeline: the same fragment loop as synth.rs, on the host
// ---------------------------------------------------------------------------

pub const ENC_LAYERS: usize = 24;

pub const AGG_LAYERS: usize = 4;

pub const ENC_PATCH: usize = 4;

pub const ENC_IN_DIM: usize = 882;

pub struct CpuVaeEnc {
    fc1: CpuLin,
    fc2: CpuLin,
    enc: Vec<CpuQwenLayer>,
    agg: Vec<CpuQwenLayer>,
    cls: Vec<f32>,
    enc_norm: Vec<f32>,
    agg_norm: Vec<f32>,
    fc3: CpuLin,
}

impl CpuVaeEnc {
    pub fn load(w: &Weights) -> Result<Self> {
        let enc = (0..ENC_LAYERS)
            .map(|i| CpuQwenLayer::load(w, &format!("audio.encoder.encoder.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let agg = (0..AGG_LAYERS)
            .map(|i| CpuQwenLayer::load(w, &format!("audio.encoder.aggregator.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            fc1: CpuLin::load(w, "audio.encoder.fc1", HIDDEN, ENC_IN_DIM)?,
            fc2: CpuLin::load(w, "audio.encoder.fc2", HIDDEN, HIDDEN)?,
            enc,
            agg,
            cls: w.get("audio.encoder.cls_embed")?.to_f32()?,
            enc_norm: w.get("audio.encoder.encoder.norm.weight")?.to_f32()?,
            agg_norm: w.get("audio.encoder.aggregator.norm.weight")?.to_f32()?,
            fc3: CpuLin::load(w, "audio.encoder.fc3", 128, HIDDEN)?,
        })
    }

    fn stack(&self, layers: &[CpuQwenLayer], x: &[f32], t: usize) -> Vec<f32> {
        let mut h = x.to_vec();
        let mut a = vec![0.0f32; t * HIDDEN];
        let mut q = vec![0.0f32; t * HIDDEN];
        let mut k = vec![0.0f32; t * KV_HEADS * HEAD_DIM];
        let mut v = vec![0.0f32; t * KV_HEADS * HEAD_DIM];
        let mut attn = vec![0.0f32; t * HIDDEN];
        let mut o = vec![0.0f32; t * HIDDEN];
        let mut gate = vec![0.0f32; t * MLP_DIM];
        let mut up = vec![0.0f32; t * MLP_DIM];
        let mut sil = vec![0.0f32; t * MLP_DIM];
        let mut a2 = vec![0.0f32; t * HIDDEN];
        let mut kc = vec![0.0f32; KV_HEADS * t * HEAD_DIM];
        let mut vc = vec![0.0f32; KV_HEADS * t * HEAD_DIM];
        let mut inv_freq = vec![0.0f32; HEAD_DIM / 2];
        for (i, f) in inv_freq.iter_mut().enumerate() {
            *f = ROPE_THETA.powf((2 * i) as f64 / HEAD_DIM as f64).recip() as f32;
        }
        for l in layers {
            rmsnorm(&h, &l.n1, t, HIDDEN, &mut a);
            l.q.linear(&a, t, &mut q);
            l.k.linear(&a, t, &mut k);
            l.v.linear(&a, t, &mut v);
            rope_split_half(&mut q, &inv_freq, Q_HEADS, 0, t);
            rope_split_half(&mut k, &inv_freq, KV_HEADS, 0, t);
            for tt in 0..t {
                for hh in 0..KV_HEADS {
                    let dst = hh * t * HEAD_DIM + tt * HEAD_DIM;
                    let src = (tt * KV_HEADS + hh) * HEAD_DIM;
                    kc[dst..dst + HEAD_DIM].copy_from_slice(&k[src..src + HEAD_DIM]);
                    vc[dst..dst + HEAD_DIM].copy_from_slice(&v[src..src + HEAD_DIM]);
                }
            }
            attention_window(&q, &kc, &vc, t, 0, Q_HEADS, KV_HEADS, VAE_WINDOW, t, &mut attn);
            l.o.linear(&attn, t, &mut o);
            add_into(&mut h, &o);
            rmsnorm(&h, &l.n2, t, HIDDEN, &mut a2);
            l.gate.linear(&a2, t, &mut gate);
            l.up.linear(&a2, t, &mut up);
            silu_mul(&gate, &up, &mut sil);
            l.down.linear(&sil, t, &mut o);
            add_into(&mut h, &o);
        }
        h
    }

    /// waveform (44.1 kHz, already padded) -> (mean, std, n_patch), each patch
    /// is 64 channels.
    pub fn encode_params(&self, waveform: &[f32]) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let n = waveform.len();
        let t_frames = n.div_ceil(ENC_IN_DIM);
        let mut frames = vec![0.0f32; t_frames * ENC_IN_DIM];
        for t in 0..t_frames {
            let start = t * ENC_IN_DIM;
            let end = (start + ENC_IN_DIM).min(n);
            frames[t * ENC_IN_DIM..t * ENC_IN_DIM + (end - start)]
                .copy_from_slice(&waveform[start..end]);
        }
        let mut e1 = vec![0.0f32; t_frames * HIDDEN];
        self.fc1.linear(&frames, t_frames, &mut e1);
        let mut e2 = vec![0.0f32; t_frames * HIDDEN];
        self.fc2.linear(&e1, t_frames, &mut e2);
        let h_raw = self.stack(&self.enc, &e2, t_frames);
        let mut h = vec![0.0f32; t_frames * HIDDEN];
        rmsnorm(&h_raw, &self.enc_norm, t_frames, HIDDEN, &mut h);

        // patch + cls frontend: row 5p+4 is the learned cls embedding
        let p_count = t_frames.div_ceil(ENC_PATCH);
        let padded = p_count * ENC_PATCH;
        let mut seq = vec![0.0f32; padded * 5 * HIDDEN];
        for p in 0..p_count {
            for f in 0..ENC_PATCH {
                let src = p * ENC_PATCH + f;
                if src < t_frames {
                    let dst = (p * 5 + f) * HIDDEN;
                    seq[dst..dst + HIDDEN].copy_from_slice(&h[src * HIDDEN..(src + 1) * HIDDEN]);
                }
            }
            let dst = (p * 5 + 4) * HIDDEN;
            seq[dst..dst + HIDDEN].copy_from_slice(&self.cls);
        }
        let a_raw = self.stack(&self.agg, &seq, p_count * 5);
        let mut a = vec![0.0f32; p_count * 5 * HIDDEN];
        rmsnorm(&a_raw, &self.agg_norm, p_count * 5, HIDDEN, &mut a);
        let mut cls_rows = vec![0.0f32; p_count * HIDDEN];
        for p in 0..p_count {
            let src = (p * 5 + 4) * HIDDEN;
            cls_rows[p * HIDDEN..(p + 1) * HIDDEN].copy_from_slice(&a[src..src + HIDDEN]);
        }
        let mut out = vec![0.0f32; p_count * 128];
        self.fc3.linear(&cls_rows, p_count, &mut out);
        let mut mean = vec![0.0f32; p_count * ZC];
        let mut std = vec![0.0f32; p_count * ZC];
        for i in 0..p_count * ZC {
            let row = i / ZC;
            let ch = i % ZC;
            mean[i] = out[row * 128 + ch];
            let scale = out[row * 128 + ZC + ch];
            // diffusers OobleckDiagonalGaussianDistribution: softplus + 1e-4
            let sp = if scale > 20.0 { scale } else { (scale.exp() + 1.0).ln() };
            std[i] = sp + 1e-4;
        }
        Ok((mean, std, p_count))
    }
}
