//! Tensor primitives shared by every CPU stage: the row-major Mat, CpuLin,
//! the GEMV/GEMM pair, normalisation, RoPE, attention and activations.
//!
//! Everything is plain f32 with weights widened from bf16 on load; rayon only
//! parallelises independent rows, so the numerics stay order-stable enough for
//! the 1e-6 parity budget.

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::qwen_lm::{HEAD_DIM, KV_CAP};
use crate::weights::Weights;

pub const RMS_EPS: f32 = 1e-6;

/// KV slots of the DiT attention caches (also the stride of their layout).
pub(super) const DIT_ROPE_CAP: usize = 512;

/// Row-major `[rows, cols]` f32 matrix (a widened bf16 weight).
pub struct Mat {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    pub fn load(w: &Weights, name: &str) -> Result<Self> {
        let t = w.get(name)?;
        if t.shape.len() != 2 {
            bail!("{name}: expected 2-D, got {:?}", t.shape);
        }
        let (rows, cols) = (t.shape[0], t.shape[1]);
        let data = t.to_f32()?;
        if data.len() != rows * cols {
            bail!("{name}: {} values for {rows}x{cols}", data.len());
        }
        Ok(Self { data, rows, cols })
    }

    /// y[m, rows] = x[m, cols] @ W^T + b, for any m >= 1.
    pub fn linear(&self, x: &[f32], m: usize, bias: Option<&[f32]>, out: &mut [f32]) {
        assert_eq!(x.len(), m * self.cols, "linear input shape");
        assert_eq!(out.len(), m * self.rows, "linear output shape");
        if m == 1 {
            self.gemv(x, out);
        } else {
            gemm_row_major(out, x, &self.data, m, self.rows, self.cols);
        }
        if let Some(b) = bias {
            out.par_chunks_mut(self.rows).for_each(|row| {
                for (o, &bi) in row.iter_mut().zip(b) {
                    *o += bi;
                }
            });
        }
    }

    /// One row of activations against every weight row, row-parallel.
    fn gemv(&self, x: &[f32], out: &mut [f32]) {
        let k = self.cols;
        let chunk = (self.rows / (rayon::current_num_threads() * 4)).clamp(32, 1024);
        out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slab)| {
            let r0 = ci * chunk;
            for (off, o) in slab.iter_mut().enumerate() {
                let row = &self.data[(r0 + off) * k..(r0 + off + 1) * k];
                *o = dot(row, x);
            }
        });
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 4-way accumulators then a tree sum: same shape as the GPU kernels (which
    // reduce 8 lanes in a tree), and enough ILP to keep a core busy.
    let quads = a.len() / 4;
    let mut acc = [0.0f32; 4];
    for q in 0..quads {
        let k = q * 4;
        acc[0] += a[k] * b[k];
        acc[1] += a[k + 1] * b[k + 1];
        acc[2] += a[k + 2] * b[k + 2];
        acc[3] += a[k + 3] * b[k + 3];
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for k in quads * 4..a.len() {
        s += a[k] * b[k];
    }
    s
}

/// C[m, n] = A[m, k] @ B[n, k]^T, with B the row-major weight matrix.
///
/// Weight-row outer, token inner: each weight row is read once and dotted
/// against every token, which is what keeps the traffic at one pass over the
/// checkpoint (the naive `for t { dot(all rows) }` version re-reads it m times).
/// Results are accumulated row-major by output row, then transposed into out.
fn gemm_row_major(out: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let mut tmp = vec![0.0f32; n * m];
    tmp.par_chunks_mut(m).enumerate().for_each(|(r, row)| {
        let w = &b[r * k..(r + 1) * k];
        for (t, o) in row.iter_mut().enumerate() {
            *o = dot(w, &a[t * k..(t + 1) * k]);
        }
    });
    out.par_chunks_mut(n).enumerate().for_each(|(t, row)| {
        for (r, o) in row.iter_mut().enumerate() {
            *o = tmp[r * m + t];
        }
    });
}

pub fn rmsnorm(x: &[f32], w: &[f32], rows: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), rows * dim);
    out.par_chunks_mut(dim).enumerate().for_each(|(r, o)| {
        let src = &x[r * dim..(r + 1) * dim];
        let mean = src.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (mean + RMS_EPS).sqrt();
        for i in 0..dim {
            o[i] = src[i] * inv * w[i];
        }
    });
}

/// Qwen2 rotate-half rope applied in place to `[tokens, n_heads, head_dim]`.
pub fn rope_split_half(x: &mut [f32], inv_freq: &[f32], n_heads: usize, pos_base: usize, tokens: usize) {
    let half = HEAD_DIM / 2;
    debug_assert_eq!(inv_freq.len(), half);
    for t in 0..tokens {
        let pos = (pos_base + t) as f32;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * HEAD_DIM;
            for i in 0..half {
                let (s, c) = (pos * inv_freq[i]).sin_cos();
                let a = x[base + i];
                let b = x[base + i + half];
                x[base + i] = a * c - b * s;
                x[base + i + half] = b * c + a * s;
            }
        }
    }
}

/// Causal GQA attention for `t` new tokens appended at `pos_base`.
///
/// `k_cache` / `v_cache` are `[KV_HEADS, KV_CAP, HEAD_DIM]` and already hold the
/// keys/values for positions `0..pos_base`; the new ones must have been written
/// by the caller. Writes `y[t, Q_HEADS * HEAD_DIM]`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    t: usize,
    pos_base: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    y: &mut [f32],
) {
    attention_window(q, k_cache, v_cache, t, pos_base, n_q_heads, n_kv_heads, 0, KV_CAP, y)
}

/// Causal attention with an optional sliding window (window = 0 means full
/// causal; the VAE towers use 64).
#[allow(clippy::too_many_arguments)]
pub fn attention_window(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    t: usize,
    pos_base: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    window: usize,
    cap: usize,
    y: &mut [f32],
) {
    let _ = t;
    let hd = HEAD_DIM;
    let group = n_q_heads / n_kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    y.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
        let t_i = idx / n_q_heads;
        let head = idx % n_q_heads;
        let kvh = head / group;
        let qpos = pos_base + t_i;
        let lo = if window == 0 || qpos < window {
            0
        } else {
            qpos + 1 - window
        };
        let qbase = idx * hd;
        let kbase = kvh * cap * hd;

        let mut scores = vec![0.0f32; qpos + 1 - lo];
        let mut mx = f32::NEG_INFINITY;
        for j in lo..=qpos {
            let kb = kbase + j * hd;
            let mut s = 0.0f32;
            for d in 0..hd {
                s += q[qbase + d] * k_cache[kb + d];
            }
            s *= scale;
            if s > mx {
                mx = s;
            }
            scores[j - lo] = s;
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for d in 0..hd {
            let mut acc = 0.0f32;
            for j in lo..=qpos {
                acc += scores[j - lo] * v_cache[kbase + j * hd + d];
            }
            out[d] = acc * inv;
        }
    });
}

pub(super) fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let g = gate[i];
        *o = g / (1.0 + (-g).exp()) * up[i];
    });
}

pub(super) fn add_into(dst: &mut [f32], src: &[f32]) {
    dst.par_iter_mut().zip(src).for_each(|(d, s)| *d += s);
}

pub(super) fn timestep_embedding(t: f32) -> Vec<f32> {
    let mut temb = vec![0f32; 256];
    for j in 0..128 {
        let e = ((j as f32) * -(10000f32.ln()) / 127.0).exp();
        let v = 1000.0 * t * e;
        temb[j] = v.sin();
        temb[128 + j] = v.cos();
    }
    temb
}

pub(super) fn gelu_tanh(x: &[f32], out: &mut [f32]) {
    out.par_iter_mut().zip(x).for_each(|(o, &v)| {
        let beta = 0.797_884_6_f32;
        let kappa = 0.044715f32;
        *o = 0.5 * v * (1.0 + (v * beta * (1.0 + v * v * kappa)).tanh());
    });
}


// ---------------------------------------------------------------------------
// DiT flow-matching head + aggregator (hand-written twin of dit.rs)
// ---------------------------------------------------------------------------

/// A linear layer with its bias (x_transformers' Linear defaults to bias=True;
/// 53 of the DiT's tensors are biases, so dropping them is not an option).
pub struct CpuLin {
    pub w: Mat,
    pub b: Vec<f32>,
}

impl CpuLin {
    pub(super) fn load(w: &Weights, base: &str, rows: usize, cols: usize) -> Result<Self> {
        let m = Mat::load(w, &format!("{base}.weight"))?;
        if m.rows != rows || m.cols != cols {
            bail!("{base}: {:?}x{:?}, expected {rows}x{cols}", m.rows, m.cols);
        }
        let b = match w.get(&format!("{base}.bias")) {
            Ok(t) => t.to_f32()?,
            Err(_) => vec![0.0; rows],
        };
        Ok(Self { w: m, b })
    }

    pub(super) fn linear(&self, x: &[f32], m: usize, out: &mut [f32]) {
        self.w.linear(x, m, Some(&self.b), out);
    }
}

/// Bidirectional attention over all rows (the DiT's non-grouped path and the
/// aggregator). `attention` above is the LLM's *causal* one -- using it here
/// was the reason the single-branch DiT forward disagreed with the GPU.
pub(super) fn attention_full(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    t: usize,
    n_q_heads: usize,
    y: &mut [f32],
) {
    let hd = HEAD_DIM;
    let scale = 1.0 / (hd as f32).sqrt();
    y.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
        let head = idx % n_q_heads;
        let qbase = idx * hd;
        let kbase = head * DIT_ROPE_CAP * hd;
        let mut scores = vec![0.0f32; t];
        let mut mx = f32::NEG_INFINITY;
        for (j, sj) in scores.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for d in 0..hd {
                s += q[qbase + d] * k_cache[kbase + j * hd + d];
            }
            s *= scale;
            if s > mx {
                mx = s;
            }
            *sj = s;
        }
        let mut sum = 0.0f32;
        for sj in scores.iter_mut() {
            *sj = (*sj - mx).exp();
            sum += *sj;
        }
        let inv = 1.0 / sum;
        for d in 0..hd {
            let mut acc = 0.0f32;
            for j in 0..t {
                acc += scores[j] * v_cache[kbase + j * hd + d];
            }
            out[d] = acc * inv;
        }
    });
}

/// Causal/grouped attention used by the DiT: rows are packed into groups of
/// `group` (3 for the CFG pair); each row attends only inside its own group.
pub(super) fn attention_group(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    _t: usize,
    group: usize,
    n_q_heads: usize,
    y: &mut [f32],
) {
    let hd = HEAD_DIM;
    let scale = 1.0 / (hd as f32).sqrt();
    y.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
        let row = idx / n_q_heads;
        let head = idx % n_q_heads;
        let g = row / group;
        let lo = g * group;
        let hi = lo + group - 1;
        let qbase = idx * hd;
        let kbase = head * DIT_ROPE_CAP * hd;
        let mut scores = vec![0.0f32; group];
        let mut mx = f32::NEG_INFINITY;
        for (j, sj) in scores.iter_mut().enumerate() {
            let p = lo + j;
            let kb = kbase + p * hd;
            let mut s = 0.0f32;
            for d in 0..hd {
                s += q[qbase + d] * k_cache[kb + d];
            }
            s *= scale;
            if s > mx {
                mx = s;
            }
            *sj = s;
        }
        let mut sum = 0.0f32;
        for sj in scores.iter_mut() {
            *sj = (*sj - mx).exp();
            sum += *sj;
        }
        let inv = 1.0 / sum;
        for d in 0..hd {
            let mut acc = 0.0f32;
            for j in 0..group {
                acc += scores[j] * v_cache[kbase + (lo + j) * hd + d];
            }
            out[d] = acc * inv;
        }
        let _ = hi;
    });
}
