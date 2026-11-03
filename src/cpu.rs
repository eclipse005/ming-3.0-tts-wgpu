//! Hand-written CPU twin of the GPU pipeline (same approach as
//! `qwen3-asr-wgpu/src/cpu_decoder.rs`): f32 activations, bf16 weights widened
//! on load, rayon for row parallelism and the `gemm` crate for M>1 batches.
//! No NN framework, no wgpu, no device.
//!
//! Stage 1 (this file): tensor primitives + the Qwen2.5-0.5B decoder (prefill
//! and incremental decode) with the stop head. The DiT/CFM and VAE towers get
//! the same treatment in later stages.

#![allow(dead_code)]

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::qwen_lm::{FF, HEAD_DIM, HIDDEN, KV_CAP, KV_HEADS, LAYERS, Q_HEADS, ROPE_THETA};
use crate::weights::Weights;

pub const RMS_EPS: f32 = 1e-6;

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
        let chunk = (self.rows / (rayon::current_num_threads() * 4)).max(32).min(1024);
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
/// checkpoint (the naive or t { dot(all rows) } version re-reads it m times).
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
        let lo = if window == 0 || qpos + 1 <= window {
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

fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let g = gate[i];
        *o = g / (1.0 + (-g).exp()) * up[i];
    });
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    dst.par_iter_mut().zip(src).for_each(|(d, s)| *d += s);
}

fn timestep_embedding(t: f32) -> Vec<f32> {
    let mut temb = vec![0f32; 256];
    for j in 0..128 {
        let e = ((j as f32) * -(10000f32.ln()) / 127.0).exp();
        let v = 1000.0 * t * e;
        temb[j] = v.sin();
        temb[128 + j] = v.cos();
    }
    temb
}

fn gelu_tanh(x: &[f32], out: &mut [f32]) {
    out.par_iter_mut().zip(x).for_each(|(o, &v)| {
        let beta = 0.7978845608028654f32;
        let kappa = 0.044715f32;
        *o = 0.5 * v * (1.0 + (v * beta * (1.0 + v * v * kappa)).tanh());
    });
}

struct CpuLayer {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
}

impl CpuLayer {
    fn load(w: &Weights, i: usize) -> Result<Self> {
        let base = format!("model.model.layers.{i}");
        let lin = |what: &str| Mat::load(w, &format!("{base}.{what}.weight"));
        let bias = |what: &str, rows: usize| -> Result<Vec<f32>> {
            w.get(&format!("{base}.{what}.bias"))
                .map(|t| t.to_f32())
                .unwrap_or_else(|_| Ok(vec![0.0; rows]))
        };
let q = lin("self_attn.q_proj")?;
        let k = lin("self_attn.k_proj")?;
        let v = lin("self_attn.v_proj")?;
        let o = lin("self_attn.o_proj")?;
        let gate = lin("mlp.gate_proj")?;
        let up = lin("mlp.up_proj")?;
        let down = lin("mlp.down_proj")?;
                let norm1 = w.get(&format!("{base}.input_layernorm.weight"))?.to_f32()?;
        let norm2 = w.get(&format!("{base}.post_attention_layernorm.weight"))?.to_f32()?;
        Ok(Self {
            q, k, v, o, gate, up, down,
            norm1, norm2,
            q_bias: bias("self_attn.q_proj", HIDDEN)?,
            k_bias: bias("self_attn.k_proj", KV_HEADS * HEAD_DIM)?,
            v_bias: bias("self_attn.v_proj", KV_HEADS * HEAD_DIM)?,
            kc: vec![0.0; KV_HEADS * KV_CAP * HEAD_DIM],
            vc: vec![0.0; KV_HEADS * KV_CAP * HEAD_DIM],
        })
    }
}

/// Qwen2.5-0.5B decoder on the host.
pub struct CpuLlm {
    layers: Vec<CpuLayer>,
    norm_final: Vec<f32>,
    stop_w: Vec<f32>,
    stop_b: Vec<f32>,
    inv_freq: Vec<f32>,
}

impl CpuLlm {
    pub fn load(w: &Weights) -> Result<Self> {
        let layers = (0..LAYERS)
            .map(|i| {
                CpuLayer::load(w, i)
            })
            .collect::<Result<Vec<_>>>()?;
        let inv_freq: Vec<f32> = (0..HEAD_DIM / 2)
            .map(|i| ROPE_THETA.powf((2 * i) as f64 / HEAD_DIM as f64).recip() as f32)
            .collect();
        let norm_final = w.get("model.model.norm.weight")?.to_f32()?;
        let stop_w = w.get("stop_head.weight")?.to_f32()?;
        let stop_b = w.get("stop_head.bias")?.to_f32()?;
        Ok(Self { layers, norm_final, stop_w, stop_b, inv_freq })
    }

    /// Run `t` tokens starting at `pos_base`, returning the last hidden state
    /// (post final norm) and the stop probabilities.
    fn forward(&mut self, x: &[f32], t: usize, pos_base: usize) -> Result<(Vec<f32>, [f32; 2])> {
        let mut h = x.to_vec();
        let mut a = vec![0.0f32; t * HIDDEN];
        let mut q = vec![0.0f32; t * Q_HEADS * HEAD_DIM];
        let mut k = vec![0.0f32; t * KV_HEADS * HEAD_DIM];
        let mut v = vec![0.0f32; t * KV_HEADS * HEAD_DIM];
        let mut attn = vec![0.0f32; t * HIDDEN];
        let mut o = vec![0.0f32; t * HIDDEN];
        let mut gate = vec![0.0f32; t * FF];
        let mut up = vec![0.0f32; t * FF];
        let mut sil = vec![0.0f32; t * FF];
        let mut a2 = vec![0.0f32; t * HIDDEN];

        for l in self.layers.iter_mut() {
            rmsnorm(&h, &l.norm1, t, HIDDEN, &mut a);
            l.q.linear(&a, t, Some(&l.q_bias), &mut q);
            l.k.linear(&a, t, Some(&l.k_bias), &mut k);
            l.v.linear(&a, t, Some(&l.v_bias), &mut v);
            rope_split_half(&mut q, &self.inv_freq, Q_HEADS, pos_base, t);
            rope_split_half(&mut k, &self.inv_freq, KV_HEADS, pos_base, t);
            // append the new keys/values
            for tt in 0..t {
                for hh in 0..KV_HEADS {
                    let dst = hh * KV_CAP * HEAD_DIM + (pos_base + tt) * HEAD_DIM;
                    let src = (tt * KV_HEADS + hh) * HEAD_DIM;
                    l.kc[dst..dst + HEAD_DIM].copy_from_slice(&k[src..src + HEAD_DIM]);
                    l.vc[dst..dst + HEAD_DIM].copy_from_slice(&v[src..src + HEAD_DIM]);
                }
            }
            attention(&q, &l.kc, &l.vc, t, pos_base, Q_HEADS, KV_HEADS, &mut attn);
            l.o.linear(&attn, t, None, &mut o);
            add_into(&mut h, &o);

            rmsnorm(&h, &l.norm2, t, HIDDEN, &mut a2);
            l.gate.linear(&a2, t, None, &mut gate);
            l.up.linear(&a2, t, None, &mut up);
            silu_mul(&gate, &up, &mut sil);
            l.down.linear(&sil, t, None, &mut o);
            add_into(&mut h, &o);
        }

        rmsnorm(&h, &self.norm_final, t, HIDDEN, &mut a);
        let last = &a[(t - 1) * HIDDEN..t * HIDDEN];
        let mut logits = [0.0f32; 2];
        for r in 0..2 {
            let mut s = self.stop_b[r];
            for i in 0..HIDDEN {
                s += self.stop_w[r * HIDDEN + i] * last[i];
            }
            logits[r] = s;
        }
        let mx = logits[0].max(logits[1]);
        let e0 = (logits[0] - mx).exp();
        let e1 = (logits[1] - mx).exp();
        Ok((last.to_vec(), [e0 / (e0 + e1), e1 / (e0 + e1)]))
    }

    pub fn prefill(&mut self, embeds: &[f32], t: usize) -> Result<(Vec<f32>, [f32; 2])> {
        self.forward(embeds, t, 0)
    }

    pub fn step(&mut self, embed: &[f32], pos: usize) -> Result<(Vec<f32>, [f32; 2])> {
        self.forward(embed, 1, pos)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small_matches_manual() {
        // W = [[1,2,3],[4,5,6]] (2 rows x 3 cols), X = [[1,0,0],[0,1,0]] (m=2)
        let w = Mat { data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rows: 2, cols: 3 };
        let x = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut out = vec![0.0f32; 4];
        w.linear(&x, 2, None, &mut out);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn gemv_small_matches_manual() {
        let w = Mat { data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rows: 2, cols: 3 };
        let x = vec![1.0, 10.0, 100.0];
        let mut out = vec![0.0f32; 2];
        w.linear(&x, 1, None, &mut out);
        assert_eq!(out, vec![321.0, 654.0]);
    }
}


// ---------------------------------------------------------------------------
// DiT flow-matching head + aggregator (hand-written twin of dit.rs)
// ---------------------------------------------------------------------------

pub const XH: usize = 1024;
pub const ZC: usize = 64;
pub const XHEADS: usize = 16;
pub const XFF: usize = 4096;
pub const DIT_STEPS: usize = 10;
/// DiT rope is interleaved (GPT-J style) with the checkpoint's own inv_freq.
pub const DIT_ROPE_CAP: usize = 512;

/// A linear layer with its bias (x_transformers' Linear defaults to bias=True;
/// 53 of the DiT's tensors are biases, so dropping them is not an option).
pub struct CpuLin {
    pub w: Mat,
    pub b: Vec<f32>,
}

impl CpuLin {
    fn load(w: &Weights, base: &str, rows: usize, cols: usize) -> Result<Self> {
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

    fn linear(&self, x: &[f32], m: usize, out: &mut [f32]) {
        self.w.linear(x, m, Some(&self.b), out);
    }
}

struct CpuXBlock {
    q: CpuLin,
    k: CpuLin,
    v: CpuLin,
    o: CpuLin,
    ff_in: CpuLin,
    ff_out: CpuLin,
    n1: Vec<f32>,
    n2: Vec<f32>,
}

impl CpuXBlock {
    fn load(w: &Weights, prefix: &str, i: usize) -> Result<Self> {
        let b = format!("{prefix}.blocks.{i}");
        Ok(Self {
            q: CpuLin::load(w, &format!("{b}.attn.to_q"), XH, XH)?,
            k: CpuLin::load(w, &format!("{b}.attn.to_k"), XH, XH)?,
            v: CpuLin::load(w, &format!("{b}.attn.to_v"), XH, XH)?,
            o: CpuLin::load(w, &format!("{b}.attn.to_out.0"), XH, XH)?,
            ff_in: CpuLin::load(w, &format!("{b}.mlp.ff.0.0"), XFF, XH)?,
            ff_out: CpuLin::load(w, &format!("{b}.mlp.ff.2"), XH, XFF)?,
            n1: w.get(&format!("{b}.norm1.weight"))?.to_f32()?,
            n2: w.get(&format!("{b}.norm2.weight"))?.to_f32()?,
        })
    }
}

/// Interleaved rope (pairs 2i, 2i+1) with the checkpoint's inv_freq.
fn rope_interleaved(x: &mut [f32], inv_freq: &[f32], n_heads: usize, pos_base: usize, tokens: usize) {
    let half = HEAD_DIM / 2;
    for t in 0..tokens {
        let pos = (pos_base + t) as f32;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * HEAD_DIM;
            for i in 0..half {
                let (s, c) = (pos * inv_freq[i]).sin_cos();
                let a = x[base + 2 * i];
                let b = x[base + 2 * i + 1];
                x[base + 2 * i] = a * c - b * s;
                x[base + 2 * i + 1] = b * c + a * s;
            }
        }
    }
}

/// Bidirectional attention over all 	 rows (the DiT's non-grouped path and the
/// aggregator). ttention above is the LLM's *causal* one -- using it here was
/// the reason the single-branch DiT forward disagreed with the GPU.
fn attention_full(
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

/// Causal/grouped attention used by the DiT: rows 	 are packed into groups of
/// group (3 for the CFG pair); each row attends only inside its own group.
fn attention_group(
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

pub struct CpuDit {
    blocks: Vec<CpuXBlock>,
    t_mlp1: CpuLin,
    t_mlp2: CpuLin,
    x_emb: CpuLin,
    c_emb: CpuLin,
    fin_norm: Vec<f32>,
    fin_lin: CpuLin,
    agg_word: Vec<f32>,
    agg_x_emb: CpuLin,
    agg_blocks: Vec<CpuXBlock>,
    agg_fin_norm: Vec<f32>,
    agg_fin_lin: CpuLin,
    inv_freq: Vec<f32>,
}

impl CpuDit {
    pub fn load(w: &Weights) -> Result<Self> {
        let cfm = "flowloss.cfm.model";
        let blocks = (0..8)
            .map(|i| CpuXBlock::load(w, cfm, i))
            .collect::<Result<Vec<_>>>()?;
        let agg_blocks = (0..8)
            .map(|i| CpuXBlock::load(w, "linear_proj_audio", i))
            .collect::<Result<Vec<_>>>()?;
        let inv_freq = w.get("flowloss.cfm.model.rotary_embed.inv_freq")?.to_f32()?;
        Ok(Self {
            blocks,
            t_mlp1: CpuLin::load(w, &format!("{cfm}.t_embedder.time_mlp.0"), XH, 256)?,
            t_mlp2: CpuLin::load(w, &format!("{cfm}.t_embedder.time_mlp.2"), XH, XH)?,
            x_emb: CpuLin::load(w, &format!("{cfm}.x_embedder"), XH, ZC)?,
            c_emb: CpuLin::load(w, &format!("{cfm}.c_embedder.cond_embedder"), XH, HIDDEN)?,
            fin_norm: w.get(&format!("{cfm}.final_layer.norm_final.weight"))?.to_f32()?,
            fin_lin: CpuLin::load(w, &format!("{cfm}.final_layer.linear"), ZC, XH)?,
            agg_word: w.get("linear_proj_audio.word_embedder.weight")?.to_f32()?,
            agg_x_emb: CpuLin::load(w, "linear_proj_audio.x_embedder", XH, ZC)?,
            agg_blocks,
            agg_fin_norm: w.get("linear_proj_audio.final_layer.norm_final.weight")?.to_f32()?,
            agg_fin_lin: CpuLin::load(w, "linear_proj_audio.final_layer.linear", HIDDEN, XH)?,
            inv_freq,
        })
    }

    fn block_stack(&self, blocks: &[CpuXBlock], seq: &[f32], t: usize, group: usize) -> Vec<f32> {
        let mut h = seq.to_vec();
        let mut a = vec![0.0f32; t * XH];
        let mut q = vec![0.0f32; t * XH];
        let mut k = vec![0.0f32; t * XH];
        let mut v = vec![0.0f32; t * XH];
        let mut attn = vec![0.0f32; t * XH];
        let mut o = vec![0.0f32; t * XH];
        let mut gate = vec![0.0f32; t * XFF];
        let mut g = vec![0.0f32; t * XFF];
        let mut a2 = vec![0.0f32; t * XH];
        let mut kc = vec![0.0f32; XHEADS * DIT_ROPE_CAP * HEAD_DIM];
        let mut vc = vec![0.0f32; XHEADS * DIT_ROPE_CAP * HEAD_DIM];
        for b in blocks {
            rmsnorm(&h, &b.n1, t, XH, &mut a);
            b.q.linear(&a, t, &mut q);
            b.k.linear(&a, t, &mut k);
            b.v.linear(&a, t, &mut v);
            if group == 0 {
                rope_interleaved(&mut q, &self.inv_freq, XHEADS, 0, t);
                rope_interleaved(&mut k, &self.inv_freq, XHEADS, 0, t);
            } else {
                let mut row = 0;
                while row < t {
                    rope_interleaved(&mut q[row * XH..(row + group) * XH], &self.inv_freq, XHEADS, 0, group);
                    rope_interleaved(&mut k[row * XH..(row + group) * XH], &self.inv_freq, XHEADS, 0, group);
                    row += group;
                }
            }
            for tt in 0..t {
                for hh in 0..XHEADS {
                    let dst = hh * DIT_ROPE_CAP * HEAD_DIM + tt * HEAD_DIM;
                    let src = (tt * XHEADS + hh) * HEAD_DIM;
                    kc[dst..dst + HEAD_DIM].copy_from_slice(&k[src..src + HEAD_DIM]);
                    vc[dst..dst + HEAD_DIM].copy_from_slice(&v[src..src + HEAD_DIM]);
                }
            }
            if group == 0 {
                attention_full(&q, &kc, &vc, t, XHEADS, &mut attn);
            } else {
                attention_group(&q, &kc, &vc, t, group, XHEADS, &mut attn);
            }
            b.o.linear(&attn, t, &mut o);
            add_into(&mut h, &o);
            rmsnorm(&h, &b.n2, t, XH, &mut a2);
            b.ff_in.linear(&a2, t, &mut gate);
            gelu_tanh(&gate, &mut g);
            b.ff_out.linear(&g, t, &mut o);
            add_into(&mut h, &o);
        }
        h
    }

    /// One (cond, null) pair forward: rows 0..2 are the conditional branch,
    /// 3..5 the null one; predictions come back in rows 2 and 5.
    pub fn forward_pair(&self, x: &[f32], tstep: f32, c: &[f32], lh: &[f32]) -> Vec<f32> {
        let temb = timestep_embedding(tstep);
        let mut tm1 = vec![0.0f32; XH];
        self.t_mlp1.linear(&temb, 1, &mut tm1);
        let mut sil = vec![0.0f32; XH];
        silu_mul(&tm1, &vec![1.0f32; XH], &mut sil);
        let mut tm2 = vec![0.0f32; XH];
        self.t_mlp2.linear(&sil, 1, &mut tm2);
        let mut xn = vec![0.0f32; XH];
        self.x_emb.linear(x, 1, &mut xn);
        let mut xh = vec![0.0f32; XH];
        self.x_emb.linear(lh, 1, &mut xh);
        let mut ce_c = vec![0.0f32; XH];
        self.c_emb.linear(c, 1, &mut ce_c);
        let mut ce_n = vec![0.0f32; XH];
        self.c_emb.linear(&vec![0.0f32; HIDDEN], 1, &mut ce_n);
        let mut seq = vec![0.0f32; 6 * XH];
        for i in 0..XH {
            seq[i] = tm2[i] + ce_c[i];
            seq[XH + i] = xh[i];
            seq[2 * XH + i] = xn[i];
            seq[3 * XH + i] = tm2[i] + ce_n[i];
            seq[4 * XH + i] = xh[i];
            seq[5 * XH + i] = xn[i];
        }
        let h = self.block_stack(&self.blocks, &seq, 6, 3);
        let mut fin = vec![0.0f32; 6 * XH];
        rmsnorm(&h, &self.fin_norm, 6, XH, &mut fin);
        let mut out = vec![0.0f32; 6 * ZC];
        self.fin_lin.linear(&fin, 6, &mut out);
        out
    }

    /// Block-0 sub-operator dump (same order as the GPU's lock0_parts).
    pub fn block0_parts(&self, x: &[f32], tstep: f32, c: &[f32], lh: &[f32]) -> Vec<(String, Vec<f32>)> {
        let temb = timestep_embedding(tstep);
        let mut tm1 = vec![0.0f32; XH];
        self.t_mlp1.linear(&temb, 1, &mut tm1);
        let mut sil = vec![0.0f32; XH];
        silu_mul(&tm1, &vec![1.0f32; XH], &mut sil);
        let mut tm2 = vec![0.0f32; XH];
        self.t_mlp2.linear(&sil, 1, &mut tm2);
        let mut xn = vec![0.0f32; XH];
        self.x_emb.linear(x, 1, &mut xn);
        let mut xh = vec![0.0f32; XH];
        self.x_emb.linear(lh, 1, &mut xh);
        let mut ce = vec![0.0f32; XH];
        self.c_emb.linear(c, 1, &mut ce);
        let mut seq = vec![0.0f32; 3 * XH];
        for i in 0..XH {
            seq[i] = tm2[i] + ce[i];
            seq[XH + i] = xh[i];
            seq[2 * XH + i] = xn[i];
        }
        let b = &self.blocks[0];
        let mut out = Vec::new();
        let mut a = vec![0.0f32; 3 * XH];
        rmsnorm(&seq, &b.n1, 3, XH, &mut a);
        out.push(("norm1".to_string(), a.clone()));
        let mut q = vec![0.0f32; 3 * XH];
        let mut k = vec![0.0f32; 3 * XH];
        let mut v = vec![0.0f32; 3 * XH];
        b.q.linear(&a, 3, &mut q);
        b.k.linear(&a, 3, &mut k);
        b.v.linear(&a, 3, &mut v);
        out.push(("q".to_string(), q.clone()));
        out.push(("k".to_string(), k.clone()));
        out.push(("v".to_string(), v.clone()));
        rope_interleaved(&mut q, &self.inv_freq, XHEADS, 0, 3);
        rope_interleaved(&mut k, &self.inv_freq, XHEADS, 0, 3);
        out.push(("q_rope".to_string(), q.clone()));
        out.push(("k_rope".to_string(), k.clone()));
        let mut kc = vec![0.0f32; XHEADS * DIT_ROPE_CAP * HEAD_DIM];
        let mut vc = vec![0.0f32; XHEADS * DIT_ROPE_CAP * HEAD_DIM];
        for tt in 0..3 {
            for hh in 0..XHEADS {
                let dst = hh * DIT_ROPE_CAP * HEAD_DIM + tt * HEAD_DIM;
                let src = (tt * XHEADS + hh) * HEAD_DIM;
                kc[dst..dst + HEAD_DIM].copy_from_slice(&k[src..src + HEAD_DIM]);
                vc[dst..dst + HEAD_DIM].copy_from_slice(&v[src..src + HEAD_DIM]);
            }
        }
        let mut attn = vec![0.0f32; 3 * XH];
        attention_full(&q, &kc, &vc, 3, XHEADS, &mut attn);
        let mut o = vec![0.0f32; 3 * XH];
        b.o.linear(&attn, 3, &mut o);
        out.push(("attn".to_string(), o.clone()));
        let mut h = seq.clone();
        add_into(&mut h, &o);
        let mut a2 = vec![0.0f32; 3 * XH];
        rmsnorm(&h, &b.n2, 3, XH, &mut a2);
        out.push(("norm2".to_string(), a2.clone()));
        let mut fi = vec![0.0f32; 3 * XFF];
        b.ff_in.linear(&a2, 3, &mut fi);
        out.push(("ff_pre".to_string(), fi.clone()));
        let mut g = vec![0.0f32; 3 * XFF];
        gelu_tanh(&fi, &mut g);
        out.push(("gelu".to_string(), g.clone()));
        let mut fo = vec![0.0f32; 3 * XH];
        b.ff_out.linear(&g, 3, &mut fo);
        out.push(("ff_out".to_string(), fo.clone()));
        let mut bout = h.clone();
        add_into(&mut bout, &fo);
        out.push(("out".to_string(), bout));
        out
    }

    /// Single-branch (3-row) forward, the reference path the GPU tests use.
    pub fn forward_single(&self, x: &[f32], tstep: f32, c: &[f32], lh: &[f32]) -> Vec<f32> {
        let temb = timestep_embedding(tstep);
        let mut tm1 = vec![0.0f32; XH];
        self.t_mlp1.linear(&temb, 1, &mut tm1);
        let mut sil = vec![0.0f32; XH];
        silu_mul(&tm1, &vec![1.0f32; XH], &mut sil);
        let mut tm2 = vec![0.0f32; XH];
        self.t_mlp2.linear(&sil, 1, &mut tm2);
        let mut xn = vec![0.0f32; XH];
        self.x_emb.linear(x, 1, &mut xn);
        let mut xh = vec![0.0f32; XH];
        self.x_emb.linear(lh, 1, &mut xh);
        let mut ce = vec![0.0f32; XH];
        self.c_emb.linear(c, 1, &mut ce);
        let mut seq = vec![0.0f32; 3 * XH];
        for i in 0..XH {
            seq[i] = tm2[i] + ce[i];
            seq[XH + i] = xh[i];
            seq[2 * XH + i] = xn[i];
        }
        let h = self.block_stack(&self.blocks, &seq, 3, 0);
        let mut fin = vec![0.0f32; 3 * XH];
        rmsnorm(&h, &self.fin_norm, 3, XH, &mut fin);
        let mut out = vec![0.0f32; 3 * ZC];
        self.fin_lin.linear(&fin, 3, &mut out);
        out[2 * ZC..3 * ZC].to_vec()
    }

    /// CFM Euler integration (epss timesteps + sway), same schedule as dit.rs.
    pub fn sample(&self, z: &[f32], lh: &[f32], noise: &[f32], sde: &[f32], cfg: f32, sigma: f32, temperature: f32) -> Vec<f32> {
        let idx = [0usize, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32];
        let dtv = 1.0f32 / 32.0;
        let mut tv = [0f32; DIT_STEPS + 1];
        for (i, &ix) in idx.iter().enumerate() {
            tv[i] = dtv * ix as f32;
        }
        for v in tv.iter_mut() {
            *v += -1.0 * ((std::f32::consts::PI / 2.0 * *v).cos() - 1.0 + *v);
        }
        let mut y = noise.to_vec();
        for step in 0..DIT_STEPS {
            let dt = tv[step + 1] - tv[step];
            let pair = self.forward_pair(&y, tv[step], z, lh);
            let pred = &pair[2 * ZC..3 * ZC];
            let null = &pair[5 * ZC..6 * ZC];
            let k = if temperature > 0.0 {
                sigma * temperature.sqrt() * dt.abs().sqrt()
            } else {
                0.0
            };
            for i in 0..ZC {
                let v = pred[i] + (pred[i] - null[i]) * cfg;
                y[i] += v * dt;
                if k != 0.0 {
                    y[i] += k * sde[step * ZC + i];
                }
            }
        }
        y
    }

    /// Aggregator: latent [64] -> projected llm condition [896].
    pub fn aggregate(&self, latent: &[f32]) -> Vec<f32> {
        let mut xn = vec![0.0f32; XH];
        self.agg_x_emb.linear(latent, 1, &mut xn);
        let mut seq = vec![0.0f32; 2 * XH];
        seq[..XH].copy_from_slice(&self.agg_word);
        seq[XH..].copy_from_slice(&xn);
        let h = self.block_stack(&self.agg_blocks, &seq, 2, 0);
        let mut fin = vec![0.0f32; 2 * XH];
        rmsnorm(&h, &self.agg_fin_norm, 2, XH, &mut fin);
        let mut out = vec![0.0f32; 2 * HIDDEN];
        self.agg_fin_lin.linear(&fin, 2, &mut out);
        out[..HIDDEN].to_vec()
    }
}


// ---------------------------------------------------------------------------
// VAE decoder (hand-written twin of vae_dec.rs): fc1 -> x4 upsample -> 24-layer
// causal stack (window 64) -> ISTFT head -> CPU iSTFT.
// ---------------------------------------------------------------------------

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
    pub fn decode_loop(
        &mut self,
        embeds: &[f32],
        t: usize,
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
        let mut history = vec![0.0f32; ZC];
        let mut latents: Vec<f32> = Vec::new();
        let mut pos = t;
        for step in 0..steps {
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
            pos += 1;
            history.copy_from_slice(&lat);
        }
        Ok(latents)
    }

    pub fn vocode(&self, latents: &[f32]) -> Vec<f32> {
        let n = latents.len() / ZC;
        self.dec.decode(latents, n)
    }
}

/// --device cpu: same CLI surface, host pipeline. Prompt/clone conditioning is
/// not wired yet (it needs the VAE encoder), and says so instead of silently
/// producing the wrong voice.
pub fn run(a: &crate::synth::Args, model_dir: &std::path::Path) -> Result<()> {
    use crate::tasks;
    // Same guard as the wgpu path: a .pt speaker embedding needs torch's
    // serialisation format, which this port does not ship.
    if !a.spk_emb_path.is_empty() && a.use_spk_emb {
        bail!(
            "--spk_emb_path needs a .pt reader, which this port does not ship yet; \
             use --prompt_wav_path with --use_spk_emb instead"
        );
    }
    let tok = crate::tok::Tok::load(model_dir)?;
    let controls = {
        let c = tasks::Controls {
            speed: a.speed.clone(),
            pitch: a.pitch.clone(),
            volume: a.volume.clone(),
            emotion: a.emotion.clone(),
            dialect: a.dialect.clone(),
            style: a.style.clone(),
            ip: a.ip.clone(),
        };
        if c.speed.is_none()
            && c.pitch.is_none()
            && c.volume.is_none()
            && c.emotion.is_none()
            && c.dialect.is_none()
            && c.style.is_none()
            && c.ip.is_none()
        {
            None
        } else {
            Some(c)
        }
    };
    let req = tasks::build_task_generation_kwargs(
        &a.task,
        &a.text,
        a.language_id.as_deref(),
        a.instruction.as_deref(),
        controls.as_ref(),
        a.instruction_schema.as_deref(),
        a.prompt_wav_path.first().map(|s| s.as_str()),
        a.prompt_text.as_deref(),
        None,
        a.use_spk_emb,
        a.use_zero_spk_emb,
        a.cfg,
        a.sigma,
        a.temperature,
    )?;
    let fragments = crate::synth::prepare_text(&tok, &req, a)?;
    // STYLE re-conditions every later fragment on the first generated chunk,
    // exactly like ming_tts.synthesize (see synth.rs).
    let reference_first_fragment = req.taskname == "STYLE" && fragments.len() > 1;
    let mut rng: Box<dyn crate::pipeline::Noise> = Box::new(crate::pipeline::Torch::new(a.seed));

    println!("device: cpu (hand-written Rust backend, rayon)");
    println!("fragments: {}", fragments.len());
    let t0 = std::time::Instant::now();
    let w = crate::weights::Weights::load(&model_dir.join("model.safetensors"))?;
    let mut pl = CpuPipeline::load(&w)?;
    println!("  model loaded in {:?}", t0.elapsed());

    // ---- voice conditioning (same rules as synth.rs) ----------------------
    let mut spk_row: Option<Vec<f32>> = None;
    let mut prompt_latent: Option<(Vec<f32>, usize)> = None;
    let mut prompt_text = req.prompt_text.clone();
    if a.use_zero_spk_emb {
        spk_row = Some(vec![0.0f32; HIDDEN]);
    }
    if !a.prompt_wav_path.is_empty() {
        let mut at_44k: Vec<f32> = Vec::new();
        let mut for_spk: Vec<f32> = Vec::new();
        for path in &a.prompt_wav_path {
            let (x, r) = crate::audio::read_wav(path)?;
            if a.use_spk_emb {
                for_spk.extend_from_slice(&if r != 16000 {
                    crate::audio::resample(&x, r, 16000)
                } else {
                    x.clone()
                });
            }
            at_44k.extend_from_slice(&if r != 44100 {
                crate::audio::resample(&x, r, 44100)
            } else {
                x
            });
        }
        if a.use_spk_emb && spk_row.is_none() {
            let onnx = model_dir.join("campplus.onnx");
            let emb = crate::spk::embed_16k_with(&onnx, &for_spk)?;
            spk_row = Some(pl.project_speaker(&emb));
        }
        if req.prompt_text.is_some() {
            prompt_latent = Some(pl.encode_prompt_latent(&mut *rng, &at_44k)?);
        }
    }
    if prompt_latent.is_none() {
        prompt_text = None;
    }
    let mut latent_rows = prompt_latent
        .as_ref()
        .map(|(lat, n)| pl.project_latent_rows(lat, *n));
    let mut latent_steps = prompt_latent.as_ref().map_or(0, |(_, n)| *n);

    let mut out: Vec<f32> = Vec::new();
    for (index, fragment) in fragments.iter().enumerate() {
        let layout = crate::tok::build_input_ids(
            &tok,
            crate::tok::DENSE,
            &crate::tok::InputSeqArgs {
                prompt: &req.prompt,
                text: fragment,
                num_speakers: spk_row.is_some() as usize,
                instruction: req.instruction.as_deref(),
                prompt_text: if latent_rows.is_some() { prompt_text.as_deref() } else { None },
                prompt_latent_steps: latent_steps,
            },
        )?;
        let mut embeds = pl.embed_ids(&layout.input_ids)?;
        if let Some(se) = &spk_row {
            let pos = layout.speaker_positions[0];
            embeds[pos * HIDDEN..(pos + 1) * HIDDEN].copy_from_slice(&se[..]);
        }
        if let Some(rows) = &latent_rows {
            let start = layout.latent_position as usize;
            embeds[start * HIDDEN..(start + latent_steps) * HIDDEN].copy_from_slice(rows);
        }
        if let Ok(path) = std::env::var("MING_DUMP_EMBEDS") {
            let path = format!("{path}.{index}.f32");
            let mut bytes = Vec::with_capacity(embeds.len() * 4);
            for v in &embeds {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&path, &bytes)?;
            println!("embeds -> {path} ({} x 896)", layout.input_ids.len());
        }
        let t = layout.input_ids.len();
        let t1 = std::time::Instant::now();
        let latents = pl.decode_loop(
            &embeds,
            t,
            &mut *rng,
            a.cfg as f32,
            a.sigma as f32,
            a.temperature as f32,
            a.max_decode_steps,
        )?;
        let n = latents.len() / ZC;
        let d = t1.elapsed();
        let t2 = std::time::Instant::now();
        let mut wav = pl.vocode(&latents);
        let v = t2.elapsed();
        if !a.no_trim_silence {
            wav = crate::pipeline::Pipeline::trim_trailing_silence(&wav, 44100);
        }
        if reference_first_fragment && index == 0 {
            // Same as ming_tts._prepare_generated_prompt: the trimmed chunk
            // becomes the acoustic prompt (and the voice) for every later one.
            let spk16 = crate::audio::resample(&wav, 44100, 16000);
            let onnx = model_dir.join("campplus.onnx");
            let se = crate::spk::embed_16k_with(&onnx, &spk16)?;
            spk_row = Some(pl.project_speaker(&se));
            let (lat, np) = pl.encode_prompt_latent(&mut *rng, &wav)?;
            latent_rows = Some(pl.project_latent_rows(&lat, np));
            latent_steps = np;
            prompt_text = Some(fragment.clone());
            println!(
                "  re-conditioning on fragment 0: {np} prompt patches, {} speaker samples",
                spk16.len()
            );
        }
        println!(
            "  fragment {index}: {n} latents in {:.2}s, vae {:.2}s",
            d.as_secs_f64(),
            v.as_secs_f64()
        );
        out.extend_from_slice(&wav);
    }
    crate::audio::write_wav(&a.output, &out, 44100)?;
    println!(
        "wrote {} ({:.2}s @ 44100)",
        a.output,
        out.len() as f64 / 44100.0
    );
    Ok(())
}


// ---------------------------------------------------------------------------
// VAE encoder (hand-written twin of vae.rs): waveform -> posterior params
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
