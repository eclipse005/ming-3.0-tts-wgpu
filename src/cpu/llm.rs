//! The Qwen2.5-0.5B decoder on the host: prefill plus incremental decode with
//! a per-layer KV cache and the stop head.

use anyhow::Result;

use super::tensor::{add_into, attention, rmsnorm, rope_split_half, silu_mul, Mat};
use crate::qwen_lm::{FF, HEAD_DIM, HIDDEN, KV_CAP, KV_HEADS, LAYERS, Q_HEADS, ROPE_THETA};
use crate::weights::Weights;

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
        for (r, slot) in logits.iter_mut().enumerate() {
            let mut s = self.stop_b[r];
            for (i, &v) in last.iter().enumerate() {
                s += self.stop_w[r * HIDDEN + i] * v;
            }
            *slot = s;
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
