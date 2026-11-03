//! DiT flow-matching head and the audio aggregator (the CPU twin of dit.rs),
//! including the interleaved RoPE and the grouped CFG attention.

use anyhow::Result;

use super::tensor::{
    add_into, attention_full, attention_group, gelu_tanh, rmsnorm, silu_mul,
    timestep_embedding, CpuLin, DIT_ROPE_CAP,
};
use super::ZC;
use crate::qwen_lm::{HEAD_DIM, HIDDEN};
use crate::weights::Weights;

pub const XH: usize = 1024;

pub const XHEADS: usize = 16;

pub const XFF: usize = 4096;

pub const DIT_STEPS: usize = 10;

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
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[allow(clippy::too_many_arguments)] // orthogonal shape/cache arguments
    pub fn sample(&self, z: &[f32], lh: &[f32], noise: &[f32], sde: &[f32], cfg: f32, sigma: f32, temperature: f32) -> Vec<f32> {
        let idx = [0usize, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32];
        let dtv = 1.0f32 / 32.0;
        let mut tv = [0f32; DIT_STEPS + 1];
        for (i, &ix) in idx.iter().enumerate() {
            tv[i] = dtv * ix as f32;
        }
        for v in tv.iter_mut() {
            *v += -((std::f32::consts::PI / 2.0 * *v).cos() - 1.0 + *v);
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
