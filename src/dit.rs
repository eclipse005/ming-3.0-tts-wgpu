//! DiT flow-matching head + Aggregator projector (1024-wide, 8+8 pre-norm
//! blocks) reusing the M1 kernel family, plus the CFM Euler sampler.

use crate::gpu::Gpu;
#[cfg(test)]
use crate::nn::gemv_bf16w_buf;
use crate::ops::{AttendParams, Lin, Ops, RopeTables};
use crate::weights::Weights;
use anyhow::{bail, Result};
use std::sync::Arc;
use wgpu::Buffer;

pub const XH: usize = 1024; // transformer width
pub const ZC: usize = 64; // latent channels
pub const XHEADS: usize = 16;
pub const XFF: usize = 4096;
/// KV cache capacity per head for the DiT. One token per block is all the
/// sampler ever puts there, so this is headroom, not a requirement; it is now
/// a kernel parameter instead of a literal inside the shader.
pub const DIT_KV_CAP: usize = 512;
pub const STEPS: usize = 10;

struct XBlock {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ff_in: Lin,
    ff_out: Lin,
    n1: Buffer,
    n2: Buffer,
}

pub struct DitEngine {
    gpu: Gpu,
    // timestep embedder
    t_mlp1: Lin, // 256 -> 1024
    t_mlp2: Lin, // 1024 -> 1024
    x_emb: Lin,  // 64 -> 1024
    c_emb: Lin,  // 896 -> 1024
    blocks: Vec<XBlock>,
    fin_norm: Buffer,
    fin_lin: Lin, // 1024 -> 64
    // aggregator
    agg_word: Buffer, // [1024] cls embedding
    agg_x_emb: Lin,   // 64 -> 1024
    agg_blocks: Vec<XBlock>,
    agg_fin_norm: Buffer,
    agg_fin_lin: Lin, // 1024 -> 896
    rope: RopeTables,
    ones_1024: Buffer,
    zero_c: Buffer, // 896 zeros: the CFG "null" conditioning branch
    // One KV scratch pair shared by every block and every call. The old code
    // allocated 2 MB per block per forward, which is 640 MB of live buffers
    // per decode step now that dispatches are batched (the GPU cannot recycle
    // a buffer while a recording references it) -- that is what ran the card
    // out of memory. Only one block is ever in flight, so one pair suffices.
    kv_k: Buffer,
    kv_v: Buffer,
    ops: Arc<Ops>,
}

/// Elementwise add of two equal-length slices (diagnostic paths only).
#[cfg(test)]
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

impl DitEngine {
    pub fn load(gpu: Gpu, ops: Arc<Ops>, w: &Weights) -> Result<Self> {
        let p = |name: &str| w.get(name);
        let lin = |base: &str, rows: usize, inf: usize| -> Result<Lin> {
            Lin::load(&gpu, w, base, rows, inf)
        };
        let nrm = |base: &str| -> Result<Buffer> {
            Ok(gpu.upload_f32(&p(&format!("{base}.weight"))?.to_f32()?, "xnorm"))
        };
        let blocks = (0..8)
            .map(|i| {
                let b = format!("flowloss.cfm.model.blocks.{i}");
                Ok(XBlock {
                    q: lin(&format!("{b}.attn.to_q"), XH, XH)?,
                    k: lin(&format!("{b}.attn.to_k"), XH, XH)?,
                    v: lin(&format!("{b}.attn.to_v"), XH, XH)?,
                    o: lin(&format!("{b}.attn.to_out.0"), XH, XH)?,
                    ff_in: lin(&format!("{b}.mlp.ff.0.0"), XFF, XH)?,
                    ff_out: lin(&format!("{b}.mlp.ff.2"), XH, XFF)?,
                    n1: nrm(&format!("{b}.norm1"))?,
                    n2: nrm(&format!("{b}.norm2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let agg_blocks = (0..8)
            .map(|i| {
                let b = format!("linear_proj_audio.blocks.{i}");
                Ok(XBlock {
                    q: lin(&format!("{b}.attn.to_q"), XH, XH)?,
                    k: lin(&format!("{b}.attn.to_k"), XH, XH)?,
                    v: lin(&format!("{b}.attn.to_v"), XH, XH)?,
                    o: lin(&format!("{b}.attn.to_out.0"), XH, XH)?,
                    ff_in: lin(&format!("{b}.mlp.ff.0.0"), XFF, XH)?,
                    ff_out: lin(&format!("{b}.mlp.ff.2"), XH, XFF)?,
                    n1: nrm(&format!("{b}.norm1"))?,
                    n2: nrm(&format!("{b}.norm2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // The 1024-wide blocks rotate with the checkpoint's own inv_freq buffer,
        // not with theta=1e4. x_transformers registers inv_freq as a buffer, so
        // the trained model carries it, and it is NOT `1e4 ** (-2i/64)`: the two
        // differ by ~1.4e-4 relative (the stored one is a bf16-rounded schedule
        // whose ratio to 1e4 drifts per index). Feeding the wrong base is the
        // single largest error source inside a DiT block: q is exact to 1.8e-7
        // before rope and 8.1e-5 after it, which then shows up as 1.1e-4 in
        // attention and 2.3e-5 at the block output. Same buffer serves the
        // aggregator (the checkpoint's two inv_freq tensors are identical).
        let inv_freq = p("flowloss.cfm.model.rotary_embed.inv_freq")?.to_f32()?;
        if inv_freq.len() * 2 != 64 {
            bail!(
                "DiT rotary inv_freq has {} entries, expected 32 (head_dim 64)",
                inv_freq.len()
            );
        }
        let rope = RopeTables::from_inv_freq(&gpu, &inv_freq, 512);
        Ok(Self {
            t_mlp1: lin("flowloss.cfm.model.t_embedder.time_mlp.0", XH, 256)?,
            t_mlp2: lin("flowloss.cfm.model.t_embedder.time_mlp.2", XH, XH)?,
            x_emb: lin("flowloss.cfm.model.x_embedder", XH, ZC)?,
            c_emb: lin("flowloss.cfm.model.c_embedder.cond_embedder", XH, 896)?,
            blocks,
            fin_norm: nrm("flowloss.cfm.model.final_layer.norm_final")?,
            fin_lin: lin("flowloss.cfm.model.final_layer.linear", ZC, XH)?,
            agg_word: gpu.upload_f32(
                &p("linear_proj_audio.word_embedder.weight")?.to_f32()?,
                "agg.cls",
            ),
            agg_x_emb: lin("linear_proj_audio.x_embedder", XH, ZC)?,
            agg_blocks,
            agg_fin_norm: nrm("linear_proj_audio.final_layer.norm_final")?,
            agg_fin_lin: lin("linear_proj_audio.final_layer.linear", 896, XH)?,
            rope,
            ones_1024: gpu.upload_f32(&vec![1.0; XH], "ones"),
            zero_c: gpu.upload_f32(&vec![0.0; 896], "zero_c"),
            kv_k: gpu.storage((XHEADS * 512 * 64 * 4) as u64, "x.kv_k"),
            kv_v: gpu.storage((XHEADS * 512 * 64 * 4) as u64, "x.kv_v"),
            ops,
            gpu,
        })
    }

    fn rms(&self, x: &Buffer, w: &Buffer, rows: usize) -> Buffer {
        self.ops.rms(x, w, rows, XH)
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

    fn gelu(&self, x: &Buffer, n: usize) -> Buffer {
        self.ops.gelu(x, n)
    }

    fn blocks_forward(&self, blocks: &[XBlock], seq: &Buffer, t: usize) -> Result<Buffer> {
        let mut h = seq.clone();
        for b in blocks {
            let a = self.rms(&h, &b.n1, t);
            let q = self.mrow(&b.q, &a, t)?;
            let k = self.mrow(&b.k, &a, t)?;
            let v = self.mrow(&b.v, &a, t)?;
            // rope: 16 heads, positions 0..t, checkpoint inv_freq, interleaved
            self.ops.rope(&self.rope, &q, XHEADS, 1, 0, t);
            self.ops.rope(&self.rope, &k, XHEADS, 1, 0, t);
            // stage k/v into the shared cache layout and attend bidirectionally
            let kc = &self.kv_k;
            let vc = &self.kv_v;
            self.ops.kv_write(&k, kc, t, 0, 512, XHEADS);
            self.ops.kv_write(&v, vc, t, 0, 512, XHEADS);
            let attn = self.ops.attend(
                &AttendParams::full(t, 64, XH, DIT_KV_CAP),
                &q,
                kc,
                vc,
                XHEADS,
            );
            let o = self.mrow(&b.o, &attn, t)?;
            h = self.add(&h, &o, t * XH);
            let a2 = self.rms(&h, &b.n2, t);
            let fi = self.mrow(&b.ff_in, &a2, t)?;
            let g = self.gelu(&fi, t * XFF);
            let fo = self.mrow(&b.ff_out, &g, t)?;
            h = self.add(&h, &fo, t * XH);
        }
        Ok(h)
    }

    /// Sinusoidal timestep embedding: 256 wide, scale 1000, half = 128,
    /// freqs = exp(arange(128) * -log(10000)/127), sin then cos.
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

    /// One DiT forward for a single (cond or null) branch, fully on the device.
    /// x/c/lh are device buffers ([64] / [896] / [64]); the result is [3, 64]
    /// with the prediction in row 2 -- callers read from offset 2 * ZC.
    pub fn dit_forward_dev(
        &self,
        x: &Buffer,
        t: f32,
        c: &Buffer,
        lh: &Buffer,
    ) -> Result<Buffer> {
        let temb = self.gpu.upload_f32(&Self::timestep_embedding(t), "temb");
        let tm1 = self.mrow(&self.t_mlp1, &temb, 1)?;
        let sil = self.silu_mul(&tm1, &self.ones_1024, XH);
        let tm2 = self.mrow(&self.t_mlp2, &sil, 1)?;
        let xn = self.mrow(&self.x_emb, x, 1)?;
        let xh = self.mrow(&self.x_emb, lh, 1)?;
        let ce = self.mrow(&self.c_emb, c, 1)?;
        let y = self.add(&tm2, &ce, XH);

        // assemble seq [3, 1024] = [y, x_hist, x_now]
        let seq = self.ops.pack3(&y, &xh, &xn, XH, 3);
        let h = self.blocks_forward(&self.blocks, &seq, 3)?;
        let fin = self.rms(&h, &self.fin_norm, 3);
        self.mrow(&self.fin_lin, &fin, 3)
    }

    /// Host-facing variant: same maths, one round-trip for the prediction.
    #[cfg(test)]
    pub fn dit_forward(&self, x: &[f32], t: f32, c: &[f32], lh: &[f32]) -> Result<Vec<f32>> {
        let xb = self.gpu.upload_f32(x, "dx");
        let cb = self.gpu.upload_f32(c, "dc");
        let lhb = self.gpu.upload_f32(lh, "dlh");
        let out = self.dit_forward_dev(&xb, t, &cb, &lhb)?;
        self.gpu.read_f32_offset(&out, ZC, (2 * ZC * 4) as u64)
    }

    /// Diagnostic: every sub-operator of block 0, for bisecting where the error
    /// inside one DiT block comes from (see tools/dump_dit_block0.py).
    #[cfg(test)]
    pub fn block0_parts(
        &self,
        x: &[f32],
        t: f32,
        c: &[f32],
        lh: &[f32],
    ) -> Result<Vec<(String, Vec<f32>)>> {
        let xb = self.gpu.upload_f32(x, "b0.x");
        let cb = self.gpu.upload_f32(c, "b0.c");
        let lhb = self.gpu.upload_f32(lh, "b0.lh");
        let temb = self.gpu.upload_f32(&Self::timestep_embedding(t), "b0.temb");
        let tm1 = self.mrow(&self.t_mlp1, &temb, 1)?;
        let sil = self.silu_mul(&tm1, &self.ones_1024, XH);
        let tm2 = self.mrow(&self.t_mlp2, &sil, 1)?;
        let xn = self.mrow(&self.x_emb, &xb, 1)?;
        let xh = self.mrow(&self.x_emb, &lhb, 1)?;
        let ce = self.mrow(&self.c_emb, &cb, 1)?;
        let y = self.add(&tm2, &ce, XH);
        let seq = self.ops.pack3(&y, &xh, &xn, XH, 3);

        let b = &self.blocks[0];
        let mut out: Vec<(String, Vec<f32>)> = Vec::new();
        let a = self.rms(&seq, &b.n1, 3);
        let q = self.mrow(&b.q, &a, 3)?;
        let k = self.mrow(&b.k, &a, 3)?;
        let v = self.mrow(&b.v, &a, 3)?;
        out.push(("norm1".into(), self.gpu.read_f32(&a, 3 * XH)?));
        out.push(("q".into(), self.gpu.read_f32(&q, 3 * XH)?));
        out.push(("k".into(), self.gpu.read_f32(&k, 3 * XH)?));
        out.push(("v".into(), self.gpu.read_f32(&v, 3 * XH)?));
        self.ops.rope(&self.rope, &q, XHEADS, 1, 0, 3);
        self.ops.rope(&self.rope, &k, XHEADS, 1, 0, 3);
        out.push(("q_rope".into(), self.gpu.read_f32(&q, 3 * XH)?));
        out.push(("k_rope".into(), self.gpu.read_f32(&k, 3 * XH)?));
        self.ops.kv_write(&k, &self.kv_k, 3, 0, DIT_KV_CAP, XHEADS);
        self.ops.kv_write(&v, &self.kv_v, 3, 0, DIT_KV_CAP, XHEADS);
        let attn = self.ops.attend(
            &AttendParams::full(3, 64, XH, DIT_KV_CAP),
            &q,
            &self.kv_k,
            &self.kv_v,
            XHEADS,
        );
        let o = self.mrow(&b.o, &attn, 3)?;
        let h = self.add(&seq, &o, 3 * XH);
        out.push(("attn".into(), self.gpu.read_f32(&o, 3 * XH)?));
        let a2 = self.rms(&h, &b.n2, 3);
        out.push(("norm2".into(), self.gpu.read_f32(&a2, 3 * XH)?));
        let fi = self.mrow(&b.ff_in, &a2, 3)?;
        out.push(("ff_pre".into(), self.gpu.read_f32(&fi, 3 * XFF)?));
        let g = self.gelu(&fi, 3 * XFF);
        out.push(("gelu".into(), self.gpu.read_f32(&g, 3 * XFF)?));
        let fo = self.mrow(&b.ff_out, &g, 3)?;
        out.push(("ff_out".into(), self.gpu.read_f32(&fo, 3 * XH)?));
        let bout = self.add(&h, &fo, 3 * XH);
        out.push(("out".into(), self.gpu.read_f32(&bout, 3 * XH)?));
        Ok(out)
    }
    /// Aggregator: latent [64] -> projected llm cond [896].
    pub fn agg_forward(&self, latent: &[f32]) -> Result<Vec<f32>> {
        let xb = self.gpu.upload_f32(latent, "agg.latent");
        let xn = self.mrow(&self.agg_x_emb, &xb, 1)?;
        // seq [2, 1024] = [cls, x] -- only the first two rows are read
        let seq = self.ops.pack3(&self.agg_word, &xn, &xn, XH, 2);
        let h = self.blocks_forward(&self.agg_blocks, &seq, 2)?;
        let fin = self.rms(&h, &self.agg_fin_norm, 2);
        // project both rows on the device, then read back only the cls row
        let out = self.mrow(&self.agg_fin_lin, &fin, 2)?;
        self.gpu.read_f32_offset(&out, 896, 0)
    }

    /// Stage-by-stage bisect variant of dit_forward (parity diagnostics).
    #[cfg(test)]
    pub fn dit_forward_bisect(
        &self,
        x: &[f32],
        t: f32,
        c: &[f32],
        lh: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut temb = vec![0f32; 256];
        for j in 0..128 {
            let e = ((j as f32) * -(10000f32.ln()) / 127.0).exp();
            let v = 1000.0 * t * e;
            temb[j] = v.sin();
            temb[128 + j] = v.cos();
        }
        let tm1 = gemv_bf16w_buf(&self.gpu, &self.t_mlp1.w, &self.t_mlp1.b, &temb, XH, 256)?;
        let tm1b = self.gpu.upload_f32(&tm1, "tm1");
        let sil = self.silu_mul(&tm1b, &self.ones_1024, XH);
        let temb_mlp = gemv_bf16w_buf(&self.gpu, &self.t_mlp2.w, &self.t_mlp2.b,
                                      &self.gpu.read_f32(&sil, XH)?, XH, XH)?;
        let xn = gemv_bf16w_buf(&self.gpu, &self.x_emb.w, &self.x_emb.b, x, XH, ZC)?;
        let xh = gemv_bf16w_buf(&self.gpu, &self.x_emb.w, &self.x_emb.b, lh, XH, ZC)?;
        let ce = gemv_bf16w_buf(&self.gpu, &self.c_emb.w, &self.c_emb.b, c, XH, 896)?;
        let y = vec_add(&temb_mlp, &ce);
        let yb = self.gpu.upload_f32(&y, "seq.y");
        let xhb = self.gpu.upload_f32(&xh, "seq.xh");
        let xnb = self.gpu.upload_f32(&xn, "seq.xn");
        let seq = self.ops.pack3(&yb, &xhb, &xnb, XH, 3);
        // block 0 alone
        let b0b = self.blocks_forward(&self.blocks[0..1], &seq, 3)?;
        let b0 = self.gpu.read_f32(&b0b, 3 * XH)?;
        let h = self.blocks_forward(&self.blocks, &seq, 3)?;
        let fin = self.rms(&h, &self.fin_norm, 3);
        let fin_all = self.gpu.read_f32(&fin, 3 * XH)?;
        let out = gemv_bf16w_buf(&self.gpu, &self.fin_lin.w, &self.fin_lin.b,
                                 &fin_all[2 * XH..3 * XH], ZC, XH)?;
        Ok((temb_mlp, xn, xh, ce, b0, out))
    }

    /// CFM Euler integration (epss timesteps + sway), replicating CFM.sample.
    /// With `trace` set, records the y vector fed to each euler fn call.
    #[cfg(test)]
    pub fn flow_sample_traced(
        &self,
        z: &[f32],
        lh: &[f32],
        noise: &[f32],
        sde: &[f32],
        cfg: f32,
        trace: Option<(&mut Vec<Vec<f32>>, &mut Vec<Vec<f32>>, &mut Vec<Vec<f32>>)>,
    ) -> Result<Vec<f32>> {
        self.flow_sample_full(z, lh, noise, sde, cfg, 0.25, 0.0, trace)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flow_sample_full(
        &self,
        z: &[f32],
        lh: &[f32],
        noise: &[f32],
        sde: &[f32],
        cfg: f32,
        sigma: f32,
        temperature: f32,
        mut trace: Option<(&mut Vec<Vec<f32>>, &mut Vec<Vec<f32>>, &mut Vec<Vec<f32>>)>,
    ) -> Result<Vec<f32>> {
        // epss timesteps for n=10, sway coef -1.0 (f32 math like torch)
        let idx = [0usize, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32];
        let dtv = 1.0f32 / 32.0;
        let mut tv = [0f32; STEPS + 1];
        for (i, &ix) in idx.iter().enumerate() {
            tv[i] = dtv * ix as f32;
        }
        for i in 0..=STEPS {
            tv[i] += -1.0 * ((std::f32::consts::PI / 2.0 * tv[i]).cos() - 1.0 + tv[i]);
        }
        // Everything below stays on the device: the Euler loop used to read
        // the prediction back to the host twice per step, and each readback
        // is a full device sync (~ms under WDDM), which is what made the port
        // two orders of magnitude slower than it should be.
        let zb = self.gpu.upload_f32(z, "cfm.z");
        let lhb = self.gpu.upload_f32(lh, "cfm.lh");
        let y = self.gpu.upload_f32(noise, "cfm.y");
        let sde_zero = self.gpu.upload_f32(&[0f32; ZC], "cfm.sde0");
        let mut out = Vec::with_capacity(ZC);
        for step in 0..STEPS {
            let dt = tv[step + 1] - tv[step];
            let pred = self.dit_forward_dev(&y, tv[step], &zb, &lhb)?;
            let null = self.dit_forward_dev(&y, tv[step], &self.zero_c, &lhb)?;
            if let Some((tx, tp, tn)) = trace.as_mut() {
                let off = (2 * ZC * 4) as u64;
                tx.push(self.gpu.read_f32(&y, ZC)?);
                tp.push(self.gpu.read_f32_offset(&pred, ZC, off)?);
                tn.push(self.gpu.read_f32_offset(&null, ZC, off)?);
            }
            let k = if temperature > 0.0 {
                sigma * temperature.sqrt() * dt.abs().sqrt()
            } else {
                0.0
            };
            let sde_buf = if k != 0.0 {
                let base = step * ZC;
                let end = (base + ZC).min(sde.len());
                self.gpu.upload_f32(&sde[base.min(sde.len())..end], "cfm.sde")
            } else {
                sde_zero.clone()
            };
            self.ops.cfm_step(&pred, &null, &sde_buf, &y, cfg, dt, k, 2 * ZC, ZC);
        }
        out.extend_from_slice(&self.gpu.read_f32(&y, ZC)?);
        Ok(out)
    }

    fn _zero_c_host(&self) -> Vec<f32> {
        vec![0.0; 896]
    }
}

impl DitEngine {
    #[cfg(test)]
    pub fn flow_sample(
        &self,
        z: &[f32],
        lh: &[f32],
        noise: &[f32],
        sde: &[f32],
        cfg: f32,
    ) -> Result<Vec<f32>> {
        self.flow_sample_full(z, lh, noise, sde, cfg, 0.25, 0.0, None)
    }

    /// CFM.sample including the stochastic term:
    ///   y += sigma * sqrt(temperature) * sqrt(|dt|) * sde_noise[step]
    /// The term is identically zero at temperature 0, which is why the M2
    /// goldens were reproducible with no RNG at all.
    #[allow(clippy::too_many_arguments)]
    pub fn flow_sample_stoch(
        &self,
        z: &[f32],
        lh: &[f32],
        noise: &[f32],
        sde: &[f32],
        cfg: f32,
        sigma: f32,
        temperature: f32,
    ) -> Result<Vec<f32>> {
        self.flow_sample_full(z, lh, noise, sde, cfg, sigma, temperature, None)
    }
}