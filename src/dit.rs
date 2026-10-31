//! DiT flow-matching head + Aggregator projector (1024-wide, 8+8 pre-norm
//! blocks) reusing the M1 kernel family, plus the CFM Euler sampler.

use crate::gpu::Gpu;
use crate::nn::{gemv_bf16w_buf, kernel, Kernel};
use crate::shaders::RMSNORM;
use crate::shaders;
use crate::weights::Weights;
use anyhow::{bail, Context, Result};
use wgpu::Buffer;

pub const XH: usize = 1024; // transformer width
pub const ZC: usize = 64; // latent channels
pub const XHEADS: usize = 16;
pub const XFF: usize = 4096;
pub const STEPS: usize = 10;
const EPS: f32 = 1e-6;

struct Lin {
    w: Buffer,
    b: Buffer,
    rows: usize,
    in_features: usize,
}

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
    cos_buf: Buffer,
    sin_buf: Buffer,
    ones_1024: Buffer,
    k_rms: Kernel,
    k_mrow: Kernel,
    k_rope: Kernel,
    k_attend: Kernel,
    k_silu: Kernel,
    k_add: Kernel,
    k_kvwrite: Kernel,
    k_pack3: Kernel,
    k_gelu: Kernel,
}

impl DitEngine {
    pub fn load(gpu: Gpu, w: &Weights) -> Result<Self> {
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
        // rope base 10000 for the 1024-wide blocks (x_transformers default)
        let (mut cos, mut sin) = (vec![0f32; 512 * 32], vec![0f32; 512 * 32]);
        for pos in 0..512 {
            for i in 0..32 {
                let freq = 10_000f64.powf((2 * i) as f64 / 64.0).recip();
                cos[pos * 32 + i] = (pos as f64 * freq).cos() as f32;
                sin[pos * 32 + i] = (pos as f64 * freq).sin() as f32;
            }
        }
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
            cos_buf: gpu.upload_f32(&cos, "xrope.cos"),
            sin_buf: gpu.upload_f32(&sin, "xrope.sin"),
            ones_1024: gpu.upload_f32(&vec![1.0; XH], "ones"),
            k_rms: kernel(&gpu, RMSNORM, 2),
            k_mrow: kernel(&gpu, shaders::GEMV_MROW, 3),
            k_rope: kernel(&gpu, shaders::ROPE, 2),
            k_attend: kernel(&gpu, shaders::ATTEND, 3),
            k_silu: kernel(&gpu, shaders::SILU_MUL, 2),
            k_add: kernel(&gpu, shaders::ADD, 2),
            k_kvwrite: kernel(&gpu, shaders::KV_WRITE, 1),
            k_pack3: kernel(&gpu, shaders::PACK3, 3),
            k_gelu: kernel(&gpu, shaders::GELU_TANH, 1),
            gpu,
        })
    }

    fn rope_u(&self) -> [u8; 16] {
        let mut u = [0u8; 16];
        u[0..4].copy_from_slice(&(XHEADS as u32).to_le_bytes());
        u[4..8].copy_from_slice(&0u32.to_le_bytes()); // pos_base
        u[8..12].copy_from_slice(&1u32.to_le_bytes()); // interleaved
        u
    }

    fn rms(&self, x: &Buffer, w: &Buffer, rows: usize) -> Buffer {
        let y = self.gpu.storage((rows * XH * 4) as u64, "x.rms");
        self.k_rms.run(
            &self.gpu,
            &uni2(XH as u32, EPS.to_bits()),
            &[x, w],
            &y,
            rows as u32,
            1,
        );
        y
    }

    fn mrow(&self, lin: &Lin, x: &Buffer, t: usize) -> Result<Buffer> {
        let y = self.gpu.storage((t * lin.rows * 4) as u64, "x.mrow");
        self.k_mrow.run(
            &self.gpu,
            &uni2(lin.rows as u32, (lin.in_features / 2) as u32),
            &[&lin.w, x, &lin.b],
            &y,
            ((lin.rows + 7) / 8) as u32,
            t as u32,
        );
        Ok(y)
    }

    fn silu_mul(&self, g: &Buffer, u: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "x.silu");
        self.k_silu.run(&self.gpu, &uni1(n as u32), &[g, u], &y, ((n + 255) / 256) as u32, 1);
        y
    }

    fn add(&self, a: &Buffer, b: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "x.add");
        self.k_add.run(&self.gpu, &uni1(n as u32), &[a, b], &y, ((n + 255) / 256) as u32, 1);
        y
    }

    fn blocks_forward(&self, blocks: &[XBlock], seq: &Buffer, t: usize) -> Result<Buffer> {
        let mut h = seq.clone();
        for b in blocks {
            let a = self.rms(&h, &b.n1, t);
            let q = self.mrow(&b.q, &a, t)?;
            let k = self.mrow(&b.k, &a, t)?;
            let v = self.mrow(&b.v, &a, t)?;
            // rope: 16 heads, positions 0..t, base-10000 tables
            let ru = self.rope_u();
            self.k_rope.run(&self.gpu, &ru, &[&self.cos_buf, &self.sin_buf], &q, t as u32, 1);
            self.k_rope.run(&self.gpu, &ru, &[&self.cos_buf, &self.sin_buf], &k, t as u32, 1);
            // stage k/v into cache layout ([16 * 512 * 64]) and attend (bidirectional)
            let kc = self.gpu.storage((XHEADS * 512 * 64 * 4) as u64, "x.kc");
            let vc = self.gpu.storage((XHEADS * 512 * 64 * 4) as u64, "x.vc");
            let mut u = [0u8; 16];
            u[0..4].copy_from_slice(&0u32.to_le_bytes()); // pos_base
            u[4..8].copy_from_slice(&512u32.to_le_bytes()); // cap
            u[8..12].copy_from_slice(&(XHEADS as u32).to_le_bytes()); // n_kv
            self.k_kvwrite.run(&self.gpu, &u, &[&k], &kc, t as u32, 1);
            self.k_kvwrite.run(&self.gpu, &u, &[&v], &vc, t as u32, 1);
            let mut au = [0u8; 28];
            au[0..4].copy_from_slice(&(t as u32).to_le_bytes());
            au[4..8].copy_from_slice(&0u32.to_le_bytes()); // q_pos_base
            au[8..12].copy_from_slice(&1u32.to_le_bytes()); // heads_per_kv = 1 (no GQA)
            au[12..16].copy_from_slice(&(1.0f32 / 8.0f32).to_le_bytes()); // 1/sqrt(64)
            au[16..20].copy_from_slice(&1u32.to_le_bytes()); // full (bidirectional)
            au[20..24].copy_from_slice(&((t - 1) as u32).to_le_bytes()); // qpos_max
            au[24..28].copy_from_slice(&(XH as u32).to_le_bytes()); // row stride
            let attn = self.gpu.storage((t * XH * 4) as u64, "x.attn");
            self.k_attend.run(&self.gpu, &au, &[&q, &kc, &vc], &attn, XHEADS as u32, t as u32);
            let o = self.mrow(&b.o, &attn, t)?;
            h = self.add(&h, &o, t * XH);
            let a2 = self.rms(&h, &b.n2, t);
            let fi = self.mrow(&b.ff_in, &a2, t)?;
            let g = self.gpu.storage((t * XFF * 4) as u64, "x.gelu");
            self.k_gelu.run(
                &self.gpu,
                &uni1((t * XFF) as u32),
                &[&fi],
                &g,
                ((t * XFF + 255) / 256) as u32,
                1,
            );
            let fo = self.mrow(&b.ff_out, &g, t)?;
            h = self.add(&h, &fo, t * XH);
        }
        Ok(h)
    }

    /// One DiT forward for a single (cond or null) branch.
    /// x: [64] noised latent; c: [896] llm cond; lh: [64] latent history.
    pub fn dit_forward(&self, x: &[f32], t: f32, c: &[f32], lh: &[f32]) -> Result<Vec<f32>> {
        // timestep embedding: sinusoidal(256, scale=1000, half=128,
        // freqs = exp(arange * -log(10000)/127)) then MLP
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
        let tm2 = gemv_bf16w_buf(&self.gpu, &self.t_mlp2.w, &self.t_mlp2.b, &self.gpu.read_f32(&sil, XH)?, XH, XH)?;

        let xn = gemv_bf16w_buf(&self.gpu, &self.x_emb.w, &self.x_emb.b, x, XH, ZC)?;
        let xh = gemv_bf16w_buf(&self.gpu, &self.x_emb.w, &self.x_emb.b, lh, XH, ZC)?;
        let ce = gemv_bf16w_buf(&self.gpu, &self.c_emb.w, &self.c_emb.b, c, XH, 896)?;
        let y = vec_add(&tm2, &ce);

        // assemble seq [3, 1024] = [y, x_hist, x_now]
        let yb = self.gpu.upload_f32(&y, "seq.y");
        let xhb = self.gpu.upload_f32(&xh, "seq.xh");
        let xnb = self.gpu.upload_f32(&xn, "seq.xn");
        let seq = self.gpu.storage((3 * XH * 4) as u64, "seq");
        self.k_pack3.run(&self.gpu, &uni1(XH as u32), &[&yb, &xhb, &xnb], &seq, ((3 * XH + 255) / 256) as u32, 1);

        let h = self.blocks_forward(&self.blocks, &seq, 3)?;
        // final layer: rmsnorm -> linear -> take last row
        let fin = self.rms(&h, &self.fin_norm, 3);
        let fin_host = self.gpu.read_f32_offset(&fin, XH, (2 * XH * 4) as u64)?;
        let out = gemv_bf16w_buf(&self.gpu, &self.fin_lin.w, &self.fin_lin.b, &fin_host, ZC, XH)?;
        Ok(out)
    }

    /// Aggregator: latent [64] -> projected llm cond [896].
    pub fn agg_forward(&self, latent: &[f32]) -> Result<Vec<f32>> {
        let xn_v = gemv_bf16w_buf(&self.gpu, &self.agg_x_emb.w, &self.agg_x_emb.b, latent, XH, ZC)?;
        let xn = self.gpu.upload_f32(&xn_v, "agg.xn");
        // seq [2, 1024] = [cls, x] — PACK3 with row 2 unused
        let seq = self.gpu.storage((2 * XH * 4) as u64, "agg.seq");
        self.k_pack3.run(
            &self.gpu,
            &uni1(XH as u32),
            &[&self.agg_word, &xn, &xn],
            &seq,
            ((2 * XH + 255) / 256) as u32,
            1,
        );
        let h = self.blocks_forward(&self.agg_blocks, &seq, 2)?;
        let fin = self.rms(&h, &self.agg_fin_norm, 2);
        let fin_host = self.gpu.read_f32(&fin, 2 * XH)?; // take row 0 below
        let row0 = &fin_host[0..XH];
        gemv_bf16w_buf(&self.gpu, &self.agg_fin_lin.w, &self.agg_fin_lin.b, row0, 896, XH)
    }

    /// Stage-by-stage bisect variant of dit_forward (parity diagnostics).
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
        let seq = self.gpu.storage((3 * XH * 4) as u64, "seq");
        self.k_pack3.run(&self.gpu, &uni1(XH as u32), &[&yb, &xhb, &xnb], &seq,
                         ((3 * XH + 255) / 256) as u32, 1);
        // block 0 alone
        let b0b = self.blocks_range(&self.blocks[0..1], &seq, 3)?;
        let b0 = self.gpu.read_f32(&b0b, 3 * XH)?;
        let h = self.blocks_range(&self.blocks, &seq, 3)?;
        let fin = self.rms(&h, &self.fin_norm, 3);
        let fin_all = self.gpu.read_f32(&fin, 3 * XH)?;
        let out = gemv_bf16w_buf(&self.gpu, &self.fin_lin.w, &self.fin_lin.b,
                                 &fin_all[2 * XH..3 * XH], ZC, XH)?;
        Ok((temb_mlp, xn, xh, ce, b0, out))
    }

    fn blocks_range(&self, blocks: &[XBlock], seq: &Buffer, t: usize) -> Result<Buffer> {
        let mut h = seq.clone();
        for b in blocks {
            let a = self.rms(&h, &b.n1, t);
            let q = self.mrow(&b.q, &a, t)?;
            let k = self.mrow(&b.k, &a, t)?;
            let v = self.mrow(&b.v, &a, t)?;
            let ru = self.rope_u();
            self.k_rope.run(&self.gpu, &ru, &[&self.cos_buf, &self.sin_buf], &q, t as u32, 1);
            self.k_rope.run(&self.gpu, &ru, &[&self.cos_buf, &self.sin_buf], &k, t as u32, 1);
            let kc = self.gpu.storage((XHEADS * 512 * 64 * 4) as u64, "x.kc");
            let vc = self.gpu.storage((XHEADS * 512 * 64 * 4) as u64, "x.vc");
            let mut u = [0u8; 16];
            u[0..4].copy_from_slice(&0u32.to_le_bytes());
            u[4..8].copy_from_slice(&512u32.to_le_bytes());
            u[8..12].copy_from_slice(&(XHEADS as u32).to_le_bytes());
            self.k_kvwrite.run(&self.gpu, &u, &[&k], &kc, t as u32, 1);
            self.k_kvwrite.run(&self.gpu, &u, &[&v], &vc, t as u32, 1);
            let mut au = [0u8; 28];
            au[0..4].copy_from_slice(&(t as u32).to_le_bytes());
            au[4..8].copy_from_slice(&0u32.to_le_bytes());
            au[8..12].copy_from_slice(&1u32.to_le_bytes());
            au[12..16].copy_from_slice(&(1.0f32 / 8.0f32).to_le_bytes());
            au[16..20].copy_from_slice(&1u32.to_le_bytes());
            au[20..24].copy_from_slice(&((t - 1) as u32).to_le_bytes());
            au[24..28].copy_from_slice(&(XH as u32).to_le_bytes());
            let attn = self.gpu.storage((t * XH * 4) as u64, "x.attn");
            self.k_attend.run(&self.gpu, &au, &[&q, &kc, &vc], &attn, XHEADS as u32, t as u32);
            let o = self.mrow(&b.o, &attn, t)?;
            h = self.add(&h, &o, t * XH);
            let a2 = self.rms(&h, &b.n2, t);
            let fi = self.mrow(&b.ff_in, &a2, t)?;
            let g = self.gpu.storage((t * XFF * 4) as u64, "x.gelu");
            self.k_gelu.run(&self.gpu, &uni1((t * XFF) as u32), &[&fi], &g,
                            ((t * XFF + 255) / 256) as u32, 1);
            let fo = self.mrow(&b.ff_out, &g, t)?;
            h = self.add(&h, &fo, t * XH);
        }
        Ok(h)
    }

    /// CFM Euler integration (epss timesteps + sway), replicating CFM.sample.
    /// With `trace` set, records the y vector fed to each euler fn call.
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
        let mut y = noise.to_vec();
        for step in 0..STEPS {
            let dt = tv[step + 1] - tv[step];
            let pred = self.dit_forward(&y, tv[step], z, lh)?;
            let null = self.dit_forward(&y, tv[step], &vec![0.0; 896], lh)?;
            if let Some((tx, tp, tn)) = trace.as_mut() {
                tx.push(y.clone());
                tp.push(pred.clone());
                tn.push(null.clone());
            }
            let mut vel = vec![0f32; ZC];
            for i in 0..ZC {
                vel[i] = pred[i] + (pred[i] - null[i]) * cfg;
            }
            for i in 0..ZC {
                y[i] += vel[i] * dt;
            }
            if temperature > 0.0 {
                let k = sigma * temperature.sqrt() * dt.abs().sqrt();
                let base = step * ZC;
                for i in 0..ZC {
                    y[i] += k * sde[base + i];
                }
            }
        }
        Ok(y)
    }

    fn _zero_c_host(&self) -> Vec<f32> {
        vec![0.0; 896]
    }
}

fn uni1(a: u32) -> [u8; 16] {
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&a.to_le_bytes());
    u
}

fn uni2(a: u32, b: u32) -> [u8; 16] {
    let mut u = uni1(a);
    u[4..8].copy_from_slice(&b.to_le_bytes());
    u
}

fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

impl DitEngine {
    pub fn flow_sample(
        &self,
        z: &[f32],
        lh: &[f32],
        noise: &[f32],
        sde: &[f32],
        cfg: f32,
    ) -> Result<Vec<f32>> {
        self.flow_sample_traced(z, lh, noise, sde, cfg, None)
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
