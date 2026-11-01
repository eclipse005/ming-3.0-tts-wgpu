//! Device-level primitives shared by every model stage.
//!
//! Before this module existed, `qwen_lm`, `dit`, `vae` and `vae_dec` each
//! carried their own copy of `Lin`, `uni1/uni2`, `rms`, `mrow`, `add`,
//! `silu_mul` and their own kernel handles. Four copies of anything that
//! touches the hot path is exactly what makes an RTF pass expensive, so the
//! plumbing lives here now and the model modules only keep their own
//! architecture (residual stacks, samplers, front ends).

use crate::gpu::Gpu;
use crate::nn::kernel;
use crate::shaders;
use crate::weights::{Weights};
use anyhow::{bail, Result};
use wgpu::Buffer;

/// rms_norm epsilon used by both Qwen2 stacks and the DiT blocks.
pub const RMS_EPS: f32 = 1e-6;

pub fn uni1(a: u32) -> [u8; 16] {
    uni2(a, 0)
}

pub fn uni2(a: u32, b: u32) -> [u8; 16] {
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&a.to_le_bytes());
    u[4..8].copy_from_slice(&b.to_le_bytes());
    u
}

pub fn uni4(a: u32, b: u32, c: u32, d: u32) -> [u8; 16] {
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&a.to_le_bytes());
    u[4..8].copy_from_slice(&b.to_le_bytes());
    u[8..12].copy_from_slice(&c.to_le_bytes());
    u[12..16].copy_from_slice(&d.to_le_bytes());
    u
}

/// A bias-ful linear layer with bf16 weights kept packed 2-per-u32.
pub struct Lin {
    pub w: Buffer,
    pub b: Buffer,
    pub rows: usize,
    pub in_features: usize,
}

impl Lin {
    /// Load `<base>.weight` (bf16, packed) and `<base>.bias` (zeros when
    /// absent -- the audio encoder's fc1 legitimately has no bias).
    pub fn load(gpu: &Gpu, w: &Weights, base: &str, rows: usize, in_features: usize) -> Result<Self> {
        let wt = w.get(&format!("{base}.weight"))?;
        if wt.shape.iter().product::<usize>() != rows * in_features {
            bail!(
                "{base}.weight has shape {:?}, expected [{rows}, {in_features}]",
                wt.shape
            );
        }
        let packed = wt.to_bf16_packed_u32()?;
        let wbuf = gpu.upload_u32(&packed, "lin.w");
        let bbuf = match w.get(&format!("{base}.bias")) {
            Ok(bt) => gpu.upload_f32(&bt.to_f32()?, "lin.b"),
            Err(_) => gpu.upload_f32(&vec![0.0; rows], "lin.b0"),
        };
        Ok(Self { w: wbuf, b: bbuf, rows, in_features })
    }
}

/// One pre-norm transformer layer of a Qwen2-family stack.
pub struct QwenLayer {
    pub q: Lin,
    pub k: Lin,
    pub v: Lin,
    pub o: Lin,
    pub gate: Lin,
    pub up: Lin,
    pub down: Lin,
    pub n1: Buffer,
    pub n2: Buffer,
}

/// Rotary tables in the layout the ROPE kernel expects: [cap, dim/4] angles.
pub struct RopeTables {
    pub cos: Buffer,
    pub sin: Buffer,
}

impl RopeTables {
    /// Tables from the usual `theta ** (-2i / head_dim)` schedule.
    ///
    /// Only correct when the reference really uses that formula. The DiT does
    /// NOT: its checkpoint carries the inv_freq buffer the model was trained
    /// with, and that buffer is not `1e4 ** (-2i/64)` (it differs by ~1.4e-4
    /// relative, which is 450x the f32 noise on q after one rope). Anything
    /// whose module state_dict has an `inv_freq` tensor must go through
    /// [`RopeTables::from_inv_freq`] instead of guessing a base.
    pub fn new(gpu: &Gpu, theta: f64, head_dim: usize, cap: usize) -> Self {
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| theta.powf((2 * i) as f64 / head_dim as f64).recip() as f32)
            .collect();
        Self::from_inv_freq(gpu, &inv_freq, cap)
    }

    /// Tables straight from a checkpoint `inv_freq` buffer (`theta ** (-2i/d)`
    /// for a plain schedule, but stored per-model, in the model's own dtype).
    /// Angles are accumulated in f64 and only the cos/sin pairs are rounded to
    /// f32, which is what the reference does before its rope multiply.
    pub fn from_inv_freq(gpu: &Gpu, inv_freq: &[f32], cap: usize) -> Self {
        let half = inv_freq.len();
        let (mut cos, mut sin) = (vec![0f32; cap * half], vec![0f32; cap * half]);
        for pos in 0..cap {
            for (i, &f) in inv_freq.iter().enumerate() {
                let ang = pos as f64 * f as f64;
                cos[pos * half + i] = ang.cos() as f32;
                sin[pos * half + i] = ang.sin() as f32;
            }
        }
        Self {
            cos: gpu.upload_f32(&cos, "rope.cos"),
            sin: gpu.upload_f32(&sin, "rope.sin"),
        }
    }
}

/// K/V scratch plus the matching rope table for a stack pass over `cap`
/// positions.
///
/// The LLM can live with the checkpoint's 512-position decode window, but the
/// VAE encoder/decoder see the whole waveform in one call, so their length is
/// unbounded: [`KvCache::ensure`] grows the buffers when a longer utterance
/// arrives. Sizing those caches from the LLM's window silently overflowed both
/// the K/V buffers and the rope table for anything longer than 6.4 s of audio.
pub struct KvCache {
    pub k: Buffer,
    pub v: Buffer,
    pub rope: RopeTables,
    pub cap: usize,
}

impl KvCache {
    pub fn new(gpu: &Gpu, n_kv: usize, head_dim: usize, theta: f64, cap: usize) -> Self {
        let bytes = (n_kv * cap * head_dim * 4) as u64;
        Self {
            k: gpu.storage(bytes, "kv.k"),
            v: gpu.storage(bytes, "kv.v"),
            rope: RopeTables::new(gpu, theta, head_dim, cap),
            cap,
        }
    }

    /// Grow (never shrink) so `t` positions fit; rebuilds the rope table too.
    /// `floor` keeps the first allocation at the historical size so short
    /// utterances do not allocate anything extra.
    pub fn ensure(
        &mut self,
        gpu: &Gpu,
        n_kv: usize,
        head_dim: usize,
        theta: f64,
        t: usize,
        floor: usize,
    ) {
        if t <= self.cap {
            return;
        }
        *self = KvCache::new(gpu, n_kv, head_dim, theta, t.next_power_of_two().max(floor));
    }
}

/// Attention kernel arguments (mirrors the ATTEND Params struct).
#[derive(Clone, Copy)]
pub struct AttendParams {
    pub t_q: usize,
    pub q_pos_base: usize,
    pub heads_per_kv: usize,
    pub scale: f32,
    pub full: bool,
    pub qpos_max: usize,
    pub stride: usize,
    /// 0 = plain causal from position 0; n = keep the last n positions.
    pub window: usize,
    /// KV cache capacity per head, in tokens.
    pub cap: usize,
    /// 0 = one sequence. n = the t_q rows are n independent n-token sequences
    /// packed back to back, each attending only its own slice. This is how the
    /// (cond, null) branches share one DiT pass.
    pub group: usize,
}

impl AttendParams {
    /// Causal, no sliding window (the LLM's attention).
    pub fn causal(
        t_q: usize,
        q_pos_base: usize,
        heads_per_kv: usize,
        head_dim: usize,
        stride: usize,
        cap: usize,
    ) -> Self {
        Self {
            t_q,
            q_pos_base,
            heads_per_kv,
            scale: 1.0 / (head_dim as f32).sqrt(),
            full: false,
            qpos_max: 0,
            stride,
            window: 0,
            cap,
            group: 0,
        }
    }

    /// Bidirectional over the whole sequence (the DiT's attention: no GQA,
    /// every query head owns its own k/v head).
    pub fn full(t_q: usize, head_dim: usize, stride: usize, cap: usize) -> Self {
        Self {
            t_q,
            q_pos_base: 0,
            heads_per_kv: 1,
            scale: 1.0 / (head_dim as f32).sqrt(),
            full: true,
            qpos_max: t_q.saturating_sub(1),
            stride,
            window: 0,
            cap,
            group: 0,
        }
    }

    /// Rows are `n` independent `n`-token sequences (CFG batching).
    pub fn with_group(mut self, n: usize) -> Self {
        self.group = n;
        self
    }

    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }
}

/// All compute kernels, compiled once and shared by every stage on one device.
pub struct Ops {
    pub gpu: Gpu,
    k_rms: crate::nn::Kernel,
    k_linear: crate::nn::Kernel,
    k_linear_t4: crate::nn::Kernel,
    k_linear_t8: crate::nn::Kernel,
    k_linear_v4: crate::nn::Kernel,
    k_linear_v4t6: crate::nn::Kernel,
    k_linear_t8u4: crate::nn::Kernel,
    k_rope: crate::nn::Kernel,
    k_attend: crate::nn::Kernel,
    k_kv_write: crate::nn::Kernel,
    k_silu: crate::nn::Kernel,
    k_gelu: crate::nn::Kernel,
    k_add: crate::nn::Kernel,
    k_pack3: crate::nn::Kernel,
    k_pack6: crate::nn::Kernel,
    k_cfm: crate::nn::Kernel,
}

impl Ops {
    pub fn new(gpu: Gpu) -> Self {
        Self {
            k_rms: kernel(&gpu, shaders::RMSNORM, 2),
            k_linear: kernel(&gpu, shaders::GEMV_MROW, 3),
            k_linear_t4: kernel(&gpu, shaders::GEMV_MROW_T4, 3),
            k_linear_t8: kernel(&gpu, shaders::GEMV_MROW_T8, 3),
            k_linear_v4: kernel(&gpu, shaders::GEMV_V4_T1, 3),
            k_linear_v4t6: kernel(&gpu, shaders::GEMV_V4_T6, 3),
            k_linear_t8u4: kernel(&gpu, shaders::GEMV_MROW_T8U4, 3),
            k_rope: kernel(&gpu, shaders::ROPE, 2),
            k_attend: kernel(&gpu, shaders::ATTEND, 3),
            k_kv_write: kernel(&gpu, shaders::KV_WRITE, 1),
            k_silu: kernel(&gpu, shaders::SILU_MUL, 2),
            k_gelu: kernel(&gpu, shaders::GELU_TANH, 1),
            k_add: kernel(&gpu, shaders::ADD, 2),
            k_pack3: kernel(&gpu, shaders::PACK3, 3),
            k_pack6: kernel(&gpu, shaders::PACK6, 4),
            k_cfm: kernel(&gpu, shaders::CFM_STEP, 3),
            gpu,
        }
    }

    /// y = rmsnorm(x) * w over `rows` rows of width `dim`.
    pub fn rms(&self, x: &Buffer, w: &Buffer, rows: usize, dim: usize) -> Buffer {
        let y = self.gpu.storage((rows * dim * 4) as u64, "rms.y");
        self.k_rms.run(
            &self.gpu,
            &uni2(dim as u32, RMS_EPS.to_bits()),
            &[x, w],
            &y,
            rows as u32,
            1,
        );
        y
    }

    /// y = W x + b for `t` rows. Batches of up to 4 use the weight-reuse
    /// kernel; the DiT only ever runs 1-3 rows, which is where it matters.
    pub fn linear(&self, lin: &Lin, x: &Buffer, t: usize) -> Result<Buffer> {
        let y = self.gpu.storage((t * lin.rows * 4) as u64, "linear.y");
        self.linear_into(lin, x, &y, t);
        Ok(y)
    }

    pub fn linear_into(&self, lin: &Lin, x: &Buffer, y: &Buffer, t: usize) {
        if t == 6 && lin.in_features % 256 == 0 {
            self.k_linear_t8u4.run(
                &self.gpu,
                &uni4(lin.rows as u32, (lin.in_features / 2) as u32, 6, 0),
                &[&lin.w, x, &lin.b],
                y,
                ((lin.rows + 7) / 8) as u32,
                1,
            );
            return;
        }
        // Six-token DiT batch: vec4 weight loads, six accumulator groups.
        if t == 6 && lin.in_features % 8 == 0 && lin.rows % 8 == 0 {
            self.k_linear_v4t6.run(
                &self.gpu,
                &uni2(lin.rows as u32, (lin.in_features / 8) as u32),
                &[&lin.w, x, &lin.b],
                y,
                (lin.rows / 8) as u32,
                1,
            );
            return;
        }
        // Single-token decode: vec4 (16 B) weight loads instead of one u32.
        if t == 1 && lin.in_features % 8 == 0 && lin.rows % 8 == 0 {
            self.k_linear_v4.run(
                &self.gpu,
                &uni2(lin.rows as u32, (lin.in_features / 8) as u32),
                &[&lin.w, x, &lin.b],
                y,
                (lin.rows / 8) as u32,
                1,
            );
            return;
        }
        let kt = if t <= 4 {
            Some(&self.k_linear_t4)
        } else if t <= 8 {
            Some(&self.k_linear_t8)
        } else {
            None
        };
        if let Some(k) = kt {
            k.run(
                &self.gpu,
                &uni4(
                    lin.rows as u32,
                    (lin.in_features / 2) as u32,
                    t as u32,
                    0,
                ),
                &[&lin.w, x, &lin.b],
                y,
                ((lin.rows + 7) / 8) as u32,
                1,
            );
        } else {
            self.k_linear.run(
                &self.gpu,
                &uni2(lin.rows as u32, (lin.in_features / 2) as u32),
                &[&lin.w, x, &lin.b],
                y,
                ((lin.rows + 7) / 8) as u32,
                t as u32,
            );
        }
    }

    pub fn add(&self, a: &Buffer, b: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "add.y");
        self.k_add.run(&self.gpu, &uni1(n as u32), &[a, b], &y, ((n + 255) / 256) as u32, 1);
        y
    }

    pub fn silu_mul(&self, g: &Buffer, u: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "silu.y");
        self.k_silu.run(&self.gpu, &uni1(n as u32), &[g, u], &y, ((n + 255) / 256) as u32, 1);
        y
    }

    pub fn gelu(&self, x: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "gelu.y");
        self.k_gelu.run(&self.gpu, &uni1(n as u32), &[x], &y, ((n + 255) / 256) as u32, 1);
        y
    }

    /// Rotary embedding, in place. interleaved: 0 = Qwen2 rotate-half,
    /// 1 = x_transformers GPT-J style.
    pub fn rope(
        &self,
        tables: &RopeTables,
        buf: &Buffer,
        n_heads: usize,
        interleaved: u32,
        pos_base: usize,
        tokens: usize,
    ) {
        self.rope_rows(tables, buf, n_heads, interleaved, pos_base, tokens, 0);
    }

    /// Same rotation, but written to rows `row_base .. row_base + tokens`, with
    /// positions still starting at `pos_base`. Batched branches need this: the
    /// null branch sits at rows 3..6 but must rotate with positions 0..3.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_rows(
        &self,
        tables: &RopeTables,
        buf: &Buffer,
        n_heads: usize,
        interleaved: u32,
        pos_base: usize,
        tokens: usize,
        row_base: usize,
    ) {
        let mut u = [0u8; 16];
        u[0..4].copy_from_slice(&(n_heads as u32).to_le_bytes());
        u[4..8].copy_from_slice(&(pos_base as u32).to_le_bytes());
        u[8..12].copy_from_slice(&interleaved.to_le_bytes());
        u[12..16].copy_from_slice(&(row_base as u32).to_le_bytes());
        self.k_rope.run(&self.gpu, &u, &[&tables.cos, &tables.sin], buf, tokens as u32, 1);
    }

    /// Copy `tokens` rows of an activation into a head-major KV cache.
    pub fn kv_write(
        &self,
        src: &Buffer,
        cache: &Buffer,
        tokens: usize,
        pos_base: usize,
        cap: usize,
        n_kv: usize,
    ) {
        let mut u = [0u8; 16];
        u[0..4].copy_from_slice(&(pos_base as u32).to_le_bytes());
        u[4..8].copy_from_slice(&(cap as u32).to_le_bytes());
        u[8..12].copy_from_slice(&(n_kv as u32).to_le_bytes());
        self.k_kv_write.run(&self.gpu, &u, &[src], cache, tokens as u32, 1);
    }

    pub fn attend(
        &self,
        p: &AttendParams,
        q: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        n_q_heads: usize,
    ) -> Buffer {
        let y = self.gpu.storage((p.t_q * p.stride * 4) as u64, "attn.y");
        let mut u = [0u8; 48];
        u[0..4].copy_from_slice(&(p.t_q as u32).to_le_bytes());
        u[4..8].copy_from_slice(&(p.q_pos_base as u32).to_le_bytes());
        u[8..12].copy_from_slice(&(p.heads_per_kv as u32).to_le_bytes());
        u[12..16].copy_from_slice(&p.scale.to_le_bytes());
        u[16..20].copy_from_slice(&(p.full as u32).to_le_bytes());
        u[20..24].copy_from_slice(&(p.qpos_max as u32).to_le_bytes());
        u[24..28].copy_from_slice(&(p.stride as u32).to_le_bytes());
        u[28..32].copy_from_slice(&(p.window as u32).to_le_bytes());
        u[32..36].copy_from_slice(&(p.cap as u32).to_le_bytes());
        u[36..40].copy_from_slice(&(p.group as u32).to_le_bytes());
        self.k_attend
            .run(&self.gpu, &u, &[q, kc, vc], &y, n_q_heads as u32, p.t_q as u32);
        y
    }

    /// Assemble [row0, row1, row2] of width `dim` into one buffer.
    pub fn pack3(&self, a: &Buffer, b: &Buffer, c: &Buffer, dim: usize, rows: usize) -> Buffer {
        let y = self.gpu.storage((rows * dim * 4) as u64, "pack.y");
        self.k_pack3.run(
            &self.gpu,
            &uni1(dim as u32),
            &[a, b, c],
            &y,
            ((rows * dim + 255) / 256) as u32,
            1,
        );
        y
    }

    /// Assemble [y_cond, x_hist, x_now, y_null, x_hist, x_now] of width `dim`.
    pub fn pack6(
        &self,
        y_cond: &Buffer,
        x_hist: &Buffer,
        x_now: &Buffer,
        y_null: &Buffer,
        dim: usize,
    ) -> Buffer {
        let rows = 6;
        let y = self.gpu.storage((rows * dim * 4) as u64, "pack6.y");
        self.k_pack6.run(
            &self.gpu,
            &uni1(dim as u32),
            &[y_cond, x_hist, x_now, y_null],
            &y,
            ((rows * dim + 255) / 256) as u32,
            1,
        );
        y
    }

    /// One CFM Euler step, on device: y += (pred + (pred-null)*cfg)*dt + k*sde.
    /// `off` / `off_null` locate the two predictions inside their buffers (the
    /// batched path keeps both in one).
    pub fn cfm_step(
        &self,
        pred: &Buffer,
        null: &Buffer,
        sde: &Buffer,
        y: &Buffer,
        cfg: f32,
        dt: f32,
        k: f32,
        off: usize,
        off_null: usize,
        n: usize,
    ) {
        let mut u = [0u8; 32];
        u[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        u[4..8].copy_from_slice(&dt.to_le_bytes());
        u[8..12].copy_from_slice(&cfg.to_le_bytes());
        u[12..16].copy_from_slice(&k.to_le_bytes());
        u[16..20].copy_from_slice(&(off as u32).to_le_bytes());
        u[20..24].copy_from_slice(&(off_null as u32).to_le_bytes());
        self.k_cfm
            .run(&self.gpu, &u, &[pred, null, sde], y, ((n + 255) / 256) as u32, 1);
    }
}