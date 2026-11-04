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

/// Five-field block for kernels that walk a token window (the 48x64 tiled
/// GEMM: rows, K, tokens in this tile, kg, token offset).
pub fn uni5(a: u32, b: u32, c: u32, d: u32, e: u32) -> [u8; 32] {
    let mut u = [0u8; 32];
    u[0..4].copy_from_slice(&a.to_le_bytes());
    u[4..8].copy_from_slice(&b.to_le_bytes());
    u[8..12].copy_from_slice(&c.to_le_bytes());
    u[12..16].copy_from_slice(&d.to_le_bytes());
    u[16..20].copy_from_slice(&e.to_le_bytes());
    u
}

/// A bias-ful linear layer with bf16 weights kept packed 2-per-u32.
pub struct Lin {
    pub w: Buffer,
    pub b: Buffer,
    pub rows: usize,
    pub in_features: usize,
    /// Stable id for bind-group caching. Zero means "not cacheable" (test and
    /// one-off layers constructed by hand).
    pub id: u64,
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
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Self { w: wbuf, b: bbuf, rows, in_features, id })
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
/// Arguments of one CFM Euler step (see [`Ops::cfm_step`]).
pub struct CfmStep<'a> {
    /// Conditional prediction buffer; the null prediction may live in the same
    /// buffer at `off_null` (the batched CFG path).
    pub pred: &'a Buffer,
    pub null: &'a Buffer,
    pub sde: &'a Buffer,
    /// Output (updated in place).
    pub y: &'a Buffer,
    pub cfg: f32,
    pub dt: f32,
    pub k: f32,
    /// Offset of the conditional prediction inside `pred`.
    pub off: usize,
    /// Offset of the null prediction inside `null` (or `pred` when shared).
    pub off_null: usize,
    pub n: usize,
}

pub struct Ops {
    pub gpu: Gpu,
    k_rms: crate::nn::Kernel,
    k_linear: crate::nn::Kernel,
    k_linear_t4: crate::nn::Kernel,
    k_linear_t8: crate::nn::Kernel,
    /// Single-token decode: vec4 weight loads, 32 rows per workgroup.
    k_linear_t1: crate::nn::Kernel,
    k_linear_v4t6: crate::nn::Kernel,
    /// Six-token DiT batch: 8 lanes per row, two rows per lane group, two
    /// granules unrolled. One tuning for every adapter -- see PROGRESS 0.16 for
    /// why the warp-per-row tuning (10.7% faster on a discrete card) is not
    /// the default.
    k_linear_t6: crate::nn::Kernel,
    /// Twelve-token window variant of the vectorised kernel: same micro-kernel,
    /// twice the weight reuse. Used for t > 8 (VAE decode, long prompts).
    k_linear_t12win: crate::nn::Kernel,
    k_prefill_tile: crate::nn::Kernel,
    /// Tile allowed for post-hoc (decoder) linears: discrete GPUs only.
    prefill_tiled: bool,
    k_rope: crate::nn::Kernel,
    k_attend: crate::nn::Kernel,
    /// Reduction-free attention for software adapters (see ATTEND_SIMPLE).
    k_attend_simple: crate::nn::Kernel,
    cpu_attend: bool,
    k_kv_write: crate::nn::Kernel,
    k_silu: crate::nn::Kernel,
    k_gelu: crate::nn::Kernel,
    /// Overflow-free GELU for software adapters (see GELU_TANH_STABLE).
    k_gelu_stable: crate::nn::Kernel,
    cpu_ops: bool,
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
            k_linear_t1: kernel(&gpu, &shaders::gemv_t1_vec4(8, 128, 2), 3),
            k_linear_v4t6: kernel(&gpu, shaders::GEMV_V4_T6, 3),
            // 8 lanes per row, 2 rows per lane group, 2 granules unrolled.
            k_linear_t6: kernel(&gpu, &shaders::gemv_t6_vec4(8, 128, 2, 2, false, 6), 3),
            // Same shape as the default, but able to slide its token window
            // across a long prompt (prefill, VAE batch).
            k_linear_t12win: kernel(&gpu, &shaders::gemv_t6_vec4(8, 128, 2, 2, true, 12), 3),
            k_prefill_tile: kernel(&gpu, shaders::GEMM_PREFILL_48X64, 3),
            // Large-t strategy follows the adapter class, because the two
            // kernels are bound by different things: the 48x64 tile is compute
            // bound and needs a discrete GPU's FP32 throughput, while the
            // register-window kernel is bandwidth bound and wins on integrated
            // ones (measured: tile is 2x faster on a P104, 1.6x slower on an
            // Intel UHD). `MING_PREFILL=base` forces the window kernel.
            prefill_tiled: match std::env::var("MING_PREFILL").ok().as_deref() {
                Some("base") => false,
                _ => gpu.device_type == wgpu::DeviceType::DiscreteGpu,
            },
            k_rope: kernel(&gpu, shaders::ROPE, 2),
            k_attend: kernel(&gpu, shaders::ATTEND, 3),
            k_attend_simple: kernel(&gpu, shaders::ATTEND_SIMPLE, 3),
            cpu_attend: gpu.device_type == wgpu::DeviceType::Cpu,
            k_kv_write: kernel(&gpu, shaders::KV_WRITE, 1),
            k_silu: kernel(&gpu, shaders::SILU_MUL, 2),
            k_gelu: kernel(&gpu, shaders::GELU_TANH, 1),
            k_gelu_stable: kernel(&gpu, shaders::GELU_TANH_STABLE, 1),
            cpu_ops: gpu.device_type == wgpu::DeviceType::Cpu,
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

    /// Which large-t (VAE decode/encode, long prefill) kernel the adapter got:
    /// the 48x64 tile on discrete GPUs, the register window elsewhere.
    pub fn large_t_tile(&self) -> bool {
        self.prefill_tiled
    }

    /// y = W x + b for `t` rows. Batches of up to 4 use the weight-reuse
    /// kernel; the DiT only ever runs 1-3 rows, which is where it matters.
    pub fn linear(&self, lin: &Lin, x: &Buffer, t: usize) -> Result<Buffer> {
        let y = self.gpu.storage((t * lin.rows * 4) as u64, "linear.y");
        self.linear_into(lin, x, &y, t);
        Ok(y)
    }

    /// Six-token batch (the DiT). The vec4 kernel reads 16 B weight granules,
    /// so it needs whole 256-wide K; anything else falls back to
    /// [`Self::k_linear_v4t6`].
    fn t6_plan(&self, lin: &Lin) -> Option<(&crate::nn::Kernel, usize)> {
        if !lin.in_features.is_multiple_of(256) {
            return None;
        }
        Some((&self.k_linear_t6, 32))
    }

    /// Same dispatch as [`Self::linear_into`], but for callers that keep the
    /// activation and output buffers alive for the whole model. `x_slot` and
    /// `y_slot` must identify those buffers; the layer's own id disambiguates
    /// the weights.
    pub fn linear_into_cached(
        &self,
        lin: &Lin,
        x: &Buffer,
        y: &Buffer,
        t: usize,
        x_slot: u64,
        y_slot: u64,
    ) {
        if t == 6 {
            if let Some((k, per_wg)) = self.t6_plan(lin) {
                let key = lin.id.rotate_left(24) ^ x_slot.rotate_left(12) ^ y_slot;
                k.run_cached(
                    &self.gpu,
                    key,
                    &uni4(lin.rows as u32, (lin.in_features / 2) as u32, 6, 0),
                    &[&lin.w, x, &lin.b],
                    y,
                    lin.rows.div_ceil(per_wg) as u32,
                    1,
                );
                return;
            }
        }
        if t == 1 && lin.in_features.is_multiple_of(8) && lin.rows.is_multiple_of(8) {
            let key = lin.id.rotate_left(24) ^ x_slot.rotate_left(12) ^ y_slot;
            self.k_linear_t1.run_cached(
                &self.gpu,
                key,
                &uni2(lin.rows as u32, (lin.in_features / 8) as u32),
                &[&lin.w, x, &lin.b],
                y,
                lin.rows.div_ceil(32) as u32,
                1,
            );
            return;
        }
        self.linear_into(lin, x, y, t);
    }

    /// Decoder-side linear. The 48x64 tile is allowed here because this output
    /// only becomes the waveform: it never feeds the autoregressive loop, so a
    /// different reduction order cannot change the token trajectory. Prefill and
    /// the VAE encoder deliberately stay on the window kernel (`linear_into`) --
    /// routing the tile through those shifted the stop step on long single
    /// segments (S3/S10) and would break the "same duration" acceptance bar.
    pub fn linear_big(&self, lin: &Lin, x: &Buffer, t: usize) -> Result<Buffer> {
        if t > 8
            && self.prefill_tiled
            && lin.rows.is_multiple_of(64)
            && lin.in_features.is_multiple_of(32)
        {
            let y = self.gpu.storage((t * lin.rows * 4) as u64, "linear.y");
            self.tile_chunks(lin, x, &y, t);
            return Ok(y);
        }
        self.linear(lin, x, t)
    }

    /// Walk `t` tokens through the 64-row x 48-token tile, one window at a time.
    fn tile_chunks(&self, lin: &Lin, x: &Buffer, y: &Buffer, t: usize) {
        let mut tok0 = 0usize;
        while tok0 < t {
            let n = (t - tok0).min(48);
            self.k_prefill_tile.run(
                &self.gpu,
                &uni5(
                    lin.rows as u32,
                    lin.in_features as u32,
                    // n_tok is the GLOBAL end of this window: the kernel compares
                    // `tok0 + local` against it, so passing the window length
                    // alone would zero every chunk but the first.
                    (tok0 + n) as u32,
                    (lin.in_features / 8) as u32,
                    tok0 as u32,
                ),
                &[&lin.w, x, &lin.b],
                y,
                (lin.rows / 64) as u32,
                n.div_ceil(48) as u32,
            );
            tok0 += n;
        }
    }

    pub fn linear_into(&self, lin: &Lin, x: &Buffer, y: &Buffer, t: usize) {
        // t > 8 (VAE decode/encode, long prefill): walk the batch in windows of
        // 12 tokens instead of 6. Same micro-kernel and register blocking, but
        // each weight read feeds twice as many tokens -- this halves the weight
        // traffic of the VAE decoder (t=252: 42 passes -> 21) without the
        // shared-memory transpose the tiled kernel needs.
        if t > 8 && lin.in_features.is_multiple_of(128) {
            let per_wg = 32;
            let mut tok0 = 0usize;
            while tok0 < t {
                let n = (t - tok0).min(12);
                self.k_linear_t12win.run(
                    &self.gpu,
                    &uni4(
                        lin.rows as u32,
                        (lin.in_features / 2) as u32,
                        n as u32,
                        tok0 as u32,
                    ),
                    &[&lin.w, x, &lin.b],
                    y,
                    lin.rows.div_ceil(per_wg) as u32,
                    1,
                );
                tok0 += n;
            }
            return;
        }
        // Long prompts / VAE batch are handled by the tiled path above; odd
        // shapes fall through to the generic kernel below.
        if t == 6 {
            if let Some((k, per_wg)) = self.t6_plan(lin) {
                k.run(
                    &self.gpu,
                    &uni4(lin.rows as u32, (lin.in_features / 2) as u32, 6, 0),
                    &[&lin.w, x, &lin.b],
                    y,
                    lin.rows.div_ceil(per_wg) as u32,
                    1,
                );
                return;
            }
        }
        // Six-token DiT batch: vec4 weight loads, six accumulator groups.
        if t == 6 && lin.in_features.is_multiple_of(8) && lin.rows.is_multiple_of(8) {
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
        if t == 1 && lin.in_features.is_multiple_of(8) && lin.rows.is_multiple_of(8) {
            self.k_linear_t1.run(
                &self.gpu,
                &uni2(lin.rows as u32, (lin.in_features / 8) as u32),
                &[&lin.w, x, &lin.b],
                y,
                lin.rows.div_ceil(32) as u32,
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
                lin.rows.div_ceil(8) as u32,
                1,
            );
        } else {
            self.k_linear.run(
                &self.gpu,
                &uni2(lin.rows as u32, (lin.in_features / 2) as u32),
                &[&lin.w, x, &lin.b],
                y,
                lin.rows.div_ceil(8) as u32,
                t as u32,
            );
        }
    }

    #[allow(dead_code)]
    pub fn rms_into(&self, x: &Buffer, w: &Buffer, y: &Buffer, rows: usize, dim: usize) {
        self.k_rms.run(
            &self.gpu,
            &uni2(dim as u32, RMS_EPS.to_bits()),
            &[x, w],
            y,
            rows as u32,
            1,
        );
    }

    pub fn rms_into_cached(
        &self,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        rows: usize,
        dim: usize,
        key: u64,
    ) {
        self.k_rms.run_cached(
            &self.gpu,
            key,
            &uni2(dim as u32, RMS_EPS.to_bits()),
            &[x, w],
            y,
            rows as u32,
            1,
        );
    }

    pub fn add(&self, a: &Buffer, b: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "add.y");
        self.add_into(a, b, &y, n);
        y
    }

    pub fn add_into(&self, a: &Buffer, b: &Buffer, y: &Buffer, n: usize) {
        self.k_add
            .run(&self.gpu, &uni1(n as u32), &[a, b], y, n.div_ceil(256) as u32, 1);
    }

    pub fn add_into_cached(&self, a: &Buffer, b: &Buffer, y: &Buffer, n: usize, key: u64) {
        self.k_add.run_cached(
            &self.gpu,
            key,
            &uni1(n as u32),
            &[a, b],
            y,
            n.div_ceil(256) as u32,
            1,
        );
    }

    pub fn silu_mul(&self, g: &Buffer, u: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "silu.y");
        self.k_silu.run(&self.gpu, &uni1(n as u32), &[g, u], &y, n.div_ceil(256) as u32, 1);
        y
    }

    pub fn silu_mul_into_cached(
        &self,
        g: &Buffer,
        u: &Buffer,
        y: &Buffer,
        n: usize,
        key: u64,
    ) {
        self.k_silu.run_cached(
            &self.gpu,
            key,
            &uni1(n as u32),
            &[g, u],
            y,
            n.div_ceil(256) as u32,
            1,
        );
    }

    pub fn gelu(&self, x: &Buffer, n: usize) -> Buffer {
        let y = self.gpu.storage((n * 4) as u64, "gelu.y");
        self.gelu_into(x, &y, n);
        y
    }

    pub fn gelu_into(&self, x: &Buffer, y: &Buffer, n: usize) {
        let k = if self.cpu_ops { &self.k_gelu_stable } else { &self.k_gelu };
        k.run(&self.gpu, &uni1(n as u32), &[x], y, n.div_ceil(256) as u32, 1);
    }

    pub fn gelu_into_cached(&self, x: &Buffer, y: &Buffer, n: usize, key: u64) {
        self.k_gelu.run_cached(
            &self.gpu,
            key,
            &uni1(n as u32),
            &[x],
            y,
            n.div_ceil(256) as u32,
            1,
        );
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

    #[allow(clippy::too_many_arguments)]
    pub fn rope_cached(
        &self,
        key: u64,
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
        self.k_rope.run_cached(
            &self.gpu,
            key,
            &u,
            &[&tables.cos, &tables.sin],
            buf,
            tokens as u32,
            1,
        );
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

    #[allow(clippy::too_many_arguments)] // orthogonal shape/cache arguments
    pub fn kv_write_cached(
        &self,
        key: u64,
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
        self.k_kv_write.run_cached(
            &self.gpu,
            key,
            &u,
            &[src],
            cache,
            tokens as u32,
            1,
        );
    }

    fn attend_uniform(p: &AttendParams) -> [u8; 48] {
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
        u
    }

    /// Software adapters get the reduction-free variant (see ATTEND_SIMPLE).
    fn attend_kernel(&self) -> &crate::nn::Kernel {
        if self.cpu_attend {
            &self.k_attend_simple
        } else {
            &self.k_attend
        }
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
        self.attend_into(p, q, kc, vc, &y, n_q_heads);
        y
    }

    pub fn attend_into(
        &self,
        p: &AttendParams,
        q: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        y: &Buffer,
        n_q_heads: usize,
    ) {
        let u = Self::attend_uniform(p);
        self.attend_kernel()
            .run(&self.gpu, &u, &[q, kc, vc], y, n_q_heads as u32, p.t_q as u32);
    }

    #[allow(clippy::too_many_arguments)] // orthogonal shape/cache arguments
    pub fn attend_into_cached(
        &self,
        p: &AttendParams,
        q: &Buffer,
        kc: &Buffer,
        vc: &Buffer,
        y: &Buffer,
        n_q_heads: usize,
        key: u64,
    ) {
        let u = Self::attend_uniform(p);
        self.attend_kernel().run_cached(
            &self.gpu,
            key,
            &u,
            &[q, kc, vc],
            y,
            n_q_heads as u32,
            p.t_q as u32,
        );
    }

    /// Assemble [row0, row1, row2] of width `dim` into one buffer.
    #[allow(dead_code)]
    pub fn pack3(&self, a: &Buffer, b: &Buffer, c: &Buffer, dim: usize, rows: usize) -> Buffer {
        let y = self.gpu.storage((rows * dim * 4) as u64, "pack.y");
        self.pack3_into(a, b, c, &y, dim, rows);
        y
    }

    pub fn pack3_into(
        &self,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        y: &Buffer,
        dim: usize,
        rows: usize,
    ) {
        self.k_pack3.run(
            &self.gpu,
            &uni1(dim as u32),
            &[a, b, c],
            y,
            (rows * dim).div_ceil(256) as u32,
            1,
        );
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)] // orthogonal shape/cache arguments
    pub fn pack3_into_cached(
        &self,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        y: &Buffer,
        dim: usize,
        rows: usize,
        key: u64,
    ) {
        self.k_pack3.run_cached(
            &self.gpu,
            key,
            &uni1(dim as u32),
            &[a, b, c],
            y,
            (rows * dim).div_ceil(256) as u32,
            1,
        );
    }

    /// Assemble [y_cond, x_hist, x_now, y_null, x_hist, x_now] of width `dim`.
    #[allow(dead_code)]
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
        self.pack6_into(y_cond, x_hist, x_now, y_null, &y, dim);
        y
    }

    pub fn pack6_into(
        &self,
        y_cond: &Buffer,
        x_hist: &Buffer,
        x_now: &Buffer,
        y_null: &Buffer,
        y: &Buffer,
        dim: usize,
    ) {
        let rows = 6;
        self.k_pack6.run(
            &self.gpu,
            &uni1(dim as u32),
            &[y_cond, x_hist, x_now, y_null],
            y,
            (rows * dim).div_ceil(256) as u32,
            1,
        );
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)] // orthogonal shape/cache arguments
    pub fn pack6_into_cached(
        &self,
        y_cond: &Buffer,
        x_hist: &Buffer,
        x_now: &Buffer,
        y_null: &Buffer,
        y: &Buffer,
        dim: usize,
        key: u64,
    ) {
        let rows = 6;
        self.k_pack6.run_cached(
            &self.gpu,
            key,
            &uni1(dim as u32),
            &[y_cond, x_hist, x_now, y_null],
            y,
            (rows * dim).div_ceil(256) as u32,
            1,
        );
    }

    /// One CFM Euler step, on device: y += (pred + (pred-null)*cfg)*dt + k*sde.
    /// `off` / `off_null` locate the two predictions inside their buffers (the
    /// batched path keeps both in one).
    pub fn cfm_step(&self, p: &CfmStep<'_>) {
        let mut u = [0u8; 32];
        u[0..4].copy_from_slice(&(p.n as u32).to_le_bytes());
        u[4..8].copy_from_slice(&p.dt.to_le_bytes());
        u[8..12].copy_from_slice(&p.cfg.to_le_bytes());
        u[12..16].copy_from_slice(&p.k.to_le_bytes());
        u[16..20].copy_from_slice(&(p.off as u32).to_le_bytes());
        u[20..24].copy_from_slice(&(p.off_null as u32).to_le_bytes());
        self.k_cfm.run(
            &self.gpu,
            &u,
            &[p.pred, p.null, p.sde],
            p.y,
            p.n.div_ceil(256) as u32,
            1,
        );
    }
}