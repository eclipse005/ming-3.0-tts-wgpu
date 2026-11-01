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
pub struct Ops {
    pub gpu: Gpu,
    k_rms: crate::nn::Kernel,
    k_linear: crate::nn::Kernel,
    k_linear_t4: crate::nn::Kernel,
    k_linear_t8: crate::nn::Kernel,
    k_linear_v4: crate::nn::Kernel,
    k_linear_t1hw: crate::nn::Kernel,
    // Single-token vectorised variants: (8,64,1), (8,128,1), (8,128,2), (16,128,1).
    k_t1v: [crate::nn::Kernel; 4],
    t1_variant: Option<usize>,
    t1_halfwarp: bool,
    k_linear_v4t6: crate::nn::Kernel,
    k_linear_t8u4: crate::nn::Kernel,
    k_linear_t8u4_base: crate::nn::Kernel,
    k_linear_t6v2: crate::nn::Kernel,
    k_linear_t6hw: crate::nn::Kernel,
    k_linear_t6qw: crate::nn::Kernel,
    k_linear_t6fl: crate::nn::Kernel,
    k_linear_t6v8: crate::nn::Kernel,
    k_linear_t6v9: crate::nn::Kernel,
    k_linear_t6v10: crate::nn::Kernel,
    k_linear_t6v11: crate::nn::Kernel,
    k_linear_t6v12: crate::nn::Kernel,
    k_linear_t6v13: crate::nn::Kernel,
    k_linear_t6v14: crate::nn::Kernel,
    k_linear_t6win: crate::nn::Kernel,
    t6_vec4: bool,
    t6_specialized: bool,
    t6_paired: bool,
    t6_halfwarp: bool,
    t6_quarterwarp: bool,
    t6_fourlane: bool,
    t6_hybrid: bool,
    k_prefill_tile: crate::nn::Kernel,
    prefill_tiled: bool,
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
            k_linear_t1hw: kernel(&gpu, &shaders::gemv_t1_halfwarp(), 3),
            k_t1v: [
                kernel(&gpu, &shaders::gemv_t1_vec4(8, 64, 1), 3),
                kernel(&gpu, &shaders::gemv_t1_vec4(8, 128, 1), 3),
                kernel(&gpu, &shaders::gemv_t1_vec4(8, 128, 2), 3),
                kernel(&gpu, &shaders::gemv_t1_vec4(16, 128, 1), 3),
            ],
            t1_variant: match std::env::var("MING_T1").unwrap_or_default().as_str() {
                "base" | "half" => None,
                "v1" => Some(0),
                "v2" => Some(1),
                "v3" => Some(2),
                "v4" => Some(3),
                _ => Some(2),
            },
            t1_halfwarp: std::env::var("MING_T1").is_ok_and(|v| v == "half"),
            k_linear_v4t6: kernel(&gpu, shaders::GEMV_V4_T6, 3),
            k_linear_t8u4: kernel(&gpu, &shaders::gemv_t6u4(), 3),
            k_linear_t8u4_base: kernel(&gpu, shaders::GEMV_MROW_T8U4, 3),
            k_linear_t6v2: kernel(&gpu, shaders::GEMV_T6_PAIRED, 3),
            k_linear_t6hw: kernel(&gpu, &shaders::gemv_t6_halfwarp(), 3),
            k_linear_t6qw: kernel(&gpu, &shaders::gemv_t6_quarterwarp(), 3),
            k_linear_t6fl: kernel(&gpu, &shaders::gemv_t6_fourlane(), 3),
            // v8: 32 lanes/row, two rows per lane group (16 rows per workgroup).
            // v9: 8 lanes/row, two rows per lane group (64 rows per workgroup).
            // v10: same 2-row blocking at 128 threads (32 rows per workgroup).
            // v11: 8 lanes/row, one row per lane group, vectorised loads only.
            // v12/v13: like v10 with 2 and 4 granules unrolled per iteration.
            k_linear_t6v8: kernel(&gpu, &shaders::gemv_t6_vec4(32, 256, 2, 1, false), 3),
            k_linear_t6v9: kernel(&gpu, &shaders::gemv_t6_vec4(8, 256, 2, 1, false), 3),
            k_linear_t6v10: kernel(&gpu, &shaders::gemv_t6_vec4(8, 128, 2, 1, false), 3),
            k_linear_t6v11: kernel(&gpu, &shaders::gemv_t6_vec4(8, 256, 1, 1, false), 3),
            k_linear_t6v12: kernel(&gpu, &shaders::gemv_t6_vec4(8, 128, 2, 2, false), 3),
            k_linear_t6v13: kernel(&gpu, &shaders::gemv_t6_vec4(8, 128, 2, 4, false), 3),
            k_linear_t6v14: kernel(&gpu, &shaders::gemv_t6_vec4(8, 128, 1, 2, false), 3),
            // Same shape as the default, but able to slide its six-token window
            // across a long prompt (prefill).
            k_linear_t6win: kernel(&gpu, &shaders::gemv_t6_vec4(8, 128, 2, 2, true), 3),
            // The vectorised 2-row kernel is the default. The older scalar
            // variants stay reachable through MING_T6=base|spec|v2..v5.
            t6_vec4: std::env::var("MING_T6")
                .map_or(true, |v| !matches!(v.as_str(), "base" | "spec" | "v2" | "v3" | "v4" | "v5")),
            t6_specialized: std::env::var("MING_T6").map_or(true, |v| v != "base"),
            t6_paired: std::env::var("MING_T6").map_or(true, |v| v != "base" && v != "spec"),
            t6_halfwarp: std::env::var("MING_T6").is_ok_and(|v| v == "v3"),
            t6_quarterwarp: std::env::var("MING_T6").is_ok_and(|v| v == "v4"),
            t6_fourlane: std::env::var("MING_T6").is_ok_and(|v| v == "v5"),
            t6_hybrid: std::env::var("MING_T6").map_or(true, |v| v == "hybrid"),
            k_prefill_tile: kernel(&gpu, shaders::GEMM_PREFILL_48X64, 3),
            prefill_tiled: std::env::var("MING_PREFILL").map_or(true, |v| v != "base"),
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

    /// Which single-token kernel a layer gets. `None` means the generic
    /// warp-per-row vec4 kernel, which is what wide projections use.
    fn t1_plan(&self, lin: &Lin) -> Option<(&crate::nn::Kernel, usize)> {
        if let Some(v) = self.t1_variant {
            // Rows per workgroup for (8,64,1), (8,128,1), (8,128,2), (16,128,1).
            const ROWS: [usize; 4] = [8, 16, 32, 8];
            return Some((&self.k_t1v[v], ROWS[v]));
        }
        // Half-warp only pays when both dimensions are small (LLM q/k/v/o).
        // Wide projections already saturate memory, and long K doubles the
        // per-lane work without buying occupancy.
        if self.t1_halfwarp && lin.rows <= 1024 && lin.in_features <= 1024 {
            Some((&self.k_linear_t1hw, 16))
        } else {
            None
        }
    }

    fn t6_plan(&self, lin: &Lin) -> Option<(&crate::nn::Kernel, usize)> {
        if lin.in_features % 256 != 0 {
            return None;
        }
        // Four lanes/row wins the isolated K=4096 microbench but only emits
        // 16 workgroups for a [1024, 4096] layer, so it under-fills the 20 SMs
        // and loses end-to-end. Keep it behind MING_T6=v5 only.
        let use_four = self.t6_fourlane;
        let use_quarter = self.t6_quarterwarp
            || (self.t6_hybrid && lin.in_features <= 2048);
        let plan = if self.t6_vec4 {
            match std::env::var("MING_T6").unwrap_or_default().as_str() {
                "v8" => (&self.k_linear_t6v8, 16),
                "v9" => (&self.k_linear_t6v9, 64),
                "v10" => (&self.k_linear_t6v10, 32),
                "v11" => (&self.k_linear_t6v11, 32),
                "v12" => (&self.k_linear_t6v12, 32),
                "v13" => (&self.k_linear_t6v13, 32),
                "v14" => (&self.k_linear_t6v14, 16),
                // Default: two rows per lane group, two granules unrolled.
                _ => (&self.k_linear_t6v12, 32),
            }
        } else if use_four {
            (&self.k_linear_t6fl, 64)
        } else if use_quarter {
            (&self.k_linear_t6qw, 32)
        } else if self.t6_halfwarp {
            (&self.k_linear_t6hw, 16)
        } else if self.t6_paired {
            (&self.k_linear_t6v2, 8)
        } else if self.t6_specialized {
            (&self.k_linear_t8u4, 8)
        } else {
            (&self.k_linear_t8u4_base, 8)
        };
        Some(plan)
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
        if t == 1 && lin.in_features % 8 == 0 && lin.rows % 8 == 0 {
            let (k, per_wg) = self
                .t1_plan(lin)
                .unwrap_or((&self.k_linear_v4, 8));
            let key = lin.id.rotate_left(24) ^ x_slot.rotate_left(12) ^ y_slot;
            k.run_cached(
                &self.gpu,
                key,
                &uni2(lin.rows as u32, (lin.in_features / 8) as u32),
                &[&lin.w, x, &lin.b],
                y,
                lin.rows.div_ceil(per_wg) as u32,
                1,
            );
            return;
        }
        self.linear_into(lin, x, y, t);
    }

    pub fn linear_into(&self, lin: &Lin, x: &Buffer, y: &Buffer, t: usize) {
        // Prompts longer than the 8-token kernel's window: the generic path
        // dispatches one workgroup per (row block, token), so a 117-token
        // prompt re-reads the entire weight set 117 times. It runs at ~150 GB/s
        // doing so -- i.e. the bandwidth is fine and the redundancy is not.
        // Walking the prompt in chunks of six lets the vectorised kernel share
        // each weight read across six tokens, which is a 6x traffic cut.
        //
        // The default kernel's granule stride is 16 vec4 = 128 K elements, which
        // is why the guard is % 128 and not t6_plan's % 256: the LLM's 896 and
        // 4864 are both multiples of 128 but not of 256.
        if t > 8 && self.t6_vec4 && lin.in_features % 128 == 0 {
            let per_wg = 32;
            let mut tok0 = 0usize;
            while tok0 < t {
                let n = (t - tok0).min(6);
                self.k_linear_t6win.run(
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
        // Long prompts are wide enough to amortise a real 2D tile. The generic
        // path below still covers odd shapes and remains available via
        // MING_PREFILL=base for A/B. Widening the range to 512 was measured at
        // t=117 and came out neutral, so the tile kernel itself -- not the
        // range -- is what limits prefill.
        if self.prefill_tiled
            && (16..=48).contains(&t)
            && lin.rows % 64 == 0
            && lin.in_features % 32 == 0
            // The 896x4864 down projection only yields 14 N tiles, so the tiled
            // grid cannot fill all SMs; the generic token-parallel kernel wins.
            && !(lin.rows < 1024 && lin.in_features >= 4096)
        {
            self.k_prefill_tile.run(
                &self.gpu,
                &uni4(
                    lin.rows as u32,
                    lin.in_features as u32,
                    t as u32,
                    (lin.in_features / 8) as u32,
                ),
                &[&lin.w, x, &lin.b],
                y,
                (lin.rows / 64) as u32,
                (t.div_ceil(48)) as u32,
            );
            return;
        }
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
            let (k, per_wg) = self
                .t1_plan(lin)
                .unwrap_or((&self.k_linear_v4, 8));
            k.run(
                &self.gpu,
                &uni2(lin.rows as u32, (lin.in_features / 8) as u32),
                &[&lin.w, x, &lin.b],
                y,
                lin.rows.div_ceil(per_wg) as u32,
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
        self.k_silu.run(&self.gpu, &uni1(n as u32), &[g, u], &y, ((n + 255) / 256) as u32, 1);
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
        self.k_gelu
            .run(&self.gpu, &uni1(n as u32), &[x], y, n.div_ceil(256) as u32, 1);
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
        self.k_attend
            .run(&self.gpu, &u, &[q, kc, vc], y, n_q_heads as u32, p.t_q as u32);
    }

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
        self.k_attend.run_cached(
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
            ((rows * dim + 255) / 256) as u32,
            1,
        );
    }

    #[allow(dead_code)]
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
            ((rows * dim + 255) / 256) as u32,
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
            ((rows * dim + 255) / 256) as u32,
            1,
        );
    }

    #[allow(dead_code)]
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
            ((rows * dim + 255) / 256) as u32,
            1,
        );
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