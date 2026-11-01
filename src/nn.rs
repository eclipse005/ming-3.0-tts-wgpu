//! Thin dispatch wrappers around the WGSL kernels.
//!
//! Every kernel takes one uniform block at binding 0. That block lives in a
//! single device-wide arena (`Gpu::alloc_uniform`) and is addressed by a
//! dynamic offset, so recording a dispatch costs no buffer creation and no
//! queue write -- two costs that used to total ~10.5 us per dispatch, i.e.
//! ~1.2 ms in every DiT forward.

use crate::gpu::{Gpu, UNIFORM_SLOT};
use crate::shaders;
use anyhow::Result;
use std::cell::RefCell;
use std::collections::HashMap;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline, ShaderModuleDescriptor};

fn make_pipeline(gpu: &Gpu, src: &str, storage_ro: usize) -> (BindGroupLayout, ComputePipeline) {
    let module = gpu.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(UNIFORM_SLOT as u64),
        },
        count: None,
    }];
    for i in 0..storage_ro {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (i + 1) as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: (storage_ro + 1) as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    let bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(
                &gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                }),
            ),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    (bgl, pipeline)
}

/// Record one dispatch: uniform block -> arena slot, bind, enqueue.
fn record(
    gpu: &Gpu,
    bgl: &BindGroupLayout,
    pipe: &ComputePipeline,
    uni: &[u8],
    ro: &[&Buffer],
    rw: &Buffer,
    wx: u32,
    wy: u32,
) {
    let (bind, offset) = make_bind(gpu, bgl, uni, ro, rw);
    gpu.run2d(pipe, &bind, offset, wx, wy);
}

/// Build one bind group. The uniform slot index is returned separately because
/// it is passed as a dynamic offset, not baked into the group.
fn make_bind(
    gpu: &Gpu,
    bgl: &BindGroupLayout,
    uni: &[u8],
    ro: &[&Buffer],
    rw: &Buffer,
) -> (wgpu::BindGroup, u32) {
    let offset = gpu.alloc_uniform(uni);
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: gpu.arena_buffer(),
            offset: 0,
            size: wgpu::BufferSize::new(UNIFORM_SLOT as u64),
        }),
    }];
    for (i, b) in ro.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: (i + 1) as u32,
            resource: b.as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: (ro.len() + 1) as u32,
        resource: rw.as_entire_binding(),
    });
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bgl,
        entries: &entries,
    });
    (bind, offset)
}

pub(crate) struct Kernel {
    pub bgl: BindGroupLayout,
    pub pipe: ComputePipeline,
    /// Groups for callers whose buffers live for the whole model. The caller's
    /// key must identify the buffer set, not merely the operation.
    cache: RefCell<HashMap<u64, wgpu::BindGroup>>,
}

pub(crate) fn kernel(gpu: &Gpu, src: &str, storage_ro: usize) -> Kernel {
    let (bgl, pipe) = make_pipeline(gpu, src, storage_ro);
    Kernel {
        bgl,
        pipe,
        cache: RefCell::new(HashMap::new()),
    }
}

impl Kernel {
    /// uniform bytes, read-only buffers in binding order 1.., rw output last.
    pub fn run(
        &self,
        gpu: &Gpu,
        uni: &[u8],
        ro: &[&Buffer],
        rw: &Buffer,
        wx: u32,
        wy: u32,
    ) {
        record(gpu, &self.bgl, &self.pipe, uni, ro, rw, wx, wy);
    }

    /// Record a dispatch, reusing the bind group previously stored under `key`.
    ///
    /// The caller must keep the same buffer set for a key. DiT uses persistent
    /// scratch and immutable weights, so its groups survive all CFM steps;
    /// one-shot paths must keep using [`Self::run`].
    pub fn run_cached(
        &self,
        gpu: &Gpu,
        key: u64,
        uni: &[u8],
        ro: &[&Buffer],
        rw: &Buffer,
        wx: u32,
        wy: u32,
    ) {
        let cache_on = std::env::var("MING_BIND_CACHE").as_deref() != Ok("0");
        if cache_on {
            if let Some(bind) = self.cache.borrow().get(&key) {
                let offset = gpu.alloc_uniform(uni);
                gpu.run2d(&self.pipe, bind, offset, wx, wy);
                return;
            }
        }
        let (bind, offset) = make_bind(gpu, &self.bgl, uni, ro, rw);
        gpu.run2d(&self.pipe, &bind, offset, wx, wy);
        self.cache.borrow_mut().insert(key, bind);
    }
}

// ---- one-shot helpers (M0 parity tests, small projections) ----

/// y = rmsnorm(x) * w, one row of length n per workgroup.
#[cfg(test)]
pub fn rmsnorm(gpu: &Gpu, x: &[f32], w: &[f32], n: usize, eps: f32) -> Result<Vec<f32>> {
    assert_eq!(x.len() % n, 0, "x rows not multiple of n");
    let rows = x.len() / n;
    let x_buf = gpu.upload_f32(x, "x");
    let w_buf = gpu.upload_f32(w, "w");
    let y_buf = gpu.storage((rows * n * 4) as u64, "y");
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&(n as u32).to_le_bytes());
    u[4..8].copy_from_slice(&eps.to_le_bytes());
    let (bgl, pipe) = make_pipeline(gpu, shaders::RMSNORM, 2);
    record(gpu, &bgl, &pipe, &u, &[&x_buf, &w_buf], &y_buf, rows as u32, 1);
    gpu.read_f32(&y_buf, rows * n)
}

/// gemv with already-resident weight/bias buffers (no re-upload per call).
pub fn gemv_bf16w_buf(
    gpu: &Gpu,
    w_buf: &Buffer,
    b_buf: &Buffer,
    x: &[f32],
    rows: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let x_buf = gpu.upload_f32(x, "x");
    let y_buf = gpu.storage((rows * 4) as u64, "y");
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
    u[4..8].copy_from_slice(&((in_features / 2) as u32).to_le_bytes());
    let (bgl, pipe) = make_pipeline(gpu, shaders::GEMV_BF16W, 3);
    record(gpu, &bgl, &pipe, &u, &[w_buf, &x_buf, b_buf], &y_buf,
           ((rows + 7) / 8) as u32, 1);
    gpu.read_f32(&y_buf, rows)
}

/// y = W @ x + b, W bf16 [rows, in] packed 2/u32; pass zeros for no bias.
#[cfg(test)]
pub fn gemv_bf16w(
    gpu: &Gpu,
    w_packed: &[u32],
    x: &[f32],
    bias: &[f32],
    rows: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let in_words = in_features / 2;
    assert_eq!(w_packed.len(), rows * in_words, "packed weight size mismatch");
    assert_eq!(x.len(), in_features, "activation length mismatch");
    assert_eq!(bias.len(), rows, "bias length mismatch");
    let w_buf = gpu.upload_u32(w_packed, "w");
    let b_buf = gpu.upload_f32(bias, "b");
    gemv_bf16w_buf(gpu, &w_buf, &b_buf, x, rows, in_features)
}
