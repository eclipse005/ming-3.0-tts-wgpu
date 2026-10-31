//! Thin dispatch wrappers around the WGSL kernels (parity-grade simplicity;
//! pipeline caching lands in M5).

use crate::gpu::Gpu;
use crate::shaders;
use anyhow::Result;
use wgpu::{BindGroupLayout, ComputePipeline, ShaderModuleDescriptor};

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
            has_dynamic_offset: false,
            min_binding_size: None,
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

fn uniform(gpu: &Gpu, data: &[u8]) -> wgpu::Buffer {
    let buf = gpu.buffer(16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "u");
    gpu.queue.write_buffer(&buf, 0, data);
    buf
}

/// y = rmsnorm(x) * w, one row of length n per workgroup.
pub fn rmsnorm(gpu: &Gpu, x: &[f32], w: &[f32], n: usize, eps: f32) -> Result<Vec<f32>> {
    assert_eq!(x.len() % n, 0, "x rows not multiple of n");
    let rows = x.len() / n;
    let x_buf = gpu.upload_f32(x, "x");
    let w_buf = gpu.upload_f32(w, "w");
    let y_buf = gpu.storage((rows * n * 4) as u64, "y");
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&(n as u32).to_le_bytes());
    u[4..8].copy_from_slice(&eps.to_le_bytes());
    let u_buf = uniform(gpu, &u);

    let (bgl, pipeline) = make_pipeline(gpu, shaders::RMSNORM, 2);
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
        ],
    });
    gpu.run(&pipeline, &bind, rows as u32);
    gpu.read_f32(&y_buf, rows * n)
}

/// y = W @ x + b, W bf16 [rows, in] packed 2/u32; pass zeros for no bias.
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
    let x_buf = gpu.upload_f32(x, "x");
    let b_buf = gpu.upload_f32(bias, "b");
    let y_buf = gpu.storage((rows * 4) as u64, "y");
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
    u[4..8].copy_from_slice(&(in_words as u32).to_le_bytes());
    let u_buf = uniform(gpu, &u);

    let (bgl, pipeline) = make_pipeline(gpu, shaders::GEMV_BF16W, 3);
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: y_buf.as_entire_binding() },
        ],
    });
    gpu.run(&pipeline, &bind, ((rows + 7) / 8) as u32);
    gpu.read_f32(&y_buf, rows)
}
