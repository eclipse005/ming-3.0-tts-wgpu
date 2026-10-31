//! wgpu device setup and buffer helpers (lean port of the qwen3-asr-wgpu patterns).

use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use wgpu::{Backends, Buffer, BufferUsages, Device, Instance, Queue};

// Device/Queue are internally reference counted, so a clone shares the same
// GPU device -- that is how the LLM, DiT and VAE stages end up on one device.
#[derive(Clone)]
pub struct Gpu {
    pub device: Device,
    pub queue: Queue,
    pub adapter_name: String,
    pub backend: &'static str,
    /// Dispatch batching. Every kernel launch used to create its own command
    /// encoder and submit it, which is thousands of WDDM submissions per
    /// utterance (the CUDA build was saved by CUDA graphs for the same
    /// reason). Launches now accumulate into one encoder that is flushed at
    /// the next synchronisation point (a readback) or when it grows large.
    dispatches: std::sync::Arc<RefCell<Vec<PendingDispatch>>>,
    pending_count: std::sync::Arc<std::cell::Cell<usize>>,
}

struct PendingDispatch {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    wgs_x: u32,
    wgs_y: u32,
}

const MAX_PENDING: usize = 512;

fn max_pending() -> usize {
    std::env::var("MING_MAX_PENDING")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_PENDING)
}

fn instance() -> Instance {
    // Vulkan first on Windows: DX12 enumeration costs ~1s and Vulkan is the
    // tested path on Pascal (qwen3-asr-wgpu perf.md).
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = Backends::VULKAN | Backends::DX12;
    Instance::new(desc)
}

pub fn list_adapters() -> Vec<String> {
    let inst = instance();
    let adapters = pollster::block_on(inst.enumerate_adapters(Backends::all()));
    let mut out = Vec::new();
    for adapter in adapters {
        let info = adapter.get_info();
        out.push(format!(
            "[{:?}] {} ({} MiB max storage)",
            info.backend,
            info.name,
            adapter.limits().max_storage_buffer_binding_size >> 20
        ));
    }
    out
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let inst = instance();
        let adapters =
            pollster::block_on(inst.enumerate_adapters(Backends::VULKAN | Backends::DX12));
        let mut picked = None;
        for adapter in adapters {
            let info = adapter.get_info();
            if info.device_type == wgpu::DeviceType::Cpu {
                continue;
            }
            picked = Some(adapter);
            if info.backend == wgpu::Backend::Vulkan {
                break; // prefer Vulkan over DX12 when both expose the device
            }
        }
        let adapter = picked.context("no discrete/integrated GPU adapter found")?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("ming-tts"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                })
                .await
        })
        .context("request_device failed")?;
        let backend = match info.backend {
            wgpu::Backend::Vulkan => "vulkan",
            wgpu::Backend::Dx12 => "dx12",
            _ => "other",
        };
        Ok(Self {
            device,
            queue,
            adapter_name: info.name,
            backend,
            dispatches: std::sync::Arc::new(RefCell::new(Vec::new())),
            pending_count: std::sync::Arc::new(std::cell::Cell::new(0)),
        })
    }

    pub fn buffer(&self, size: u64, usage: BufferUsages, label: &str) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(1),
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn storage(&self, size: u64, label: &str) -> Buffer {
        self.buffer(
            size,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            label,
        )
    }

    pub fn upload_f32(&self, data: &[f32], label: &str) -> Buffer {
        let buf = self.storage(
            (data.len() * 4) as u64,
            label,
        );
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    }

    pub fn upload_u32(&self, data: &[u32], label: &str) -> Buffer {
        let buf = self.storage((data.len() * 4) as u64, label);
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    }

    pub fn read_f32(&self, buf: &Buffer, len: usize) -> Result<Vec<f32>> {
        self.read_f32_offset(buf, len, 0)
    }

    pub fn read_f32_offset(&self, buf: &Buffer, len: usize, offset: u64) -> Result<Vec<f32>> {
        self.flush();
        let size = (len * 4) as u64;
        let stage = self.buffer(
            size,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            "readback",
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("read") });
        enc.copy_buffer_to_buffer(buf, offset, &stage, 0, size);
        self.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        let slice = stage.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|e| anyhow::anyhow!("map channel: {e}"))?
            .map_err(|e| anyhow::anyhow!("map failed: {e}"))?;
        let out = bytemuck::cast_slice(&slice.get_mapped_range()?).to_vec();
        stage.unmap();
        Ok(out)
    }

    pub fn run(&self, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, wgs: u32) {
        self.run2d(pipeline, bind, wgs, 1);
    }

    /// Record a dispatch into the pending batch.
    ///
    /// The batch is replayed into ONE compute pass at flush time. That is not
    /// a micro-optimisation: a pass per dispatch made every kernel pay the
    /// full pipeline-drain latency (a 3-token DiT forward measured 17 ms with
    /// ~110 dispatches, i.e. ~150 us each for ~3.5 us of arithmetic), which is
    /// the same launch-latency problem the CUDA build solved with CUDA graphs.
    /// WebGPU guarantees each dispatch sees the previous one's writes, so the
    /// ordering is preserved.
    pub fn flush(&self) {
        let mut dq = self.dispatches.borrow_mut();
        if dq.is_empty() {
            return;
        }
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("batch") },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for d in dq.iter() {
                pass.set_pipeline(&d.pipeline);
                pass.set_bind_group(0, &d.bind, &[]);
                pass.dispatch_workgroups(d.wgs_x, d.wgs_y, 1);
            }
        }
        dq.clear();
        self.queue.submit(Some(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    pub fn run2d(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        wgs_x: u32,
        wgs_y: u32,
    ) {
        {
            let mut dq = self.dispatches.borrow_mut();
            dq.push(PendingDispatch {
                pipeline: pipeline.clone(),
                bind: bind.clone(),
                wgs_x,
                wgs_y,
            });
        }
        let n = self.pending_count.get() + 1;
        self.pending_count.set(n);
        if n % max_pending() == 0 {
            self.flush();
        }
    }}

pub fn assert_f32_multiple(n: usize) -> Result<()> {
    if n % 4 != 0 {
        bail!("f32 count {n} not word-multiple");
    }
    Ok(())
}
