//! wgpu device setup and buffer helpers (lean port of the qwen3-asr-wgpu patterns).

use anyhow::{Context, Result};
use std::cell::RefCell;
use wgpu::{Backends, Buffer, BufferUsages, Device, Instance, Queue};

// Device/Queue are internally reference counted, so a clone shares the same
// GPU device -- that is how the LLM, DiT and VAE stages end up on one device.
#[derive(Clone)]
pub struct Gpu {
    pub device: Device,
    pub queue: Queue,
    /// Informational, printed by `devices` and useful in logs.
    #[allow(dead_code)]
    pub adapter_name: String,
    #[allow(dead_code)]
    pub backend: &'static str,
    /// Dispatch batching. Every kernel launch used to create its own command
    /// encoder and submit it, which is thousands of WDDM submissions per
    /// utterance (the CUDA build was saved by CUDA graphs for the same
    /// reason). Launches now accumulate into one encoder that is flushed at
    /// the next synchronisation point (a readback) or when it grows large.
    dispatches: std::sync::Arc<RefCell<Vec<PendingDispatch>>>,
    pending_count: std::sync::Arc<std::cell::Cell<usize>>,
    /// One upload buffer for every dispatch's uniform block. Each recorded
    /// dispatch owns a 256-byte slot and the whole thing is uploaded once per
    /// flush; that is the only way to keep per-dispatch uniforms *and* a
    /// deferred submission, because Queue::write_buffer would otherwise land
    /// before the batch it is supposed to configure.
    arena: Buffer,
    arena_host: std::sync::Arc<RefCell<Vec<u8>>>,
    arena_used: std::sync::Arc<std::cell::Cell<usize>>,
    /// Bytes handed to `device.create_buffer` so far, and the largest amount
    /// created between two flushes (a flush is what lets the previous wave of
    /// recordings -- and therefore their buffers -- go). `MING_MEM=1` prints
    /// both: the first is "what the pipeline asked for", the second is a good
    /// proxy for the peak live set of one stage.
    alloc_bytes: std::sync::Arc<std::cell::Cell<u64>>,
    alloc_since_flush: std::sync::Arc<std::cell::Cell<u64>>,
    peak_window: std::sync::Arc<std::cell::Cell<u64>>,
}

struct PendingDispatch {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    uniform_offset: u32,
    wgs_x: u32,
    wgs_y: u32,
}

/// Uniform slots are 256 B: the guaranteed minUniformBufferOffsetAlignment.
pub const UNIFORM_SLOT: usize = 256;
const ARENA_SLOTS: usize = 4096;

const MAX_PENDING: usize = 2048;

fn max_pending() -> usize {
    std::env::var("MING_MAX_PENDING")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_PENDING)
}

/// Every backend wgpu can drive a compute queue on. Vulkan is preferred on
/// Windows/Linux (DX12 enumeration costs ~1s and Vulkan is the tested path on
/// Pascal); Metal is what macOS exposes; GL is the last resort.
fn backends() -> Backends {
    Backends::VULKAN | Backends::METAL | Backends::DX12 | Backends::GL
}

fn backend_rank(b: wgpu::Backend) -> u8 {
    match b {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        wgpu::Backend::Gl => 3,
        _ => 4,
    }
}

fn instance() -> Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends();
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
        let adapters = pollster::block_on(inst.enumerate_adapters(backends()));
        let mut picked: Option<(u8, wgpu::Adapter)> = None;
        for adapter in adapters {
            let info = adapter.get_info();
            if info.device_type == wgpu::DeviceType::Cpu {
                continue;
            }
            let rank = backend_rank(info.backend);
            if picked.as_ref().map_or(true, |(best, _)| rank < *best) {
                picked = Some((rank, adapter));
            }
        }
        let picked = picked.map(|(_, a)| a);
        let adapter = picked.context("no discrete/integrated GPU adapter found")?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("ming-tts"),
                    required_features: wgpu::Features::empty(),
                    // Take what the adapter offers instead of wgpu's conservative
                    // baseline: bigger workgroups and storage bindings become
                    // available on capable cards without changing any shader.
                    required_limits: adapter.limits(),
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
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform arena"),
            size: (UNIFORM_SLOT * ARENA_SLOTS) as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            device,
            queue,
            adapter_name: info.name,
            backend,
            dispatches: std::sync::Arc::new(RefCell::new(Vec::new())),
            pending_count: std::sync::Arc::new(std::cell::Cell::new(0)),
            arena,
            arena_host: std::sync::Arc::new(RefCell::new(Vec::new())),
            arena_used: std::sync::Arc::new(std::cell::Cell::new(0)),
            alloc_bytes: std::sync::Arc::new(std::cell::Cell::new(0)),
            alloc_since_flush: std::sync::Arc::new(std::cell::Cell::new(0)),
            peak_window: std::sync::Arc::new(std::cell::Cell::new(0)),
        })
    }

    pub fn buffer(&self, size: u64, usage: BufferUsages, label: &str) -> Buffer {
        self.alloc_bytes.set(self.alloc_bytes.get() + size);
        self.alloc_since_flush
            .set(self.alloc_since_flush.get() + size);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(1),
            usage,
            mapped_at_creation: false,
        })
    }

    /// (cumulative bytes, peak bytes allocated in one flush window).
    pub fn mem_stats(&self) -> (u64, u64) {
        (self.alloc_bytes.get(), self.peak_window.get())
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
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|e| anyhow::anyhow!("map channel: {e}"))?
            .map_err(|e| anyhow::anyhow!("map failed: {e}"))?;
        let out = bytemuck::cast_slice(&slice.get_mapped_range()?).to_vec();
        stage.unmap();
        Ok(out)
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
    /// Reserve a uniform slot, copying the bytes into the arena's host staging
    /// area. Returns the byte offset to pass as a dynamic bind-group offset.
    pub fn alloc_uniform(&self, data: &[u8]) -> u32 {
        if data.len() > UNIFORM_SLOT {
            panic!("uniform block of {} B exceeds the {UNIFORM_SLOT} B slot", data.len());
        }
        let mut used = self.arena_used.get();
        if used + UNIFORM_SLOT > UNIFORM_SLOT * ARENA_SLOTS {
            self.flush();
            used = 0;
        }
        let mut host = self.arena_host.borrow_mut();
        if host.len() < used + UNIFORM_SLOT {
            host.resize(used + UNIFORM_SLOT, 0);
        }
        host[used..used + data.len()].copy_from_slice(data);
        for b in &mut host[used + data.len()..used + UNIFORM_SLOT] {
            *b = 0;
        }
        self.arena_used.set(used + UNIFORM_SLOT);
        used as u32
    }

    pub fn arena_buffer(&self) -> &Buffer {
        &self.arena
    }

    pub fn flush(&self) {
        // Upload every uniform written since the last flush in one go, before
        // the batch that reads them is submitted.
        let used = self.arena_used.get();
        if used > 0 {
            let host = self.arena_host.borrow();
            self.queue.write_buffer(&self.arena, 0, &host[..used]);
        }
        self.arena_used.set(0);
        // Everything recorded since the previous flush is about to become
        // recyclable, so this is the point where the live set peaks.
        let window = self.alloc_since_flush.get();
        if window > self.peak_window.get() {
            self.peak_window.set(window);
        }
        self.alloc_since_flush.set(0);
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
                pass.set_bind_group(0, &d.bind, &[d.uniform_offset]);
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
        uniform_offset: u32,
        wgs_x: u32,
        wgs_y: u32,
    ) {
        {
            let mut dq = self.dispatches.borrow_mut();
            dq.push(PendingDispatch {
                pipeline: pipeline.clone(),
                bind: bind.clone(),
                uniform_offset,
                wgs_x,
                wgs_y,
            });
        }
        let n = self.pending_count.get() + 1;
        self.pending_count.set(n);
        if n % max_pending() == 0 {
            self.flush();
        }
    }
}