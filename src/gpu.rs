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
    pub device_type: wgpu::DeviceType,
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
    alloc_count: std::sync::Arc<std::cell::Cell<u64>>,
    dispatch_count: std::sync::Arc<std::cell::Cell<u64>>,
    alloc_bytes: std::sync::Arc<std::cell::Cell<u64>>,
    alloc_since_flush: std::sync::Arc<std::cell::Cell<u64>>,
    peak_window: std::sync::Arc<std::cell::Cell<u64>>,
    /// Submits in flight without a wait. Keeping a couple queued lets the CPU
    /// record the next batch while the GPU drains the previous one; a hard cap
    /// (and a byte cap) keeps the live set bounded.
    in_flight: std::sync::Arc<std::cell::Cell<u32>>,
    inflight_bytes: std::sync::Arc<std::cell::Cell<u64>>,
    /// Dispatches per submit. A submit of >=1024 recorded dispatches lost the
    /// device on Vulkan/NVIDIA (Windows TDR while the driver compiles that
    /// batch's pipelines), so the bound is 256 everywhere -- vendor-specific
    /// batching is a footgun and the smaller bound costs nothing measurable.
    /// `MING_MAX_PENDING` overrides.
    max_pending: usize,
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

const MAX_PENDING: usize = 256;

fn max_in_flight() -> u32 {
    std::env::var("MING_IN_FLIGHT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(MAX_IN_FLIGHT)
}
/// Submits kept in flight before a blocking wait. 16 overlaps the CPU's
/// recording with the GPU's execution; the byte ceiling below bounds the live
/// set.
const MAX_IN_FLIGHT: u32 = 16;
/// ... and a byte ceiling, so the VAE decoder's multi-hundred-MB windows do not
/// stack up on a small card.
const MAX_INFLIGHT_BYTES: u64 = 512 << 20;

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

/// Which class of adapter to run on. `--device` maps onto this: `cuda:0` is
/// what `infer.py` gets on the reference side and means the discrete GPU,
/// `cpu` is the software rasteriser (WARP / lavapipe), and `igpu` pins the
/// integrated one (handy for A/B).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevicePref {
    /// Discrete first, then integrated. What production should use.
    Auto,
    Discrete,
    Integrated,
    Cpu,
}

impl DevicePref {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        Ok(match s.as_str() {
            "" | "auto" | "gpu" => Self::Auto,
            "cpu" => Self::Cpu,
            "igpu" | "integrated" | "intel" | "i" => Self::Integrated,
            "d" | "discrete" | "dgpu" | "nvidia" => Self::Discrete,
            other if other.starts_with("cuda") => Self::Discrete,
            other => anyhow::bail!(
                "unknown --device {other:?}: use cuda:0 (discrete), igpu (integrated) or cpu"
            ),
        })
    }

    fn type_rank(self, t: wgpu::DeviceType) -> u8 {
        let is = |want: wgpu::DeviceType| t == want;
        match self {
            // Explicit pins still fall back gracefully (rank 1) if the class
            // is absent, except for CPU, where a GPU would be a surprise.
            DevicePref::Cpu => {
                if is(wgpu::DeviceType::Cpu) { 0 } else { 9 }
            }
            DevicePref::Discrete => {
                if is(wgpu::DeviceType::DiscreteGpu) {
                    0
                } else if is(wgpu::DeviceType::Cpu) {
                    9
                } else {
                    1
                }
            }
            DevicePref::Integrated => {
                if is(wgpu::DeviceType::IntegratedGpu) {
                    0
                } else if is(wgpu::DeviceType::Cpu) {
                    9
                } else {
                    1
                }
            }
            DevicePref::Auto => match t {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::VirtualGpu => 2,
                wgpu::DeviceType::Cpu => 9,
                wgpu::DeviceType::Other => 3,
            },
        }
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
    /// Auto-preference entry point (discrete first). The CLI goes through
    /// [`Gpu::with_pref`] so `--device` is honoured; tests use this one, and
    /// `MING_DEVICE` lets a test run pin a class without a CLI flag.
    #[allow(dead_code)]
    pub fn new() -> Result<Self> {
        let pref = match std::env::var("MING_DEVICE") {
            Ok(v) if !v.trim().is_empty() => DevicePref::parse(&v)?,
            _ => DevicePref::Auto,
        };
        Self::with_pref(pref)
    }

    pub fn with_pref(pref: DevicePref) -> Result<Self> {
        let inst = instance();
        let adapters = pollster::block_on(inst.enumerate_adapters(backends()));
        let mut picked: Option<((u8, u8), wgpu::Adapter)> = None;
        for adapter in adapters {
            let info = adapter.get_info();
            let tr = pref.type_rank(info.device_type);
            if tr == 9 {
                continue;
            }
            let rank = (tr, backend_rank(info.backend));
            if picked.as_ref().map_or(true, |(best, _)| rank < *best) {
                picked = Some((rank, adapter));
            }
        }
        if pref == DevicePref::Cpu && picked.is_none() {
            anyhow::bail!(
                "--device cpu needs a software adapter (Vulkan lavapipe or DX12 WARP), \
                 and none is visible on this machine; use cuda:0 or igpu instead"
            );
        }
        let picked = picked.map(|(_, a)| a);
        let adapter = picked.context("no usable GPU adapter found")?;
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
        // Keep one submit small enough that no single command buffer can sit
        // near the Windows TDR watchdog (~2 s): 256 dispatches is a few ms of
        // GPU work on any supported device, and it is also the batch size at
        // which the driver's lazy pipeline compilation stays bounded. Device
        // independent on purpose -- vendor-specific batching is a footgun.
        let max_pending = std::env::var("MING_MAX_PENDING")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(MAX_PENDING);
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
            device_type: info.device_type,
            dispatches: std::sync::Arc::new(RefCell::new(Vec::new())),
            pending_count: std::sync::Arc::new(std::cell::Cell::new(0)),
            arena,
            arena_host: std::sync::Arc::new(RefCell::new(Vec::new())),
            arena_used: std::sync::Arc::new(std::cell::Cell::new(0)),
            alloc_count: std::sync::Arc::new(std::cell::Cell::new(0)),
            dispatch_count: std::sync::Arc::new(std::cell::Cell::new(0)),
            alloc_bytes: std::sync::Arc::new(std::cell::Cell::new(0)),
            alloc_since_flush: std::sync::Arc::new(std::cell::Cell::new(0)),
            peak_window: std::sync::Arc::new(std::cell::Cell::new(0)),
            in_flight: std::sync::Arc::new(std::cell::Cell::new(0)),
            inflight_bytes: std::sync::Arc::new(std::cell::Cell::new(0)),
            max_pending,
        })
    }

    pub fn buffer(&self, size: u64, usage: BufferUsages, label: &str) -> Buffer {
        self.alloc_count.set(self.alloc_count.get() + 1);
        self.alloc_bytes.set(self.alloc_bytes.get() + size);
        self.alloc_since_flush
            .set(self.alloc_since_flush.get() + size);
        self.inflight_bytes
            .set(self.inflight_bytes.get() + size);
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

    /// Human-readable adapter identity, printed once per synthesis so the log
    /// always says which device the numbers belong to.
    pub fn describe(&self) -> String {
        format!(
            "{} / {} ({:?})",
            self.backend, self.adapter_name, self.device_type
        )
    }

    /// (buffers created, dispatches recorded) since process start.
    pub fn counters(&self) -> (u64, u64) {
        (self.alloc_count.get(), self.dispatch_count.get())
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

    /// Overwrite an existing buffer (no allocation, no extra bind group).
    pub fn write_f32(&self, buf: &Buffer, data: &[f32]) {
        self.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
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
        if !self.submit_pending() {
            return;
        }
        let n = self.in_flight.get() + 1;
        self.in_flight.set(n);
        if n >= max_in_flight() || self.inflight_bytes.get() >= MAX_INFLIGHT_BYTES {
            self.wait_idle();
        } else {
            // Non-blocking: advance callbacks, let the driver start the batch,
            // and return so the CPU can record the next one while it runs.
            let _ = self.device.poll(wgpu::PollType::Poll);
        }
    }

    /// Submit and block until the queue is empty (timing, readbacks, OOM
    /// pressure). Also the only place the in-flight counters reset.
    #[allow(dead_code)] // used by the timing benches; the pipeline uses flush()
    pub fn flush_wait(&self) {
        self.submit_pending();
        self.wait_idle();
    }

    fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        self.in_flight.set(0);
        self.inflight_bytes.set(0);
    }

    /// Upload the uniform arena, replay the recorded dispatches into one
    /// compute pass and submit. Returns false when there was nothing pending.
    fn submit_pending(&self) -> bool {
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
            return false;
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
        true
    }

    pub fn run2d(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        uniform_offset: u32,
        wgs_x: u32,
        wgs_y: u32,
    ) {
        self.dispatch_count.set(self.dispatch_count.get() + 1);
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
        if n % self.max_pending == 0 {
            self.flush();
        }
    }
}