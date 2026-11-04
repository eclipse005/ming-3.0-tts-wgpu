//! Hand-written CPU twin of the GPU pipeline (same approach as
//! `qwen3-asr-wgpu/src/cpu_decoder.rs`): f32 activations, bf16 weights widened
//! on load, rayon for row parallelism and a hand-written weight-reuse GEMM.
//! No NN framework, no wgpu, no device.
//!
//! Layout:
//! * `tensor` - Mat/CpuLin and the shared primitives;
//! * `llm` - Qwen2.5-0.5B prefill + incremental decode;
//! * `dit` - DiT/CFM and the aggregator;
//! * `vae` - VAE encoder/decoder;
//! * `pipeline` - the decode loop, conditioning and the `Backend` impl.

use anyhow::{Context, Result};
use std::path::Path;

mod dit;
mod llm;
mod pipeline;
mod tensor;
mod vae;

pub use pipeline::CpuPipeline;
// The parity tests reach the stage types through crate::cpu::*; the CLI only
// needs CpuPipeline, so keep these out of release builds.
/// Latent width of one audio frame (shared by the DiT and both VAE towers).
pub const ZC: usize = 64;

/// `--device cpu`: load the checkpoint into the host pipeline. The shared
/// `crate::backend::synthesize` drives it from there on.
pub fn load(model_dir: &Path) -> Result<CpuPipeline> {
    println!("device: cpu (hand-written Rust backend, rayon)");
    let t0 = std::time::Instant::now();
    let w = crate::weights::Weights::load(&model_dir.join("model.safetensors"))
        .context("load model.safetensors")?;
    let pl = CpuPipeline::load(&w)?;
    println!("  model loaded in {:?}", t0.elapsed());
    Ok(pl)
}
