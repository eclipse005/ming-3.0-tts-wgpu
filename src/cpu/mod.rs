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
//! * `pipeline` - the decode loop and conditioning;
//! * `run` - the `--device cpu` entry point.

mod dit;
mod llm;
mod pipeline;
mod run;
mod tensor;
mod vae;

pub use pipeline::CpuPipeline;
pub use run::run;
// The parity tests reach the stage types through crate::cpu::*;
// the binary only needs run(), so keep these out of release builds.
#[cfg(test)]
pub use dit::CpuDit;
#[cfg(test)]
pub use llm::CpuLlm;
#[cfg(test)]
pub use tensor::{CpuLin, Mat};
#[cfg(test)]
pub use vae::{CpuVaeDec, CpuVaeEnc};

/// Latent width of one audio frame (shared by the DiT and both VAE towers).
pub const ZC: usize = 64;
