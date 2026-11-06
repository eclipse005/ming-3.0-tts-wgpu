//! Library surface of the Ming-3.0-tts-0.5b wgpu port.
//!
//! The binary (`src/main.rs`) is a thin CLI over this crate. In-process
//! consumers (VoxTrans) use [`synth::Engine`] directly: load the model once,
//! then call [`synth::plan`] + [`synth::Engine::synthesize`] per utterance.
//! No subprocess, no socket, no serialized protocol.
//!
//! ```ignore
//! let mut args = ming_tts_wgpu::synth::Args::default();
//! args.model_path = model_dir.to_string_lossy().into_owned();
//! args.device = "cuda:0".into();
//! let mut engine = ming_tts_wgpu::synth::Engine::load(&args, true)?;
//! // per utterance: fill args.text / args.output / voice fields
//! let plan = ming_tts_wgpu::synth::plan(&args)?;
//! engine.synthesize(&args, &plan)?;
//! ```
//!
//! `Engine` contains `std::rc::Rc` internally and is therefore **not `Send`**:
//! keep it on the thread that created it and drive it through an mpsc channel.

pub mod audio;
mod backend;
mod cpu;
mod dit;
mod golden;
pub mod gpu;
mod istft;
mod nn;
mod ops;
mod pt;
mod pipeline;
mod qwen_lm;
mod rng;
mod shaders;
#[cfg(test)]
mod smoke;
pub mod synth;
mod spk;
pub mod tasks;
mod text;
mod tn;
mod tok;
mod vae;
mod vae_dec;
mod weights;
pub mod worker;
