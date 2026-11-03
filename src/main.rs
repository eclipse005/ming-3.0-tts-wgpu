//! ming-tts: pure-Rust wgpu port of Ming-3.0-tts-0.5b.
//!
//! CLI surface mirrors infer.py one-to-one (acceptance requirement); the
//! synth subcommand reports which milestone gates each flag until the
//! corresponding port stage lands (see PROGRESS.md).

use anyhow::Result;
use clap::{Parser, Subcommand};

mod audio;
mod backend;
mod cpu;
mod dit;
mod golden;
mod gpu;
mod istft;
mod nn;
mod ops;
mod pt;
#[cfg(test)]
mod parity;
#[cfg(test)]
mod parity_m5;
mod pipeline;
mod qwen_lm;
mod rng;
mod shaders;
mod synth;
mod spk;
mod tasks;
mod text;
mod tn;
mod tok;
mod vae;
mod vae_dec;
mod weights;

#[derive(Parser)]
#[command(name = "ming-tts", version, about = "Ming-3.0-tts-0.5b on wgpu")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)] // parsed once at startup
enum Cmd {
    /// List GPU adapters visible to wgpu.
    Devices,
    /// Synthesize speech (same argument surface as Python infer.py).
    /// rename_all keeps the flags byte-identical to infer.py's (--model_path,
    /// --no_text_normalization, ...) instead of clap's default kebab-case.
    #[command(rename_all = "snake_case")]
    Synth {
        #[command(flatten)]
        args: synth::Args,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Devices => {
            for a in gpu::list_adapters() {
                println!("{a}");
            }
            Ok(())
        }
        Cmd::Synth { args } => synth::run(args),
    }
}
