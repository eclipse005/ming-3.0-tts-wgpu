//! ming-tts: pure-Rust wgpu port of Ming-3.0-tts-0.5b.
//!
//! The CLI surface mirrors infer.py one-to-one, down to the flag spelling;
//! `synth` is the only subcommand besides `devices`.

use anyhow::Result;
use clap::{Parser, Subcommand};
// The CLI is a consumer of the library, not a second copy of the modules.
use ming_tts_wgpu::{gpu, synth, worker};

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
    /// Persistent JSON-lines worker: loads the model once, then serves one
    /// synthesis request per stdin line. Used by service/ming_tts_service.py.
    #[command(rename_all = "snake_case")]
    Worker {
        #[command(flatten)]
        args: worker::Args,
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
        Cmd::Worker { args } => worker::run(args),
    }
}
