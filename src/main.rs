//! ming-tts: pure-Rust wgpu port of Ming-3.0-tts-0.5b.
//!
//! CLI surface mirrors infer.py one-to-one (acceptance requirement); the
//! synth subcommand reports which milestone gates each flag until the
//! corresponding port stage lands (see PROGRESS.md).

use anyhow::Result;
use clap::{Parser, Subcommand};

mod dit;
mod gpu;
mod nn;
mod parity;
mod qwen_lm;
mod shaders;
mod weights;

#[derive(Parser)]
#[command(name = "ming-tts", version, about = "Ming-3.0-tts-0.5b on wgpu")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List GPU adapters visible to wgpu.
    Devices,
    /// Synthesize speech (same argument surface as Python infer.py).
    Synth {
        #[arg(long)] model_path: String,
        #[arg(long)] task: String,
        #[arg(long)] text: String,
        #[arg(long)] output: String,
        #[arg(long, default_value = "cuda:0")] device: String,
        #[arg(long, default_value_t = 1895)] seed: i64,
        #[arg(long)] language_id: Option<String>,
        #[arg(long)] instruction: Option<String>,
        #[arg(long)] instruction_schema: Option<String>,
        #[arg(long)] speed: Option<String>,
        #[arg(long)] pitch: Option<String>,
        #[arg(long)] volume: Option<String>,
        #[arg(long)] emotion: Option<String>,
        #[arg(long)] dialect: Option<String>,
        #[arg(long)] style: Option<String>,
        #[arg(long)] ip: Option<String>,
        #[arg(long, default_value_t = 2.0)] cfg: f64,
        #[arg(long, default_value_t = 0.25)] sigma: f64,
        #[arg(long, default_value_t = 0.0)] temperature: f64,
        #[arg(long, default_value_t = 24)] max_length: usize,
        #[arg(long, default_value_t = 1000)] max_decode_steps: usize,
        #[arg(long)] prompt_wav_path: Option<String>,
        #[arg(long)] prompt_text: Option<String>,
        #[arg(long)] spk_emb_path: Option<String>,
        #[arg(long)] use_spk_emb: bool,
        #[arg(long)] use_zero_spk_emb: bool,
        #[arg(long)] no_text_normalization: bool,
        #[arg(long)] raw_text: bool,
        #[arg(long)] no_trim_silence: bool,
    },
}

fn main() -> Result<()> {
    match &Cli::parse().cmd {
        Cmd::Devices => {
            for a in gpu::list_adapters() {
                println!("{a}");
            }
            Ok(())
        }
        Cmd::Synth { task, device, .. } => {
            let _ = task;
            let _ = device;
            eprintln!("synth: not implemented yet.");
            eprintln!("Landing order (see PROGRESS.md): M1 LLM decode -> M2 DiT ->");
            eprintln!("M3/M4 VAE+iSTFT -> M5 full params_test alignment.");
            eprintln!("The full infer.py flag surface above is already accepted and");
            eprintln!("will be honored as each stage lands.");
            std::process::exit(2);
        }
    }
}
