//! ming-tts: pure-Rust wgpu port of Ming-3.0-tts-0.5b.
//!
//! CLI surface mirrors infer.py one-to-one (acceptance requirement); the
//! synth subcommand reports which milestone gates each flag until the
//! corresponding port stage lands (see PROGRESS.md).

use anyhow::Result;
use clap::{Parser, Subcommand};

mod audio;
mod cpu;
mod dit;
mod golden;
mod gpu;
mod istft;
mod nn;
mod ops;
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
enum Cmd {
    /// List GPU adapters visible to wgpu.
    Devices,
    /// Synthesize speech (same argument surface as Python infer.py).
    /// rename_all keeps the flags byte-identical to infer.py's (--model_path,
    /// --no_text_normalization, ...) instead of clap's default kebab-case.
    #[command(rename_all = "snake_case")]
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
        #[arg(long, num_args = 1..)] prompt_wav_path: Vec<String>,
        #[arg(long)] prompt_text: Option<String>,
        #[arg(long, num_args = 1..)] spk_emb_path: Vec<String>,
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
        Cmd::Synth {
            model_path, task, text, output, device, seed, language_id,
            instruction, instruction_schema, speed, pitch, volume, emotion,
            dialect, style, ip, cfg, sigma, temperature, max_length,
            max_decode_steps, prompt_wav_path, prompt_text, spk_emb_path,
            use_spk_emb, use_zero_spk_emb, no_text_normalization, raw_text,
            no_trim_silence,
        } => {
            synth::run(synth::Args {
                model_path: model_path.clone(),
                task: task.clone(),
                text: text.clone(),
                output: output.clone(),
                device: device.clone(),
                seed: *seed,
                language_id: language_id.clone(),
                instruction: instruction.clone(),
                instruction_schema: instruction_schema.clone(),
                speed: speed.clone(),
                pitch: pitch.clone(),
                volume: volume.clone(),
                emotion: emotion.clone(),
                dialect: dialect.clone(),
                style: style.clone(),
                ip: ip.clone(),
                cfg: *cfg,
                sigma: *sigma,
                temperature: *temperature,
                max_length: *max_length,
                max_decode_steps: *max_decode_steps,
                prompt_wav_path: prompt_wav_path.clone(),
                prompt_text: prompt_text.clone(),
                spk_emb_path: spk_emb_path.clone(),
                use_spk_emb: *use_spk_emb,
                use_zero_spk_emb: *use_zero_spk_emb,
                no_text_normalization: *no_text_normalization,
                raw_text: *raw_text,
                no_trim_silence: *no_trim_silence,
            })
        }
    }
}
