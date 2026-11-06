//! Persistent JSON-lines worker: load the checkpoint once, then serve one
//! synthesis request per stdin line.
//!
//! `synth` loads a 2.9 GB checkpoint per utterance; a dubbing job has hundreds
//! of utterances. This subcommand loads the tokenizer + backend once and then
//! serves one request per stdin line.
//!
//! Protocol (all text is UTF-8, one JSON object per line):
//!
//! ```text
//! stdin : {"id":"7","text":"你好","output":"out/7.wav",
//!          "prompt_wav_path":["ref.wav"],"use_spk_emb":true, ...}
//! stdout: @@MING@@{"id":"7","ok":true,"output":"out/7.wav","elapsed_ms":1234}
//! ```
//!
//! Every key of `synth::Args` is accepted (snake_case, same as the CLI flags).
//! `model_path` / `device` are always taken from the worker's own CLI: they
//! describe the already-loaded engine, not an individual utterance. Lines on
//! stdout without the `@@MING@@` marker are the inference log and are ignored
//! by the adapter; `{"cmd":"ping"}` and `{"cmd":"shutdown"}` are control
//! messages that do not touch the model.

use anyhow::{bail, Context, Result};
use clap::Args as ClapArgs;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::io::{self, BufRead, Write};
use std::time::Instant;

/// Prefix that separates protocol messages from the pipeline's own printlns.
pub const MARKER: &str = "@@MING@@";

#[derive(ClapArgs, Debug)]
#[command(rename_all = "snake_case")]
pub struct Args {
    /// Model directory (must match the one the HTTP adapter is configured with).
    #[arg(long)]
    pub model_path: String,
    /// `cuda:0` / `igpu` / `cpu`, resolved once at startup.
    #[arg(long, default_value = "cuda:0")]
    pub device: String,
}

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    id: String,
    #[serde(default)]
    cmd: Option<String>,
    /// All remaining keys are `synth::Args` fields.
    #[serde(flatten)]
    fields: Map<String, Value>,
}

fn emit(v: &Value) -> Result<()> {
    let mut out = io::stdout().lock();
    writeln!(out, "{MARKER}{v}")?;
    out.flush()?;
    Ok(())
}

fn build_args(a: &Args, fields: &Map<String, Value>) -> Result<crate::synth::Args> {
    let mut merged = fields.clone();
    // Never let a request retarget the loaded engine.
    merged.insert("model_path".into(), Value::String(a.model_path.clone()));
    merged.insert("device".into(), Value::String(a.device.clone()));
    let args: crate::synth::Args =
        serde_json::from_value(Value::Object(merged)).context("invalid synth args in request")?;

    if args.text.trim().is_empty() {
        bail!("text must not be empty");
    }
    if args.output.trim().is_empty() {
        bail!("output must not be empty");
    }
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            bail!("output directory does not exist: {}", parent.display());
        }
    }
    for p in args.prompt_wav_path.iter().chain(args.spk_emb_path.iter()) {
        if !std::path::Path::new(p).is_file() {
            bail!("voice file not found: {p}");
        }
    }
    Ok(args)
}

pub fn run(a: Args) -> Result<()> {
    // Golden-noise replay is a CLI parity-debugging aid. A worker always uses
    // a fresh per-request RNG, otherwise every utterance would repeat the
    // recorded draws.
    std::env::remove_var("MING_NOISE_GOLDEN");

    eprintln!("worker: loading {} on {}", a.model_path, a.device);
    let template = crate::synth::Args {
        model_path: a.model_path.clone(),
        device: a.device.clone(),
        ..Default::default()
    };
    // The encoder is resident from the start: a later request may carry a
    // prompt transcript, and reloading the backend mid-session would defeat
    // the whole point of a worker.
    let mut engine = crate::synth::Engine::load(&template, true).context("load model")?;

    emit(&json!({
        "event": "ready",
        "pid": std::process::id(),
        "model_path": a.model_path,
        "device": a.device,
    }))?;
    eprintln!("worker: ready");

    for line in io::stdin().lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("worker: stdin closed: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                emit(&json!({"ok": false, "error": format!("bad request JSON: {e}")}))?;
                continue;
            }
        };

        match req.cmd.as_deref() {
            Some("shutdown") => {
                emit(&json!({"id": req.id, "ok": true, "event": "shutdown"}))?;
                break;
            }
            Some("ping") => {
                emit(&json!({"id": req.id, "ok": true, "event": "pong"}))?;
                continue;
            }
            Some(other) => {
                emit(&json!({
                    "id": req.id,
                    "ok": false,
                    "error": format!("unknown worker command: {other}"),
                }))?;
                continue;
            }
            None => {}
        }

        let t0 = Instant::now();
        let outcome = build_args(&a, &req.fields).and_then(|args| {
            let p = crate::synth::plan(&args)?;
            engine.synthesize(&args, &p)?;
            Ok(args.output)
        });
        let elapsed_ms = t0.elapsed().as_millis() as u64;
        let response = match outcome {
            Ok(output) => json!({
                "id": req.id,
                "ok": true,
                "output": output,
                "elapsed_ms": elapsed_ms,
            }),
            Err(e) => json!({
                "id": req.id,
                "ok": false,
                "error": format!("{e:#}"),
                "elapsed_ms": elapsed_ms,
            }),
        };
        emit(&response)?;
    }

    eprintln!("worker: exiting");
    Ok(())
}
