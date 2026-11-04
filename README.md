# Ming-3.0-tts wgpu

**Ming-3.0-tts speech synthesis in Rust with wgpu.**

[![License](https://img.shields.io/github/license/eclipse005/ming-3.0-tts-wgpu?style=flat-square&color=16a34a)](LICENSE) [![Rust](https://img.shields.io/badge/Rust-2021-CE422B?style=flat-square&logo=rust&logoColor=white)](#install) [![wgpu](https://img.shields.io/badge/wgpu-30-0F7B6C?style=flat-square)](https://github.com/gfx-rs/wgpu)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square)]( #install ) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#install) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#install)

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of
[Ming-3.0-tts-0.5b](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5b), using
[wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration.

The goal is simple: run Ming-3.0-tts **locally and natively** — no Python, no
PyTorch, no CUDA toolkit at runtime.

### Features

* 🦀 Pure Rust — one binary, nothing else to install
* 🎮 GPU acceleration through wgpu (Vulkan / Metal / DX12 / GL)
* 🖥️ Windows / macOS / Linux
* ⚡ Hand-written CPU backend: no NN framework, just rayon and a weight-reuse GEMM
* 📦 Offline local inference
* 🗣️ Every task mode of the 0.5b model — `TTS`, `SEED`, `PLAIN`, `MULTILINGUAL`,
  `BASIC`, `EMOTION`, `DIALECT`, `IP`, `VOICE_DESIGN`, `STYLE`, `FREEFORM`
* 🎙️ Voice cloning from a prompt recording, or from a precomputed `.pt` speaker embedding
* ✂️ Sentence-level text splitting, with the STYLE re-conditioning the reference does
* 🔤 Chinese/English text normalisation: the reference's pynini FST chain, its
  regex rule layer, and inflect-style English number wording
* 🧪 Small smoke suite, plus the measured per-stage parity table below

### Install

```bash
git clone https://github.com/eclipse005/ming-3.0-tts-wgpu.git
cd ming-3.0-tts-wgpu
cargo build --release        # target/release/ming-tts-wgpu
```

Requires a recent stable Rust (edition 2021) and a GPU driver exposing Vulkan,
Metal or DX12. Building does **not** need a CUDA toolkit, a Python environment or
the model itself.

| Dependency | When |
|---|---|
| `onnxruntime` shared library | Only for `--use_spk_emb` with a prompt recording (the CAMPlus speaker encoder). `--spk_emb_path` and everything else need nothing extra. Point `ORT_DYLIB_PATH` (or `MING_ONNXRUNTIME_DIR`) at it. |
| `tn_data/` | Shipped in this repository: the compiled pynini automata the text normaliser uses. `MING_TN_DIR` overrides the location, `--no_text_normalization` skips it. |

### Model download

Weights are **not** included in this repository. Download the 0.5b checkpoint
(rights remain with the original authors) and pass the directory to
`--model_path` unchanged:

* [inclusionAI/Ming-3.0-tts-0.5b](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5b) (Hugging Face)
* [inclusionAI/Ming-3.0-tts-0.5b](https://modelscope.cn/models/inclusionAI/Ming-3.0-tts-0.5b) (ModelScope)

The directory must contain `model.safetensors`, `tokenizer.json`,
`config.json`, `campplus.onnx` and the `audio/` tensors used by the tokenizer.

### Quick start

```bash
ming-tts-wgpu synth \
  --model_path ./Ming-3.0-tts-0.5b \
  --task TTS \
  --text "今天天气真不错，我们一起去公园散步吧。" \
  --output ./hello.wav
```

The run prints the adapter in use, the fragment count, per-fragment decode/VAE
timings and the stop step, then writes a 44.1 kHz mono WAV.

Clone a voice instead of using the built-in one:

```bash
ming-tts-wgpu synth --model_path ./Ming-3.0-tts-0.5b --task TTS \
  --text "今天天气真不错。" --output ./cloned.wav \
  --prompt_wav_path ./reference.wav \
  --prompt_text "诶，你今天有空吗？我们能不能聊聊这件事啊？"
```

| Option | Description |
|---|---|
| `--model_path <dir>` | Model directory (required) |
| `--task <name>` | `TTS`, `SEED`, `PLAIN`, `MULTILINGUAL`, `BASIC`, `EMOTION`, `DIALECT`, `IP`, `VOICE_DESIGN`, `STYLE`, `FREEFORM` |
| `--text <str>` / `--output <file>` | Text to speak and the WAV to write |
| `--device <name>` | `cuda:0` (discrete, default), `igpu`, `cpu` (hand-written backend, no wgpu). `devices` lists the adapters |
| `--seed <n>` | Sampler seed, default 1895 (same seed as the reference) |
| `--language_id <code>` | `zh`, `en`, `ja`, `ko`, `de`, `fr`, `es` — bypasses zh/en normalisation for the others |
| `--instruction <str>` | Free-form delivery instruction (`FREEFORM`) |
| `--speed/--pitch/--volume/--emotion/--dialect/--style/--ip` | Task controls (`BASIC`, `EMOTION`, `DIALECT`, `IP`, `STYLE`) |
| `--prompt_wav_path <wav>` / `--prompt_text <str>` | Acoustic prompt for voice cloning |
| `--spk_emb_path <pt>` / `--use_spk_emb` / `--use_zero_spk_emb` | Precomputed speaker vectors, CAMPlus embedding from the prompt, or a zeroed speaker slot |
| `--max_length <n>` | Fragmentation threshold in characters, default 24 |
| `--max_decode_steps <n>` | Safety cap on the autoregressive loop, default 1000 |
| `--cfg <f>` / `--sigma <f>` / `--temperature <f>` | Sampler knobs, defaults 2.0 / 0.25 / 0.0 |
| `--no_text_normalization` / `--raw_text` / `--no_trim_silence` | Disable normalisation, don't split the text, don't trim the tail |

The flag surface is the same as `infer.py` in the upstream repository, down to
the snake_case spelling.

### How close is it to the reference?

Every stage is compared against goldens dumped from the Python implementation:

| Stage | Measure | Result |
|---|---|---|
| LLM prefill + 40 decode steps | hidden-state rel_l2 | 8.1e-7, same stop step |
| DiT single forward | rel_l2 | 1.6e-7 |
| VAE encoder | mean / std / latent | 1.4e-6 / 6e-8 abs / 1.4e-6 |
| VAE decoder + iSTFT | waveform | 1.9e-6, sample-exact length |
| RNG (`torch.randn` clone) | 28160 values | max error 1.4e-6 |

The 36-case `params_test` suite from upstream reproduces **matching durations on
35 of 36 cases** and a median log-mel correlation of 1.00000; 26 cases are
sample-level close, the rest drift in the late phase. Decoding is chaotic —
a 1e-7 perturbation grows about 1e5× over 40 steps — so one Korean case ends one
latent later than the reference. That is measured behaviour, not a bug.

Speech rate on a Tesla P104 (3.2 s of audio, decode only): **RTF ≈ 0.60**,
against 0.78 for the Python/CUDA-graph build on the same text. The card's power
state alone changes the same command by up to 2.3×, so treat any single number
as a same-window comparison.

### Tests

```bash
cargo test --release         # three smoke tests, no assets needed
```

* `wav_roundtrip` — write a waveform, read it back, compare samples
* `text_front_end` — splitting and the reference's normalisation chain over the
  shipped `tn_data/` automata (digits, money, dates must come out as spoken Chinese)
* `synthesize_short_utterance` — one end-to-end utterance. It runs when a
  checkpoint is reachable; point `MING_MODEL` at the downloaded directory (the
  default is `../Ming-3.0-tts-0.5b`), otherwise it prints `[skip]`.

The numerical comparison behind the table above was done against goldens dumped
from the Python reference. That data is not part of a source checkout, so it is
not shipped; `tools/` keeps the harness for re-running it against a machine with
the reference available.

### Verification tooling

| Script | Purpose | Needs |
|---|---|---|
| `tools/check_params_align.py` | Runs the 36 upstream cases and prints duration ratio / rel_l2 / log-mel / speaker cosine | the reference's rendered WAVs (`output/params_test/`) |
| `tools/_goldens_align.py` | Same comparison from recorded noise draws, no Python needed | a `goldens/` directory |
| `tools/stress_compare.py` | 10-case stress set: long text, multi-fragment, tiny inputs, mixed script, `--raw_text` | a Python reference checkout |
| `tools/dump_params_goldens.py` | Records the reference's `torch.randn` draws and audio for the 36 cases | a Python reference checkout |
| `tools/ref_nocache.py` | Re-renders the cases whose reference audio reused a cached prompt | a Python reference checkout |
| `tools/compare_wav.py` | Duration / level / correlation between two WAVs | — |

### Layout

```
src/
  main.rs, synth.rs          CLI: flags -> task -> text fragments -> backend
  backend.rs                 the fragment loop shared by every backend
  pipeline.rs                wgpu backend: decode loop, sampler noise
  cpu/                       hand-written CPU backend (LLM, DiT, VAE, GEMM)
  ops.rs, nn.rs, gpu.rs      device primitives, uniform arena, submission batching
  shaders/                   WGSL by family: gemv, gemv_t1, gemv_t6, prefill, attention, elementwise
  qwen_lm.rs, dit.rs         Qwen2.5-0.5B decoder, DiT + flow matching
  vae.rs, vae_dec.rs, istft.rs   audio tokenizer (encoder, decoder, iSTFT)
  text.rs, tn/, tok.rs       text frontend: splitting, normalisation, tokenizer
  audio.rs, spk.rs           WAV I/O + resampling, CAMPlus speaker encoder
  weights.rs, pt.rs, golden.rs   safetensors, torch `.pt`, `.npy` readers
```

### Why wgpu?

Instead of depending on CUDA or another vendor-specific runtime, this port uses
**wgpu** as the GPU abstraction, so one Rust binary runs on NVIDIA, AMD, Intel,
Apple and software adapters. Where a kernel is bound by different things on
discrete and integrated GPUs, the port picks the tuning per adapter class — and
on `--device cpu` it falls back to a hand-written host implementation of the same
pipeline, not to a different model.

### Project status

The port is functionally complete against `infer.py`: same flags, all task modes,
fragmentation, cloning, zero-speaker, STYLE re-conditioning and text
normalisation. Streaming is not implemented because `infer.py` does not expose
it either.

### Related

* [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) — the official model, reference implementation and task cookbook
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) — the sibling speech port this one borrows its CPU-backend and kernel patterns from

### License

MIT, matching the upstream project.

This repository is an **independent Rust inference implementation** for loading
and running the officially released Ming-3.0-tts-0.5b weights — not an official
inclusionAI release, and not affiliated with the original authors. Model weights
and the compiled text-normalisation automata remain under the terms of their
respective owners.
