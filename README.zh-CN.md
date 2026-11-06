# Ming-3.0-tts wgpu

**用 Rust + wgpu 跑 Ming-3.0-tts 语音合成。**

[![GitHub stars](https://img.shields.io/github/stars/eclipse005/ming-3.0-tts-wgpu?style=flat-square&logo=github&color=f59e0b)](https://github.com/eclipse005/ming-3.0-tts-wgpu/stargazers) [![GitHub forks](https://img.shields.io/github/forks/eclipse005/ming-3.0-tts-wgpu?style=flat-square&logo=github)](https://github.com/eclipse005/ming-3.0-tts-wgpu/network/members) [![License](https://img.shields.io/github/license/eclipse005/ming-3.0-tts-wgpu?style=flat-square&color=16a34a)](LICENSE) [![Rust](https://img.shields.io/badge/Rust-2021-CE422B?style=flat-square&logo=rust&logoColor=white)](#安装) [![wgpu](https://img.shields.io/badge/wgpu-30-0F7B6C?style=flat-square)](https://github.com/gfx-rs/wgpu)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square)](#安装) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#安装) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#安装)

[English](README.md) · **简体中文**

[Ming-3.0-tts-0.5b](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5b) 的轻量级跨平台
Rust 实现，使用 [wgpu](https://github.com/gfx-rs/wgpu) 做 GPU 加速。

目标很简单：让 Ming-3.0-tts **在本地原生运行** —— 运行时不需要 Python、不需要 PyTorch，
也不需要 CUDA toolkit。

### 特性

* 🦀 纯 Rust —— 一个二进制，不需要额外环境
* 🎮 wgpu GPU 加速（Vulkan / Metal / GL）
* 🖥️ Windows / macOS / Linux
* ⚡ 手写 CPU 后端：不用任何 NN 框架，只有 rayon 和一个权重复用 GEMM
* 📦 离线本地推理
* 🔁 常驻 worker：2.9 GB 权重只加载一次，之后按行喂 JSON 请求
* 🗣️ 0.5b 模型的全部 task 模式 —— `TTS`、`SEED`、`PLAIN`、`MULTILINGUAL`、`BASIC`、
  `EMOTION`、`DIALECT`、`IP`、`VOICE_DESIGN`、`STYLE`、`FREEFORM`
* 🎙️ 声音克隆：给一段参考录音，或者给预计算好的 `.pt` 说话人向量
* ✂️ 按句切分长文本，并复刻参考实现的 STYLE 首片重条件
* 🔤 中英文本规范化：参考实现那套 pynini FST 链 + 正则规则层 + inflect 风格英文数字措辞
* 🧪 自带一套冒烟测试，以及下面那张逐阶段误差实测表

### 安装

```bash
git clone https://github.com/eclipse005/ming-3.0-tts-wgpu.git
cd ming-3.0-tts-wgpu
cargo build --release        # target/release/ming-tts-wgpu
```

需要较新的 stable Rust（edition 2021）和一个支持 Vulkan / Metal / GL 的显卡驱动。
**编译**不需要 CUDA toolkit、Python 环境或模型文件。

| 依赖 | 什么时候需要 |
|---|---|
| `onnxruntime` 动态库 | 只有"给 `--prompt_wav_path` 并且 `--use_spk_emb`"才需要（CAMPlus 说话人编码器）。`--spk_emb_path` 和其余功能都不需要。用 `ORT_DYLIB_PATH`（或 `MING_ONNXRUNTIME_DIR`）指向它。 |
| `tn_data/` | 已随仓库提供：文本规范化用的 pynini 编译自动机。`MING_TN_DIR` 可以换路径，`--no_text_normalization` 可以跳过。 |

### 模型下载

权重**不在**本仓库里。下载 0.5b checkpoint（版权归原作者）后把目录原样传给
`--model_path`：

* [inclusionAI/Ming-3.0-tts-0.5b](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5b)（Hugging Face）
* [inclusionAI/Ming-3.0-tts-0.5b](https://modelscope.cn/models/inclusionAI/Ming-3.0-tts-0.5b)（ModelScope）

目录里需要有 `model.safetensors`、`tokenizer.json`、`config.json`、`campplus.onnx`
以及 `audio/` 下的 audio tokenizer 张量。

### 快速开始

```bash
ming-tts-wgpu synth \
  --model_path ./Ming-3.0-tts-0.5b \
  --task TTS \
  --text "今天天气真不错，我们一起去公园散步吧。" \
  --output ./hello.wav
```

运行时会打印实际用的适配器、分片数、每片的解码 / VAE 耗时和停止步，最后写一个
44.1 kHz 单声道 WAV。

克隆一个音色：

```bash
ming-tts-wgpu synth --model_path ./Ming-3.0-tts-0.5b --task TTS \
  --text "今天天气真不错。" --output ./cloned.wav \
  --prompt_wav_path ./reference.wav \
  --prompt_text "诶，你今天有空吗？我们能不能聊聊这件事啊？"
```

| 参数 | 说明 |
|---|---|
| `--model_path <dir>` | 模型目录（必填） |
| `--task <name>` | `TTS`、`SEED`、`PLAIN`、`MULTILINGUAL`、`BASIC`、`EMOTION`、`DIALECT`、`IP`、`VOICE_DESIGN`、`STYLE`、`FREEFORM` |
| `--text <str>` / `--output <file>` | 要合成的文本和写出的 WAV |
| `--device <name>` | `cuda:0`（独显，默认）、`igpu`、`cpu`（手写后端，不走 wgpu）；`devices` 子命令列出适配器 |
| `--seed <n>` | 采样种子，默认 1895（与参考实现同 seed） |
| `--language_id <code>` | `zh`、`en`、`ja`、`ko`、`de`、`fr`、`es` —— 非中英会跳过中英规范化 |
| `--instruction <str>` | 自然语言风格指令（`FREEFORM`） |
| `--speed/--pitch/--volume/--emotion/--dialect/--style/--ip` | 各种控制（`BASIC`、`EMOTION`、`DIALECT`、`IP`、`STYLE`） |
| `--prompt_wav_path <wav>` / `--prompt_text <str>` | 克隆用的声学提示 |
| `--spk_emb_path <pt>` / `--use_spk_emb` / `--use_zero_spk_emb` | 预计算说话人向量 / 从提示音算 CAMPlus / 零说话人槽位 |
| `--max_length <n>` | 分片阈值（字符数），默认 24 |
| `--max_decode_steps <n>` | 自回归循环的安全上限，默认 1000 |
| `--cfg <f>` / `--sigma <f>` / `--temperature <f>` | 采样参数，默认 2.0 / 0.25 / 0.0 |
| `--no_text_normalization` / `--raw_text` / `--no_trim_silence` | 关掉规范化 / 不切分 / 不裁尾静音 |

参数面和上游 `infer.py` 完全一致，连 snake_case 拼写都一样。

### Worker

`synth` 每合成一句都要付一次 2.9 GB 的加载；`worker` 只付一次，之后按行服务请求，
这才是配音任务或编辑器要的形态：

```bash
ming-tts-wgpu worker --model_path ./Ming-3.0-tts-0.5b --device cuda:0 <<'EOF'
{"id":"1","task":"TTS","text":"今天天气真不错。","output":"./one.wav"}
{"id":"2","task":"TTS","text":"我们一起去公园散步吧。","output":"./two.wav"}
{"cmd":"shutdown"}
EOF
```

请求里可以带 `synth::Args` 的任意字段（snake_case，和命令行参数同名）；
`model_path` 和 `device` 永远取自 worker 自己的命令行，请求改不了已加载的引擎。
响应是一行一个、以 `@@MING@@` 开头，和推理日志混在 stdout 上：

```text
@@MING@@{"event":"ready","pid":4711,"model_path":"./Ming-3.0-tts-0.5b","device":"cuda:0"}
@@MING@@{"id":"1","ok":true,"output":"./one.wav","elapsed_ms":1320}
```

`{"cmd":"ping"}` 回 `{"event":"pong"}`，`{"cmd":"shutdown"}` 结束循环。
单条请求失败只回 `ok:false` 和错误文本，会话继续跑——坏一行只损失一句，不用重新加载模型。

### 库

二进制只是这个 crate 的薄壳；进程内调用可以完全跳过子进程、socket 和序列化协议：

```rust
use ming_tts_wgpu::synth;

let mut args = synth::Args::default();
args.model_path = "./Ming-3.0-tts-0.5b".into();
args.device = "cuda:0".into();
let mut engine = synth::Engine::load(&args, true)?;   // 只加载一次，VAE 编码器常驻

args.task = "TTS".into();
args.text = "今天天气真不错。".into();
args.output = "./hello.wav".into();
let plan = synth::plan(&args)?;                        // 每句都走这一步
engine.synthesize(&args, &plan)?;
```

`Engine` 内部持有 `Rc`，因此**不是 `Send`**：请留在创建它的线程上，用 channel 驱动。

### 和参考实现差多少

每个阶段都和 Python 参考导出的金标对拍过：

| 阶段 | 指标 | 实测 |
|---|---|---|
| LLM prefill + 40 步解码 | hidden-state rel_l2 | 8.1e-7，停止步一致 |
| DiT 单次前向 | rel_l2 | 1.6e-7 |
| VAE 编码器 | mean / std / latent | 1.4e-6 / 6e-8(abs) / 1.4e-6 |
| VAE 解码器 + iSTFT | 波形 | 1.9e-6，长度逐采样点一致 |
| RNG（`torch.randn` 复刻） | 28160 个值 | 最大误差 1.4e-6 |

上游的 36 例 `params_test` 里 **35 例时长完全一致**，log-mel 中位数 1.00000；
26 例波形级重合，其余是后段相位分岔。解码环是混沌的 —— 1e-7 的扰动 40 步会放大
约 1e5 倍 —— 所以有一个韩语用例比参考多收一个 latent。这是实测行为，不是 bug。

同一套对拍还在模型官网的 24 条 demo 请求上重跑过（Voice design、Director mode、
副语言标签、7 种语言 × 2 个内置音色、2 种方言）：**24/24 时长**一致到 10 ms 以内，
17 条相关 ≥0.995，13 条峰值差 <0.02。剩下 7 条是在约 1e-4 的 latent 漂移累积后分岔，
也就是上面说的混沌——而参考实现**自己**换设备的分歧比这大得多：同 seed、同文本，
CPU 与 CUDA 出的长度不同、波形相关只有 0.07。

这轮里还抓到并修掉一个真 bug：文本规范化会把行内标签改写（`[Laughter]` 变成
`，Laughter，`），模型于是去"念"标签而不是笑。现在 `prepare_text` 让 FST 链绕开标签，
标签卡与参考实现逐采样点一致。

Tesla P104 上的语速（3.2 s 音频，仅解码）：**RTF ≈ 0.60**，同文本的 Python + CUDA Graph
是 0.78。这张卡的电源状态本身就能让同一条命令差 2.3×，所以任何单个数字都只算同一窗口内的
对比。

### 测试

```bash
cargo test --release         # 三个冒烟测试，不需要任何模型/数据
```

* `wav_roundtrip` —— 写一段波形再读回来，逐采样点比对
* `text_front_end` —— 文本切分 + 参考实现那套规范化链（数字、金额、日期必须变成中文读法），
  用的是随仓库提供的 `tn_data/`
* `synthesize_short_utterance` —— 一次端到端合成。能找到 checkpoint 时才跑：用 `MING_MODEL`
  指向模型目录（默认找 `../Ming-3.0-tts-0.5b`），找不到就打印 `[skip]`

上面那张表里的数值，是对着 Python 参考导出的金标逐阶段比出来的，那套脚本服务于移植过程
而不是运行时，已经随着调参一起下线：干净 clone 里带的是上面的冒烟测试、随仓库提交的
`tn_data/`，以及这些实测数字。

### 代码结构

```
src/
  main.rs, worker.rs         二进制：`synth`（单句）与 `worker`（JSON 行）
  lib.rs                     库接口：synth::Engine、tasks、worker
  synth.rs                   参数 → 任务 → 文本分片 → 选后端
  backend.rs                 三条后端共用的分片循环
  pipeline.rs                wgpu 后端：解码环、采样噪声
  cpu/                       手写 CPU 后端（LLM、DiT、VAE、GEMM）
  ops.rs, nn.rs, gpu.rs      设备原语、uniform arena、提交批处理
  shaders/                   WGSL 按家族分模块：gemv / gemv_t1 / gemv_t6 / prefill / attention / elementwise
  qwen_lm.rs, dit.rs         Qwen2.5-0.5B 解码器、DiT + flow matching
  vae.rs, vae_dec.rs, istft.rs   audio tokenizer（编码器、解码器、iSTFT）
  text.rs, tn/, tok.rs       文本前端：切分、规范化、分词器
  audio.rs, spk.rs           WAV 读写 + 重采样、CAMPlus 说话人编码器
  weights.rs, pt.rs, golden.rs   safetensors、torch `.pt`、`.npy` 读取
```

### 为什么用 wgpu

不依赖 CUDA 或任何厂商专用运行时，用 **wgpu** 做 GPU 抽象，所以同一个 Rust 二进制能在
NVIDIA、AMD、Intel、Apple 和软件适配器上跑。某些 kernel 在独显和核显上受限于不同的东西，
移植按适配器类别选调参；`--device cpu` 则走同一套 pipeline 的手写主机实现，而不是换一个模型。

DX12 是**有意**编译掉的：wgpu 30 默认的 `dx12` 后端会拉 `windows = 0.62`，和
`gpu-allocator` 在固定 `windows 0.61` 的宿主里编不过。宿主需要就自己把 feature 加回来，
推理结果不受影响。

### 项目状态

对照 `infer.py` 功能已经完整：参数面、全部 task 模式、分片、克隆、零说话人、STYLE
首片重条件、文本规范化和常驻 worker。流式没有做，因为 `infer.py` 本身也没暴露。

### 相关

* [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) —— 官方模型、参考实现和 task cookbook
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) —— 兄弟语音移植项目，本项目的 CPU 后端和 kernel 思路来自它

### 许可证

MIT，与上游一致。

本仓库是加载并运行官方发布的 Ming-3.0-tts-0.5b 权重的**独立 Rust 推理实现**，
不是 inclusionAI 的官方发布，也与原作者无隶属关系。模型权重和编译好的文本规范化自动机
仍归各自所有者，遵循其原始条款。
