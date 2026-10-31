# Ming-3.0-tts → wgpu 移植进度看板

目标：把手写 Python/PyTorch 的 Ming-3.0-tts-0.5b 推理移植为纯 Rust + wgpu（无
Python/torch/CUDA 依赖），全程用逐层 parity 对拍保真。分析见
[WGPU_PORT_ANALYSIS.md](WGPU_PORT_ANALYSIS.md)。

## 验收锚点（用户指定）

1. **初步对齐**：与 `D:\Ming-3.0-tts\output\` 下现有音频对齐（clone_demo / ab_graph 等）
2. **最终验收**：与 `D:\Ming-3.0-tts\output\params_test\` 36 个全参数样本全面对齐
3. **API 一致**：CLI 参数面与 `infer.py` 完全一致
   （--model_path/--task/--text/--device/--seed/--language_id/--instruction/
   --speed/--pitch/--volume/--emotion/--dialect/--style/--ip/--cfg/--sigma/
   --temperature/--max_length/--prompt_wav_path/--prompt_text/--use_spk_emb/
   --use_zero_spk_emb/--output）

## 里程碑

| 阶段 | 内容 | parity 断言 | 状态 |
|---|---|---|---|
| M0 | 骨架：gpu 初始化 + bf16 权重加载 + rmsnorm/gemv 内核 + 单层 parity | rel_l2 < 1e-5 | ✅ 2026-10-31（rmsnorm 1.0e-7 / gemv 3.0e-7，Vulkan 上 P104-100 实测） |
| M1 | LLM：prefill + KV 解码 + stop head | 停止步数强断言（60+ 文本/seed） | ⬜ 下一步 |
| M2 | DiT + Aggregator（固定 seed 噪声注入） | latent rel_l2 < 1e-3 | ⬜ |
| M3 | VAE Encoder（prompt → latent） | 与 Python 导出 latent 对拍 | ⬜ |
| M4 | VAE Decoder + iSTFT 全链路 | 频谱 L2 + 时长 + 试听 | ⬜ |
| M5 | 性能调优 + 全参数验收 | params_test/ 36 样本对齐 | ⬜ |

## M0 记录（2026-10-31）

- `cargo build`/`test` 全绿；`ming-tts-wgpu.exe devices` 列出 [Vulkan] P104-100。
- golden dump：真实 clone-demo 输入（lingguang + A3 文本），T=109，h3 激活含 7.63e3 尖峰。
- 内核：`RMSNORM`（256 lane 行归约）、`GEMV_BF16W`（warp/行 + bf16 解包 f32 累加 + bias）。
- **教训**：① 本 checkpoint 的 q/k/v 投影带 bias（absmax ~17），gemv 必须融合 bias；
  ② checkpoint key 有 `audio.encoder/decoder` 前缀的同名层，匹配要用 `model.model.layers.*` 全前缀；
  ③ bf16 1.0 LE = `[0x80,0x3F]`（测试数据写错过一次）。

## 决策记录

- 权重：**bf16 原样存储 + shader 内解包 f32 计算**（bf16⊂f32 无损；总线减半）；
  不走 ASR 项目的 f16 权重路线。
- 激活：**全 f32**（LLM 残差流平台 7.44e4 超 f16 上限；indextts f16-CFM 漂移 9e-2 教训）。
- iSTFT：CPU rustfft（12.5 帧/s，不值得写 GPU FFT）。
- CAMPlus 说话人：ort CPU ONNX，不移植。
- 独立 git 仓库（本目录），与 Python 主仓分离，过程按里程碑提交。

## 复用来源

- `D:\qwen3-asr-wgpu`：wgpu 30 设备/BulkUpload/weights/safetensors、Qwen 解码器、
  WGSL 内核族（改 f32）、tokenizers 集成。
- `D:\index-tts\indextts-rs`：golden npy + rel_l2 parity 方法论、CPU oracle twin、
  f32 CFM 精度教训、预分配工作区模式。
