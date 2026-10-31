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
| M1 | LLM：prefill + KV 解码 + stop head | 停止步数强断言（60+ 文本/seed） | ✅ 2026-10-31（真实序列 40 步重放：z rel_l2 8.1e-7、prob 差 3.9e-6、stop_step=40 与 Python 一致；多文本/seed 批量断言留 M5 验收一并跑） |
| M2 | DiT + Aggregator（固定 seed 噪声注入） | latent rel_l2 < 2e-3 | ✅ 2026-10-31（41 步独立对拍 worst 1.12e-3；首链 y 轨迹 3e-4；单次 dit_forward 1.26e-4） |
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

## M1 记录（2026-10-31）

- 新内核：GEMV_MROW（prefill 多 token 线性层）、ROPE（rotate-half，宿主 f64 表 θ=1e6）、
  ATTEND（每 (head,token) 一个 workgroup 的因果 GQA，直读 f32 KV cache）、SILU_MUL、
  ADD、KV_WRITE。
- 引擎：24 层、bf16 权重常驻显存、f32 KV cache（每层 2×512×64）。
- 踩坑：Pascal wgpu 默认 workgroup 上限 256（ROPE 512→循环式）；每个内核的
  storage-ro 数量必须逐一对准（ATTEND=3/ROPE=2/GEMV_MROW=3）；ATTEND Params 16B。
- Python 侧 dump 注意：decode 步输入走 `_project_audio_latent`（CUDA Graph replay
  绕过模块 forward），spy 必须挂在这里而不是 linear_proj_audio.forward。
- 下一步 M2：DiT/Aggregator（1024 宽小 transformer，同内核族参数化）+ flowloss
  Euler 采样 + 固定 seed 噪声注入对拍。

## M2 记录（2026-10-31）

- 新内核：PACK3（序列拼装）、GELU_ERF（A&S 7.1.26 近似 erf）；KV_WRITE 泛化
  （cap/n_kv 参数，支持 16 头无 GQA 的 DiT 注意力）；ATTEND 增加 full/qpos_max/stride
  （双向注意力 + 1024 宽行距 —— 行距曾是硬编码 896，DiT 全错）。
- dit.rs：TimestepEmbedder（宿主正弦表 + GEMV+SiLU+GEMV）、CondEmbedder、8+8 块、
  FinalLayer、Aggregator（cls embedding + 2-token 序列）、CFM Euler 采样
  （epss + sway 时间步，temperature=0 时 sde 项为零）。
- **RoPE 约定差异**：x_transformers 是交错对（GPT-J 式），与 Qwen2 的对半分不同 ——
  ROPE 内核加 interleaved 开关。
- **关键教训**：① x_transformers FeedForward 的 Sequential 命名 ff.0.0/ff.2（ff.1 是
  Dropout）；② checkpoint key 用全前缀匹配；③ 测试脚本自身的切片/形状错误
  （z 是 896 维、sde 要 reshape 成 [10,1,1,64]）曾造成大量假阴性。
- **开放问题**：torch 的 (pred, null) 分解随调用上下文（cfm.sample 内 vs 直接
  forward_with_cfg）出现 (δ, 1.5δ) 漂移，而速度 3p−2n 与积分链 y 完全不变 ——
  已在测试中注明，断言只放在不变量（y 链）上。
