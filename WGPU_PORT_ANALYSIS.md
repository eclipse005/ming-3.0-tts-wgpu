# Ming-3.0-tts → wgpu 移植分析

基于对两个前作的侦察（`D:\index-tts\indextts-rs` 27.6k 行 Rust+CUDA、
`D:\qwen3-asr-wgpu` 33k 行 Rust+WGSL）和 Ming-3.0-tts 全部模型代码/config 的核对。
日期：2026-10-31。

## 0. 结论

**可行，且比预期简单。** 全模型 1.48B 参数中约 1.26B 是**同一几何**的
Qwen-style Transformer（hidden 896 / 24 层 / GQA 14+2 / intermediate 4864）——
LLM、VAE Encoder、VAE Decoder 三者几何完全相同，一套内核族吃掉 85% 的参数。
DiT 头与 Aggregator 是同模式的 8 层小 Transformer（hidden 1024）。
没有 BigVGAN 式卷积栈、没有 GPU 端 FFT 的硬需求、没有动态 shape 的采样器。

核心价值排序：**无依赖部署**（单 exe、Vulkan/DX12、跨厂商 GPU）> 性能。
预期 RTF 0.6~1.5（Pascal/Vulkan，f32 激活 + bf16 权重总线），
与现有 Python+CUDA 路径（0.88）相当或更好。

## 1. 模型解剖（1.48B 参数，44.1kHz 输出）

| 模块 | 参数 | 结构 | 注意力 | 备注 |
|---|---|---|---|---|
| LLM (Qwen2.5-0.5B) | 494M | 24L×896, GQA 14/2, FF 4864, rope_theta 1e6, tied emb, vocab 151936 | causal + KV cache | 解码预算 512 token 静态上限（MAX_CACHE_LEN） |
| VAE Encoder | ~390M | patch 前端（hop 882=50Hz 帧→patch4=12.5Hz）+ cls + **24L×896 同几何 Transformer** + LayerNorm + 对角高斯(latent 64) | bidirectional 全序列 | 只在 prompt 编码时跑一次 |
| VAE Decoder | ~390M | fc1 → **线性上采样×4**（nn.Upsample linear，非卷积！）→ **24L×896 同几何 Transformer** → head → iSTFT | bidirectional | 0.5b 无 semantic(whisper) 模块（config 为 null） |
| DiT (flowloss) | 103M | 8 blocks ×1024（16 头）, pre-norm 残差块, RoPE, GELU-tanh FF×4, FinalLayer | bidirectional 3-token | 每 decode 步 10 次 Euler × batch-2 CFG |
| Aggregator | 102M | 同 DiTBlock 结构 | bidirectional | 每 decode 步 1 次 |
| stop/spk head | <1M | 单 Linear | — | stop 判定每步一次 |
| CAMPlus 说话人 | 27MB | ONNX | — | **不移植**，ort CPU 跑（与 Python 参考一致） |

iSTFT 参数（`audio_tokenizer/istft.py`，自定义 "same" padding 实现，非 torch.istft）：
n_fft 2048（input_dim 882 = n_fft/2+1）、hop 882、hann 窗。实现时与 istft.py 参数逐项核对。

**关键红利**：三种 896 Transformer 几何相同（含 GQA 头数），
`qwen3-asr-wgpu` 的 Qwen3 解码器改造后一份实现服务三处；
1024 的 DiT/Agg 是同模式的形状参数化。

## 2. 复用地图

### 从 qwen3-asr-wgpu（wgpu 基建，几乎全复用）
- `gpu.rs`：wgpu 30.0.1、设备选择（Vulkan 优先）、adapter 全 limits、BulkUpload
  （256MiB 预算分批上传）、readback、pipeline 缓存 → **原样复制**。Pascal/Vulkan 已验证可跑。
- `weights.rs`：safetensors 0.4 + memmap2 零拷贝、分片 index、HF 名归一化 → 复制；
  **改 dtype 策略**（见 §3）：保留 bf16，不要像 ASR 那样转 f16。
- `shaders.rs`（Rust 字符串生成器模式，非 .wgsl 文件）：gemv、tiled GEMM(16×16 WG)、
  rms_norm、softmax_causal_tree、gqa_decode(+split-merge)、silu_mul、argmax、embed_lookup、
  im2col → **九成是"去 f16 打包改 f32/解包 bf16"的机械改写**。
- `decoder.rs`：Qwen 家族 KV cache（按 256 步增长、f16 packed）、prefill+增量解码、
  融合 q/kv 与 gate/up、MRoPE → 改 f32 激活 + rope_theta 1e6 + 本模型 config。
- `expf_bt`（位精确 exp）、`tokenizers` crate 集成、贪心循环、bench 脚本、
  probe 二进制惯例 → 原样方法论。
- ASR 项目不存在、需要新写的：conv1d/convT（本项目也不需要了！）、GPU FFT（本项目走 CPU）。

### 从 indextts-rs（方法论 + 精度教训）
- **CPU oracle twin**：每个 GPU 模块配纯 Rust CPU 路径（gemm crate + rayon），容错回退。
- **golden npy 逐阶段对拍** + `rel_l2` 门限 + informational/gating 分级 + 无 fixture 时优雅跳过。
- **f32 CFM 教训**（s2mel_cuda.rs:2-11）：他们实测 f16 激活经 25 步 Euler + CFG 放大，
  漂移 9e-2 超过 1e-2 门限 → Ming 直接全 f32 激活，且本模型激活峰值 7.44e4 本就超出
  f16 表示范围（见 §3）。
- 预分配工作区零热路径分配、`alloc_uninit`、pinned 仅用于每步 1 token 的 D2H、
  CFG 合并 batch=2、权重加载期融合 qkv/gate_up、`compute_tag()` 诚实报告、
  thiserror 错误枚举、`--test-threads=1`（显存隔离）。

## 3. 精度策略（本移植最重要的决策）

1. **权重存 bf16，shader 内解包 f32 计算**：checkpoint 本身是 bf16（2.96GB）。
   bf16→f32 是纯移位（bf16 ⊂ f32，**无损**），GEMV/GEMM 内解包 + f32 累加，
   数学上与 Python 的 fp32 权重路径严格等价。收益：
   - 显存：核心权重 ~2.8GiB（Python fp32 是 5.5GiB）→ 全模型 ~4GiB；
   - **总线带宽减半**：Pascal 解码是带宽瓶颈（indextts 实测 GEMV 243-371GB/s，
     asr 实测 Vulkan f32 194-283GB/s），LLM 每步 19ms 预计降到 ~10-13ms。
2. **激活全 f32**：两个硬理由——
   - LLM 残差流从 layer 3 起平台在 **7.44e4**，超出 f16 上限 65504（本会话已实测）；
   - indextts 的 Euler+CFG 漂移实测否决了 f16 激活。
   Pascal 无 f16 算力（1/64 速率），f32 激活在本卡上**零代价**。
3. **KV cache f32**（24L×2×512×896×2×4B ≈ 88MB，无所谓）。
4. **停止头数值敏感**是本项目特有风险：Python 侧已实测两种数值扰动
   （transformers 4.57、CPU 偶发）导致 stop 永不触发 → 跑满 512 预算。
   Rust 侧必须：`max_decode_steps` 硬上限 + 生成时长 sanity 检查 + 失败自动
   reseed 重试；**parity 测试把"停止步数"作为强断言**。

## 4. 新增/改写内核清单（WGSL，均为小工作量）

| 内核 | 来源 | 说明 |
|---|---|---|
| gemv_bf16w_f32 | asr gemv 改 | 解包 bf16→f32 fma，f32 激活 I/O，LLM 解码主路径 |
| gemm_f32 tiled | asr prefill_gemm 改 | prefill + VAE/DiT/Agg 全序列 GEMM |
| rms_norm_f32 / rope_f32(1e6) / softmax_f32 / gqa_decode_f32 | asr 对应改 | 去 f16 打包 |
| silu_mul_f32 / gelu_tanh_f32 / add_residual | asr 改 | FF 与残差 |
| xformer_block 调度（bidirectional, no-KV） | 复用上面 | VAE enc/dec + DiT + Agg 四处共用 |
| diag_gaussian_sample | 新 | 逐元素，仅 enc 一次 |
| linear_upsample_x4 | 新/CPU | 逐元素插值，量小 |
| iSTFT | **CPU rustfft** | 12.5 帧/s × 时长，一次几 MB 下载；不值得写 GPU FFT；注意自定义 "same" padding 语义（istft.py 逐行对齐 + 单测） |
| campplus | ort CPU ONNX | 不移植；或 pipeline 预计算 spk_emb 文件 |

控制流全部在 Rust host：10 步 Euler + sway/epss 时间步（纯标量表）、CFG 已批 B=2、
stop 判定、latent_history 滑窗（每步 1 行 copy）、512 步预算——无 GPU 端动态性，
无需 CUDA Graph 的等价物。

## 5. 模块规划（`D:\Ming-3.0-tts\wgpu\` crate）

```
wgpu/
  Cargo.toml        # wgpu 30, pollster, safetensors, memmap2, tokenizers, rustfft, half, thiserror, anyhow, ort
  src/
    gpu.rs          # ← asr 原样
    weights.rs      # ← asr 改 bf16 保留
    shaders.rs      # f32 内核族（§4）
    qwen_lm.rs      # ← asr decoder.rs 改：f32 激活、causal KV≤512、rope 1e6、tied head、stop head
    xfmer.rs        # bidirectional f32 Transformer（VAE enc/dec、DiT、Agg 共用）
    dit.rs          # Euler 采样循环 + CFG + sway/epss + latent_history + stop 判定
    vae.rs          # encode(prompt→latent) / decode(latent→spectrogram)
    istft.rs        # CPU rustfft，hann + same-padding 语义
    frontend.rs     # tokenizers + cut_text_by_semantic_length 移植（无 TN）
    pipeline.rs     # prompt 契约、组装、trim silence（ming_tts.py 同款阈值）、44.1k 输出
    parity/         # golden npy 读取 + rel_l2 + 每阶段 dump 工具
  bench/goldens/    # Python 侧导出的逐阶段张量（沿 indextts 布局）
```

## 6. 里程碑与 parity 门限

| 阶段 | 内容 | parity 断言 |
|---|---|---|
| M0 | 骨架：gpu/weights 加载 + 单层前向 | 单层输出 rel_l2 < 1e-5（f32 vs f32） |
| M1 | LLM prefill + KV 解码 + stop head | hidden_states 序列对齐；**停止步数强断言**（60+ 文本/seed 组合） |
| M2 | DiT + Aggregator（Python 侧固定 seed 注入噪声） | sampled latent rel_l2 < 1e-3；端到端音频频谱 L2 |
| M3 | VAE Encoder | 用 Python 导出的 latent 直接喂 Rust decoder 隔离验证 |
| M4 | VAE Decoder + iSTFT 全链路 | 44.1k 波形频谱 L2 + 时长一致 + 试听 |
| M5 | 性能调优 | bf16 GEMV、dispatch 合并、缓冲池；对齐 Python 0.88 |

Python 侧需加一个 `--dump` 钩子逐阶段导出 npy（沿 indextts 的 golden 布局）。

## 7. 风险清单

1. **停止头数值敏感**（已实锤两例）→ 硬上限 + sanity + reseed（§3.4）。
2. **Vulkan dispatch 开销**：每句 ~60 步 ×（LLM ~22 + DiT 10×8 块 + Agg 8 块）≈ 数千次
   dispatch。asr 实测 227 dispatch/步 @8.5ms 可行，但 DiT 段块多核小 → 照搬 indextts
   的融合（fused qkv、fused residual+norm）压 dispatch 数。
3. **Vulkan f32 GEMV 带宽**（194-283 GB/s）→ bf16 权重总线是决定性优化，M1 就要做。
4. **iSTFT "same" padding 语义**与 torch.istft 不同（NOLA 规避），逐行移植 + 对拍。
5. **对角高斯采样的 RNG**：与 Python 非位等 → parity 用 mean 或导出 latent；生产用
   自有 seeded RNG（统计等价；本会话实测不同 seed 下模型行为健康，偶发失控靠 §3.4 缓解）。
6. **文本切分移植**（cut_text_by_semantic_length）影响 fragment 数与韵律，需逐例对拍。
7. 中文 IP 音色名等 prompt 契约字符串要逐字节复刻（tts_tasks.py 是协议）。

## 8. 工作量与参照

- qwen3-asr-wgpu 33k 行；本项目 transformer 内核可大量复制，新增主要是 DiT/VAE 编排
  与 frontend，估 **20-30k 行**。
- 性能参照：asr 在同卡 Vulkan 上 RTFx 22-29（Python CUDA 4-8，约 3.5 倍差）——但 ASR
  是 f16 激活被迫走标量路径；本项目 f32 主路径 + bf16 权重总线在 Pascal 上结构更优，
  综合 RTF 估计 0.6~1.5。
