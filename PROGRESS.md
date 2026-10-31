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
| M3 | VAE Encoder（prompt → latent） | 与 Python 导出 latent 对拍 | ✅ 2026-10-31（24 层栈 1.9e-6 / aggregator 2.1e-6 / mean 1.4e-6 / std 8.3e-7 / latent 1.4e-6） |
| M4 | VAE Decoder + iSTFT 全链路 | 频谱 L2 + 时长 + 试听 | ✅ 2026-10-31（fc1 1.6e-7 / 上采样 1.5e-7 / 24 层栈 1.5e-6 / head 6.8e-7 / **波形 1.9e-6**，长度 148176 一致） |
| M5 | 性能调优 + 全参数验收 | params_test/ 36 样本对齐 | 🔨 **A1 端到端跑通**（时长逐样本一致、log-mel 相关 0.999991）；36 样本待跑；**RTF 3.08 vs Python 0.88，性能还在收**（见下） |


## M5 记录（2026-10-31，端到端首次跑通）

### 能跑什么

```bat
wgpu\target\release\ming-tts-wgpu.exe synth ^
  --model_path D:/Ming-3.0-tts/Ming-3.0-tts-0.5b ^
  --task TTS --text "今天天气真不错，我们一起去公园散步吧。" ^
  --output D:/Ming-3.0-tts/output/rust/A1_bare.wav --no_text_normalization
```

参数面与 `infer.py` 逐字一致（clap 用 `rename_all = "snake_case"`，否则会变成
`--model-path` 之类）。`MING_NOISE_GOLDEN=<dir>` 可灌入金标 torch.randn 抽样，
用来和 Python 逐样本对拍；`MING_PROFILE=1` 打印分段耗时。

### A1_bare 验收（金标 RNG 注入，与 `output/params_test/A1_bare.wav` 对比）

| 指标 | 结果 |
|---|---|
| 时长 | 3.200 s vs 3.200 s（**采样点数 141120 完全一致**） |
| 波形 rel_l2 | 6.46e-3 |
| 峰值 | 0.5910 vs 0.5916 |
| **log-mel 相关** | **0.999991** |
| log-mel 平均绝对差 | 0.0045 |

也就是说：同一个 seed 下，Rust 端产出的是**同一句话、同一时长、频谱几乎重合**的音频。

### 音色（克隆）链路验收 — CAMPlus 相似度

用 CAMPlus 对这四段音频算 192 维 embedding 后两两余弦：

| A | B | cos |
|---|---|---|
| ref A1（无提示音） | **rust A1** | **+1.0000** |
| ref A3（Python 克隆） | **rust A3** | **+0.9446** |
| 提示音 lingguang | ref A3 | +0.8019 |
| 提示音 lingguang | **rust A3** | **+0.7629** |
| 提示音 lingguang | ref A1（无提示） | +0.1534 |
| 提示音 lingguang | rust A1（无提示） | +0.1539 |

即：无提示音的 A1 与 Python 版**音色完全一致（1.0000）**；克隆路径 A3 与 Python 的
克隆结果 0.9446、与提示音 0.76（Python 自己是 0.80）——**克隆确实"像"提示音**，
use_spk_emb（CAMPlus → spk_head 896 维 → 插入 speaker pad 行）整条链路正确。

A2（use_zero_spk_emb）/ A3 均已跑通，产出 output/rust/A2_zero_spk.wav（3.92 s）、
output/rust/A3_clone.wav（3.28 s）。时长与 Python 不同是**预期**的：两边 RNG 不同，
停止步落在别处。

### 前端移植（子代理完成，全部对拍）

- `text.rs`（toolkit + text_segment_cut）：695 条语料 / 10425 个字段相等 / 4526 个
  切分向量一致；1,112,064 个码点 × 5 类字符判定的 0 差异。
- `tasks.rs`（tts_tasks）：params_test 36 个 request **JSON 逐字节一致**；
  19 个正常 + 13 个报错路径（连 ValueError 文案）一致。
- `audio.rs`：`read_wav` 与 torchaudio **逐样本相同**（158072/158072，max abs 0）；
  `write_wav` 与 torchaudio.save **逐字节相同**（含 libsndfile 的 fact/PEAK 块）；
  `resample` 与 torchaudio.transforms.Resample 最大绝对误差 4.5e-8 ~ 5.7e-6。
- `tok.rs`：90 条语料 / 838 个 token、11 条 `build_input_ids` 序列 / 946 个 token
  **完全一致**，8 个 pinned id 全部命中。
- `spk.rs`：CAMPlus 走 ort（`load-dynamic`，不下载不链接，运行时定位
  onnxruntime.dll）；kaldi fbank 全链路自实现，192 维 embedding 最大绝对误差 1.5e-5。

### 参考实现的两个"怪癖"，移植时**故意保留**

1. `frontend/toolkit.py:180` 的 `re.sub(r"([{ch_puncs}])\1+"…)` **漏了 f 前缀**，
   字符类其实是字面量 `{ch_puncs}`（即 c,h,p,u,n,s,_,{,}）。所以 `……` 从不被折叠，
   而 `cc`/`ss`/`pp` 会被折叠。Rust 端照抄，金标钉死。
2. `build_task_generation_kwargs` 会把 `**kwargs` 塞进返回 dict，`synthesize` 传了
   `spk_emb_path`，所以每个 request 都带 `"spk_emb_path": null`。

### 性能现状与下一步

分段（40 步、3.2 s 音频）：prefill 0.53 s / **DiT 7.92 s** / LLM+agg 1.42 s，
合计 9.87 s → **RTF 3.08**（Python 0.88）。模型加载 ~5 s（一次性）。

DiT 占 80%，单次 `dit_forward` 8.1 ms，对应 192 MB 权重流量 → 约 24 GB/s，
只有卡面带宽的 ~1/8。已排除的原因：dispatch 提交（已并成单 pass）、readback
（环内零同步）、buffer 分配（2.8 µs/次）、4-word 展开（反而更慢 → 不是纯延迟受限）。

下一步按收益排序：
1. **pred/null 合批**：`forward_with_cfg` 在 Python 里是 batch-2、权重只读一遍。
   要做就得给 ATTEND 加"分组注意力"（0-2 行与 3-5 行互不可见）+ PACK6，
   预计 DiT 7.9 s → ~4 s。
2. **GEMV 换成真正的 tiled GEMM**（每 workgroup 把权重 tile 读进 shared memory，
   再对全部 t 个 token 复用）：这是把 24 GB/s 拉向 100+ GB/s 的正路，
   预计还有 2-3 倍。做完才可能压到 0.88 以内。
3. 顺带：`linear_proj_audio`（agg）每步 1 次 8 层前向、LLM 每步满读 715 MB 权重，
   后者是 LLM+agg 那 1.42 s 的主因，量化或 fp16（DiT/VAE 部分）可再省。

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

## M3 记录（2026-10-31）

- 结构澄清：0.5b 的 audio encoder = Qwen2Model（24 层 896 因果）+ pad_patch_insert_cls
  （每 4 帧插 cls）+ 4 层 aggregator + fc3 → 对角高斯；`norm` LayerNorm 在 forward 中
  未使用；whisper 模块（semantic）在 0.5b 未启用 —— **全部复用 M1 内核族，零新内核**。
- dump_m3.py：waveform/frames/embeds/layer0 子模块/enc_out/agg_out/mean/std/latent/eps。
- vae.rs：frames 宿主 unfold、fc1（无 bias）fc2、24 层因果栈、patch+cls 宿主拼装、
  4 层栈、cls 抽取（每 5 行取第 5）、fc3 → mean/std（logvar=fc3 后 64 维，std=exp(0.5·logvar)）。
- bisect 结论：frames=0、embeds=3.7e-7、layer0 的 rms/q/k/v=1e-7、**rope 内核经独立
  numpy 复算验证 3.7e-8（第一版 numpy 金标有 in-place view bug，内核无误）**；
  发散点锁定 attention 输出（0.73）。已排除：双向注意力（更差 0.81）、theta=10000（0.79）。
- **M3 根因确认（2026-11-01）**：audio encoder 的 attention = 因果 + **滑动窗口 64**
（config: sliding_window=64, use_sliding_window=true，transformers 4.57 该路径真实生效）
+ Qwen2 rope(theta 1e6, split-half)——逐 token 验证：t<64 与 golden 误差为 0，
t≥64 有窗口边界残差（≤0.03 mean-abs）。ATTEND 内核的 window 参数已实现并接线
（qwen_lm window=0 / vae window=64），残差为窗口偏移细节（t−63 vs t−64 对齐），
下一轮微调后即可收口 M3（mean/std 门限 1e-3）。
注意：rope.py（torchtune 式）与 whisper 模块在 0.5b 中是死代码，真正生效的是
Qwen2Model 自带的 Qwen2Attention+Qwen2RotaryEmbedding。

## M3 续（2026-11-01b）

- numpy 参考确认：split-half@1e6 + window64 的 attention 与 golden rel 1.67e-7（attention 层面完全对上）。
- Rust 症状精确定位：my attend 输出 tok0-2 与 numpy 因果逐位一致（diff 0.0000），
  tok4 差 0.032、tok8 差 0.52、tok16 差 1.20、tok64 差 1.06 —— **误差随 token 索引增长**，
  高度怀疑 KV cache 写入只生效了位置 0（或 attend 读到的 cache 是陈旧的）。
- 下一步（精确续接点）：layer0_parts_check 里给 kv_write 之后的 kc 加 readback dump
  （对比 numpy rope 后的 k）——上一轮 patch 因 vae.rs 中 MING_VAE_FULL env 行导致
  anchor 不匹配未打入，需以实际文本（含 env 行）为锚重打；parity 测试里 l0_kcache
  分支已预留（continue 跳过）。
- 注意：layer0_parts 的 au[16..20] 是 MING_VAE_FULL env 行（默认 causal=0 ✓）。

## M3 收口（2026-10-31，✅ 全绿）

**根因 1（真 bug）：k_rope 用了 Q_HEADS 而不是 KV_HEADS。** ROPE 内核用
`token * n_heads * 64 + head * 64` 定位，而 k 缓冲区的行距是 `n_kv*64 = 128`；
传 14 头进去后，第 t 个 token 的写入基址变成 `t*896`（应为 `t*128`）——
token 0 恰好落在 0 所以对，token 越大偏得越远。这正好解释"tok0-2 精确、误差随
token 索引单调增长"的症状，也让"KV cache 只写了位置 0"的怀疑得以排除（cache 本身没问题）。
修法：k 的 rope 传 `uni2(KV_HEADS, 0)`（vae.rs 三处：stack / stack_range_check / layer0_parts）。

**根因 2（金标 bug）：dump_m3.py 的 cls 抽取写错。**
`a.reshape(1, -1, 5)` 会把 hidden 维一起卷进分组维（[1,210,896] → [1,37632,5]），
于是 `fc3_in/fc3_out/mean/std` 四个金标全是垃圾（彼此自洽因此骗过了
"fc3_in→fc3_out 1.3e-7"的自检）。正确写法是
`a.reshape(-1, patch+1, dim)[:, -1:, :].reshape(bsz, -1, dim)`，与
`Encoder.forward` 的 `x.reshape(-1, patch_size+1, dim)[:, -1:, :]` 一致。
修正后 fc3_in 1.3e-6、fc3_out 2.0e-7、mean 1.4e-6、std 8.3e-7、latent 1.4e-6。
**教训**：手工复刻 forward 路径的金标极容易与被复刻的实现分叉；能挂 hook 就不要手写。

**其余修正**：·滑窗 64 真正的缺口是 ATTEND 的 window 参数（现在 qwen_lm window=0 /
vae window=64）；·enc/agg 末端的 `norm` RMSNorm 确实生效（Qwen2Model.forward 末尾
self.norm），已加 `enc_norm/agg_norm`；·fc3 的 mean/std 是 [row, 128] 的前 64/后 64 维，
不是交错；·门限收紧到 1e-4。

## M4 记录（2026-10-31，✅ 全绿）

- 结构：latent [42,64] → fc1(64→896) → **nn.Upsample(x4, mode=linear, align_corners=False)**
  → 24 层 Qwen2（因果 + 滑窗 64）→ 末尾 RMSNorm → head.out(896→3530) →
  mag=exp(·) 截断 1e2 / phase=arg → iSTFT(n_fft=3528, hop=882, hann, padding="same")。
- **滑窗 64 在解码器上同样生效**：Python 端确实打印了
  "Sliding Window Attention is enabled but not implemented for sdpa" 的警告，
  但 4.57 是把窗口编进 4D mask 的，堆栈 rel_l2 1.5e-6 反证窗口确实生效——
  警告是误导，以数字为准。
- 上采样是**插值**不是可学习层（StreamingLinearUpsample 的非流式分支直接调
  self.upsampler）；align_corners=False 的源坐标 `t*n_in/n_out - 0.5`（负值夹到 0）
  在 CPU 上算即可。
- iSTFT 走 CPU rustfft（12.5 帧/s，不值得写 GPU FFT）：Hermitian 补全 → 逆变换
  带 1/N（norm="backward"）→ 乘 hann → 重叠相加 → 除以窗能量包络 → 两侧各去掉
  pad=(win-hop)/2=1323 点。窗口直接用 checkpoint 里的
  `audio.decoder.head.istft.window`。
- 踩坑：np.save 对 F 连续的数组会写成 fortran_order=True，Rust 的 npy 读取器不支持；
  金标导出时统一 `np.ascontiguousarray`。
- 新增 `src/istft.rs` / `src/vae_dec.rs`，`Gpu` 改为 Clone（设备句柄可共享，
  为 M5 单设备多阶段铺路）。

## M5 记录（进行中）

### 解码环（已跑通）

- `pipeline.rs`：ids → embeddings → prefill → 每步 CFM 采样 → latent history 滑动
  → VAE decode → 尾静音裁剪。顺序严格照抄 `sample()/generate()` 的怪癖：
  停止那一步的 latent 仍然计入、history 在 step 0 用 prompt latent 预热、
  每步先抽两次 torch.randn（noise + sde_noise）再跑采样器。
- **首个 grid 里的大坑：DiT 的 GELU 是 tanh 变体。**
  `fm/modules.py` 的 FeedForward 用 `nn.GELU(approximate="tanh")`，M2 里我用的是
  erf 近似（A&S 7.1.26），两者差到 1e-3，单次 `dit_forward` 就卡在 1.2e-4 上不去。
  改成 `0.5x(1+tanh(0.79788456(x+0.044715x³)))`（PyTorch 内核同式）后：
  block0 1.2e-4 → **2.4e-5**，flowloss 1.12e-3 → **1.28e-4**，
  端到端波形 1.8e-2 → **6.5e-3**。
- 40 步真实解码环对拍（金标 `goldens/m5/`，含 80 次 torch.randn 的原始抽样）：
  **步数 40 与 Python 完全一致**（停止判据一致 → 时长一致），
  z 逐步 6e-7 → 步 1 9e-5 → 末步 2e-3，波形 rel_l2 6.5e-3。
- **逐位对齐做不到，也不该拿它做门限**：单次 DiT 前向的 2.6e-5 已经是 f32 底噪
  （cuBLAS 分块归约 vs 我们的顺序 GEMV，不是 bug——权重是 bf16 无损展宽，值完全相同），
  而自回归 + 扩散把每步噪声放大成 ~1e-2。所以断言放在**结构不变量**上：
  步数/停止判据必须完全一致，波形给 2e-2 的宽容差。
  这也是为什么金标里的 `A1_bare.wav` 与 `goldens/m5/waveform.npy` **逐位相同**
  （rel_l2 = 0.0）——验收锚点是同一个东西。
- **RNG 决策（待用户拍板）**：对拍用金标注入 noise；生产路径用
  xoshiro256** + Box-Muller，**没有**复刻 torch 的 Philox/CUDA 抽样。
  代价：同 seed 下 Rust 波形与 Python 不会逐样本相同（内容和时长一致）。
  要逐样本一致就得复刻 `curand_normal4`（Philox4x32-10 + 具体的 counter/offset
  与 Box-Muller 映射），是一块独立且可验证的工作（手里已有 1895 seed 的前 64 个
  真值可以当测试向量）。

### 性能（定位与第一轮修复）

三个结构性问题，逐个都量过：

1. **每个 kernel dispatch 单独建 encoder + submit** —— 一句话音上万次 WDDM 提交。
   改成把 dispatch 记进队列、flush 时回放到**同一个 compute pass**
   （WebGPU 保证 dispatch 间可见性，顺序不变）。
2. **每次 readback 都 `poll(wait)`** —— 实测 ~175 µs/次。原来每个 DiT forward 有
   6 次 readback（t_mlp×2、x_emb×3、final×1）× 20 forward × 40 步 = 4800 次同步。
   现在采样环内**零 readback**：新增 `CFM_STEP` 内核在设备上完成
   `y += (pred + (pred-null)*cfg)*dt + k*sde`，DiT 的 t-embed/cond/final 也全走
   `GEMV_MROW` + 设备缓冲，只有每步末尾取一次 latent。
3. **buffer 生命周期**：批量化后一个 recording 引用着的 buffer 不能回收，
   累积几千个后 Vulkan 直接报 "Out of Memory"（不是真没显存）。所以 `flush()`
   必须 submit + poll，另外 DiT 的 KV scratch 从"每 block 每前向 4 MB"改成
   全引擎共用一对（原本单步 640 MB 存活缓冲）。

实测（release，P104-100）：单个 DiT forward 从 17 ms 降到 **12.9 ms**，
但**瓶颈不在提交了**：`GEMV_MROW` 按 (row, token) 分派，等于每个 token 重读一遍权重
（t=3 → 每前向 576 MB 权重流量）；Python 端 `forward_with_cfg` 是 batch-2、
权重只读一遍。**下一步（明确的优化项）**：改写 MROW 为"一次加载权重、对 t 个
token 累加"，预期 13 ms → ~3 ms；再叠加 pred/null 合批。这样才可能压进
Python 的 110 ms/步（RTF 0.88）。

**RNG 备注（M5 会用）**：prompt latent 与 CFM 噪声都走 torch.randn，
对拍时由金标注入 eps（已存 m3_eps.npy）；要做到 bit-exact 需复刻 torch Philox，
留到 M5 再决策。
