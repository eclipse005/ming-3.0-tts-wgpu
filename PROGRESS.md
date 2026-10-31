# Ming-3.0-tts → wgpu 移植（交接文档）

> **一句话状态**：Ming-3.0-tts-0.5b 已完整移植为纯 Rust + wgpu（无 Python / torch /
> CUDA 依赖），五个里程碑全部数值对齐，36 个参数用例全部跑通；给同一个 `--seed`
> 就能复现参考音频（时长逐采样点一致）。**唯一未达成的验收项是 RTF：2.51 vs Python 0.88。**

接手顺序：**本文** → [REVIEW.md](REVIEW.md)（代码结构评审与重构记录）→
[WGPU_PORT_ANALYSIS.md](WGPU_PORT_ANALYSIS.md)（最初的拆解方案）。

## 0. 交接速览

### 它是什么

`wgpu/` 是一个**独立 git 仓库**（与 Python 主仓 `D:\Ming-3.0-tts` 分离），把
`Ming-3.0-tts-0.5b` 的全部推理链路重写为 Rust + wgpu：Qwen2.5-0.5B LLM、DiT + CFM
流匹配、VAE 编码/解码、iSTFT、文本前端、分词、CAMPlus 说话人。约 8 千行 Rust，
21 个源文件，54 个测试。

### 三条命令

```bat
cd D:\Ming-3.0-tts\wgpu
cargo build --release

:: 单条合成（参数面与 Python infer.py 逐字一致）
target\release\ming-tts-wgpu.exe synth ^
  --model_path D:/Ming-3.0-tts/Ming-3.0-tts-0.5b ^
  --task TTS --text "今天天气真不错，我们一起去公园散步吧。" ^
  --output D:/Ming-3.0-tts/output/rust/A1_bare.wav --no_text_normalization

:: 全部测试（约 90 s，需要 GPU；release 才有意义，debug 慢 5-10 倍）
cargo test --release -- --test-threads=1
```

### 接手前必须知道的五件事

1. **数值不可能逐位一致，也不该拿它做门限。** 解码环是自回归 + 扩散：把输入扰动
   1e-7，我们**自己的**两次运行 40 步后 `z` 就差 1.5e-2（放大 ~1e5 倍，见测试
   `loop_is_chaotic`）。能保证的是**每步 ~1e-4 数值一致 + 停止判据一致 +
   音色一致 + 内容一致**。
2. **RNG 已经和 torch 完全对齐**（`src/rng.rs`，28160 个值误差 1.4e-6）。
   不要动它——它现在是"同 seed 能复现参考"的唯一依据。
3. **DiT 占 79% 的时间**，是 RTF 唯一的战场；LLM 15%，prefill 5%。
4. **权重按 bf16 原样上传、激活全 f32**。不要改 f16：LLM 残差流平台在 7.4e4，超 f16
   上限（Python 侧同样结论，见决策记录）。
5. **`goldens/` 是回归网的根**，不许为了过测试改金标；改代码。

### 下一步从哪开始

只剩性能。按收益排序的四步在 [第 5 节](#5-未完成与已知问题)，其中 **pred/null 合批**
和 **tiled GEMM** 是决定能不能进 0.88 的两条。

## 1. 现状

| 验收项 | 状态 | 证据 |
|---|---|---|
| M0 骨架（设备 / bf16 权重 / 内核） | ✅ | rmsnorm 1.0e-7、gemv 3.0e-7 |
| M1 LLM（prefill + KV 解码 + 停止头） | ✅ | 真实 40 步重放：z 8.1e-7、停止步一致 |
| M2 DiT + CFM 采样 | ✅ | 41 步对拍 worst 1.28e-4、y 链 5.8e-5 |
| M3 VAE 编码（波形→latent） | ✅ | mean 1.4e-6 / std 6e-8(abs) / latent 1.4e-6 |
| M4 VAE 解码 + iSTFT | ✅ | 最终波形 1.9e-6，长度逐采样点一致 |
| M5 文本前端 | ✅ | 695 条语料 / 10425 个字段、36/36 request dict 逐字节 |
| M5 音频 / 分词 / 说话人 | ✅ | wav 逐字节、resample 4.5e-8、tokenizer 946 token、CAMPlus 1.5e-5 |
| M5 解码环 | ✅ | 步数与参考一致，z 逐步 6e-7 → 1e-3 |
| M5 RNG（torch 对齐） | ✅ | 28160 个抽样值最大误差 1.4e-6 |
| M5 36 个参数用例 | ✅ 跑通 / ⚠️ 6 例轨迹分岔 | 33/36 时长一致，log-mel 中位 0.99949 |
| CLI 参数面 = infer.py | ✅ | `rename_all = snake_case`，逐 flag 对齐 |
| **RTF** | ❌ | **2.51**（Python 0.88） |

## 2. 怎么跑

### 运行期环境变量

合成不带提示音的用例（A1/A2/B*/C*/D*/E*/F*/G*/H*）不需要任何额外配置。
带提示音 / `--use_spk_emb` 的用例要能加载 onnxruntime：按 `ORT_DYLIB_PATH` →
`MING_ONNXRUNTIME_DIR` → `./onnxruntime.dll` → 本机 conda 路径的顺序查找，
换机器时通常要设第一个。`MING_CAMPPLUS_ONNX` 可覆盖 campplus.onnx 路径。

### 调试开关

| 环境变量 | 作用 |
|---|---|
| `MING_PROFILE=1` | 打印 prefill / DiT / LLM+agg 分段耗时 |
| `MING_NOISE_GOLDEN=<dir>` | 注入金标 torch.randn 抽样（`goldens/m5/` 或 `goldens/m5_params/<case>/`），用于逐样本对拍 |
| `MING_DUMP_IDS=<file>` | 把首条分片的 token id 写盘，用于和参考的 `build_input_ids` 比 |
| `MING_MAX_PENDING=<n>` | 一次提交的 dispatch 上限（默认 2048） |
| `MING_KV_TRACE=1` | VAE 编码器 dump KV cache（诊断） |
| `MING_MODEL` / `MING_GOLDEN` | 覆盖模型 / 金标目录（测试用） |

### 验证脚本（Python 侧，需要 videolingo 环境）

```bat
set PYTHONPATH=D:\Ming-3.0-tts\.runtime\pylibs446
set PY=C:\Users\ADMIN\miniconda3\envs\videolingo\python.exe

:: 单条对比：时长 / rel_l2 / log-mel 相关 / 音色
%PY% wgpu\tools\compare_wav.py output\params_test\A1_bare.wav output\rust_params\A1_bare.wav

:: 36 用例全量跑 + 打分（写 output/rust_params/report.json）
%PY% wgpu\tools\check_params_align.py

:: 音色相似度（CAMPlus 余弦）
%PY% wgpu\tools\check_speaker.py
```

### 重新生成金标（只在改了参考实现时才需要）

`tools/dump_*.py` 每个里程碑一个，用 `python -I` +
`sys.path.insert(0, 'D:/Ming-3.0-tts')` 运行；涉及 torch 的再加上面的 `PYTHONPATH`。
**注意**：`.npy` 必须是 C 顺序（`np.ascontiguousarray`），Rust 读不了 fortran_order。

## 3. 代码地图

| 文件 | 职责 | 不变量 / 注意 |
|---|---|---|
| `src/main.rs` | CLI（与 infer.py 同参数面） | clap 必须 `rename_all = "snake_case"` |
| `src/synth.rs` | 一次合成的编排：条件 → 分片 → 写盘 | 说话人 embedding 必须从**原生**采样率一跳重采样 |
| `src/pipeline.rs` | 解码环、Noise trait、尾静音裁剪 | 步数/停止判据与参考一致；裁剪先截整帧 |
| `src/rng.rs` | torch CUDA randn 精确复刻（Philox4x32-10） | 每个 `randn` = 一次 `draw(n)`，计数 +1 |
| `src/ops.rs` | **所有设备原语**：Lin / RopeTables / AttendParams / 全套 kernel | 改内核只改这里 |
| `src/nn.rs` | uniform arena + dispatch 录制 | binding 0 是动态偏移 uniform |
| `src/gpu.rs` | 设备、缓冲、批量提交与 flush | flush 必须 submit + poll |
| `src/shaders.rs` | 全部 WGSL | DiT 的 GELU 是 **tanh** 变体 |
| `src/qwen_lm.rs` | LLM | 窗口 0（全因果），rope θ=1e6 split-half |
| `src/dit.rs` | DiT + Aggregator + CFM | 窗口 0、双向、rope θ=1e4 **交错** |
| `src/vae.rs` | VAE 编码器 | 因果 + **滑窗 64**；后验 std = softplus + 1e-4 |
| `src/vae_dec.rs` | VAE 解码器 | 因果 + 滑窗 64、上采样 ×4 |
| `src/istft.rs` | CPU iSTFT（rustfft） | 逆变换带 1/N，窗能量包络归一 |
| `src/text.rs` | 文本前端 | 参考实现的 f 前缀 bug 要照抄 |
| `src/tasks.rs` | 任务模板 | 36/36 逐字节；caption JSON 顺序有意义 |
| `src/tok.rs` | 分词器 + `build_input_ids` | 8 个 pinned id 加载时校验 |
| `src/spk.rs` | kaldi fbank + CAMPlus ONNX | 16 kHz 输入、均值减除 |
| `src/audio.rs` | wav 读写 + torchaudio 重采样 | 写盘逐字节复刻 libsndfile |
| `src/weights.rs` | safetensors 加载（bf16 原样） | 全量拷进 RAM（~2.9 GB） |
| `src/golden.rs` | npy 读取 + rel_l2 | CLI 的 `MING_NOISE_GOLDEN` 也用它 |
| `src/parity.rs` | parity 测试（`#[cfg(test)]`） | 与各模块内测试合计 54 个 |

## 4. 验收数据

### 4.1 逐模块数值对齐（对 Python 金标）

| 阶段 | 指标 | 实测 |
|---|---|---|
| LLM 40 步重放 | z rel_l2 | 8.1e-7，停止步 40 = Python |
| DiT 单前向 | rel_l2 | 2.6e-5（f32 累加序底噪） |
| CFM 41 步 | worst rel_l2 | 1.28e-4 |
| VAE 编码 | mean / std / latent | 1.4e-6 / 6e-8(abs) / 1.4e-6 |
| VAE 解码 | 最终波形 | 1.9e-6 |
| 克隆条件 | prompt latent / spk_head | 2.2e-6 / 1.2e-7 |
| torch RNG | 28160 个抽样值 | 1.4e-6 |

### 4.2 36 个 params_test 用例

A4 / I1-I4 的参考音频是批量脚本**复用缓存 prompt 编码**产出的（不消耗后验抽样），
单进程 CLI 复现不了；用 `tools/ref_nocache.py` 重渲染后对比（表里标 cache-free）。

| 用例 | 时长比 | 波形 rel_l2 | log-mel 相关 | 音色 cos | 备注 |
|---|---|---|---|---|---|
| A1_bare | 1.0000 | 6.57e-03 | 0.99999 | 1.0000 |  |
| A2_zero_spk | 1.0000 | 3.19e-02 | 0.99993 | 0.9999 |  |
| A3_clone | 1.0000 | 1.03e+00 | 0.90553 | 0.9685 | 轨迹分岔 |
| A4_clone_latent_only | 1.0000 | 1.19e-01 | 0.99967 | 0.9995 | cache-free 参考 |
| B1_speed_slow | 1.0000 | 1.50e-01 | 0.99926 | 0.9985 |  |
| B2_speed_mid | 1.0000 | 8.51e-01 | 0.97498 | 0.9775 |  |
| B3_speed_fast | 1.0000 | 9.07e-02 | 0.99958 | 0.9983 |  |
| B4_pitch_low | 1.0000 | 6.38e-01 | 0.98930 | 0.9912 |  |
| B5_pitch_high | 1.0000 | 5.77e-02 | 0.99992 | 0.9999 |  |
| B6_volume_low | 1.0000 | 1.20e+00 | 0.96920 | 0.9272 | 分岔 |
| B7_volume_high | 1.0000 | 2.16e-02 | 0.99996 | 0.9999 |  |
| C1_happy | 1.0000 | 7.91e-01 | 0.99311 | 0.9840 |  |
| C2_sad | 1.0000 | 9.99e-01 | 0.94007 | 0.9627 | 分岔 |
| C3_angry | 1.0000 | 5.18e-01 | 0.99924 | 0.9981 |  |
| C4_fearful | 1.0000 | 1.08e+00 | 0.99267 | 0.9864 |  |
| C5_disgusted | 1.0000 | 1.67e-01 | 0.99921 | 0.9989 |  |
| C6_surprised | 1.0000 | 1.22e+00 | 0.93321 | 0.9292 | 分岔 |
| C7_neutral | 1.0400 | 1.14e+00 | 0.94224 | 0.9723 | 分岔+步数差1 |
| D1_cantonese | 1.0000 | 1.10e-01 | 0.99986 | 0.9997 |  |
| D2_sichuanese | 1.0000 | 7.36e-02 | 0.99995 | 0.9999 |  |
| E1_ip_lingyuechuan | 1.0000 | 5.43e-02 | 0.99993 | 0.9999 |  |
| E2_ip_lingxiyue_happy | 1.0000 | 6.13e-01 | 0.99669 | 0.9929 |  |
| E3_ip_us_emma | 1.0000 | 3.76e-02 | 0.99996 | 1.0000 |  |
| F1_japanese | 1.0000 | 1.14e-02 | 0.99998 | 0.9999 |  |
| F2_korean | 1.0000 | 4.66e-01 | 0.99618 | 0.9889 |  |
| F3_german | 1.0000 | 9.78e-02 | 0.99940 | 0.9993 |  |
| F4_french | 1.0000 | 6.45e-02 | 0.99974 | 0.9994 |  |
| F5_spanish | 1.0000 | 5.49e-02 | 0.99997 | 0.9999 |  |
| G1_asmr_whisper | 1.0000 | 5.91e-03 | 1.00000 | 1.0000 |  |
| G2_empress | 1.0704 | 4.28e-01 | 0.88707 | 0.9551 | 分岔+步数差 |
| H1_whisper_secret | 1.0270 | 9.92e-01 | 0.84501 | 0.9694 | 分岔+步数差 |
| H2_news_anchor | 1.0000 | 9.86e-03 | 1.00000 | 1.0000 |  |
| I1_temperature_06 | 1.0000 | 2.26e-03 | 1.00000 | 1.0000 | cache-free 参考 |
| I2_temperature_10 | 1.0000 | 6.50e-02 | 0.99990 | 0.9998 | cache-free 参考 |
| I3_cfg_10 | 1.0000 | 3.33e-03 | 1.00000 | 1.0000 | cache-free 参考 |
| I4_cfg_35 | 1.0000 | 4.63e-01 | 0.99798 | 0.9929 | cache-free 参考 |

> 汇总：**36/36 跑通**；**33/36 时长逐采样点一致**；log-mel 中位 **0.99949**
> （22 个 ≥0.999、27 个 ≥0.99）；音色 cos 33 个 ≥0.96。6 个 log-mel < 0.95 的用例
> 属轨迹分岔（见 4.4），不是功能错误。

### 4.3 音色（克隆）链路

| A | B | CAMPlus cos |
|---|---|---|
| 参考 A1（无提示） | rust A1 | **1.0000** |
| 参考 A3（Python 克隆） | rust A3 | **0.9685** |
| 提示音 lingguang | 参考 A3 | 0.8019 |
| 提示音 lingguang | rust A3 | 0.8013 |

### 4.4 为什么做不到逐采样点一致（量化过，别再试）

测试 `m5_chaos_tests::loop_is_chaotic`：把同一份输入乘 (1+1e-7)，我们**自己的**
两次运行 40 步后 `z` 的相对差就到 **1.5e-2**，放大约 1e5。而移植版每步至少有
~1e-4 的数值差（cuBLAS 分块归约 vs 顺序 GEMV 的 f32 底噪），所以某些轨迹必然翻面。

证据：分岔前逐步都对齐（step0 `z` 8.6e-7，前 11 步 1e-4 量级）；换 cfg / temperature
（I1 / I3）反而能到 **rel_l2 2-3e-3、log-mel 1.00000**——流程没问题，是轨迹敏感度。

### 4.6 逐位一致到底能不能做到（用测量回答，别再猜）

**不能，而且原因不在我们这边：参考实现自己不是精确值。**

拿一个真实的 896×896 GEMM（M0 金标：`layer4_norm_out` × `layer4_q_w`）算 f64 真值：

| 结果 | 离 f64 真值 |
|---|---|
| **torch（参考实现）** | **1.553e-07** |
| 我们的归约顺序（32 lane + 5 级树，numpy 复刻） | **5.171e-08** |
| 两者之差 | 1.635e-07 |

也就是说：**我们的求和比 cuBLAS 更准 3 倍**，而"离 torch 1.6e-7"里的每一分都是 torch 自己的 f32 误差。
又扫了归约顺序（split = 2/4/8/16/32 × 树形/线性共 10 种），**没有任何一种**比 1.635e-7 更接近 torch
——cuBLAS 的 tile/split-K 顺序无法靠调参逼近。

那 2.6e-5 的单次前向误差从哪来？看串联：单算子 3e-7 → 一个 block 输出 2.35e-5。
放大发生在 **4096 宽的 MLP**：它是"大数相消"的和（正负抵消，条件数 ~√4096 = 64），
把每个元素上 ulp 级的差异放大成 ~1e-5 相对误差。这也解释了为什么把 GELU 从 erf 换成
tanh（每元素差 1e-3）能让 block0 从 1.2e-4 掉到 2.35e-5——**逐元素误差会被这个和放大**。

要做到逐位一致，必须同时满足：

1. 每个 GEMM 的归约树与 cuBLAS 逐位相同——算法未公开，且随 shape / 架构变化；
2. 每个超越函数（exp / tanh / sin）与 CUDA libdevice 逐位相同——**WGSL 的 exp/tanh/sin
   由 Vulkan 驱动实现，我们无法指定、也无法得知它用哪个近似**；
3. softmax / RMSNorm 的归约同理。

这不是"努力不够"，是**在 wgpu 层面拿不到那个控制权**。同一台设备确实消除了硬件差异
（我们自己的两次运行逐位一致，否则测试早红了），但消除不了"与另一套实现逐位相同"
所需的顺序与超越函数信息。

> 结论：能保证的是**每步 1e-4 量级数值一致 + 停止判据一致 + 音色一致 + 内容一致**；
> 逐采样点一致需要复刻 cuBLAS 与 CUDA 数学库的逐位行为，不在可及范围内。
> 想让 6 个分岔用例收敛，唯一现实的路径是把 2.35e-5 这个 block 级误差再压一个数量级
> （见第 5 节 P1）。

### 4.5 性能（release，P104-100，A1_bare：40 步 / 3.2 s 音频）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| prefill（43 token） | 0.44 s | 5% |
| **DiT + CFM（20 次前向 × 40 步）** | **6.36 s** | **79%** |
| LLM 解码 + aggregator | 1.23 s | 15% |
| 合计（解码环） | 8.03 s | **RTF 2.51** |
| 模型加载 + VAE 解码 | ~5.7 s | 一次性 |

## 5. 未完成与已知问题

### P0：RTF 2.51 → 目标 < 0.88（唯一未达成的验收项）

DiT 是唯一战场（79%）。单次 DiT 前向 8 ms，对应 192 MB 权重流量 → 约 24 GB/s，
只有卡面带宽的 ~1/8。**已验证排除**：dispatch 提交（已并成单 pass）、readback
（环内零同步）、buffer 分配（2.8 µs/次）、4-word 展开（反而更慢 → 不是纯延迟受限）。

按收益排序：

1. **pred/null 合批**：Python 用 `forward_with_cfg` 一次 batch-2 前向，权重只读一遍。
   给 `AttendParams` 加分组注意力（前 3 行与后 3 行互不可见）+ 一个 PACK6 即可。
   预期 DiT 6.36 s → ~3.5 s。改动点：`src/shaders.rs`（ATTEND 加 group 参数）、
   `src/dit.rs`（`dit_forward_dev` 拼 6 行、CFM 读两处偏移）。
2. **tiled GEMM**：`Ops::linear` 换成"权重 tile 进 shared memory、对全部 token 复用"。
   这是把 24 GB/s 拉向 100+ GB/s 的正路，预期再 2-3 倍。改动点：`src/shaders.rs`
   新 kernel + `src/ops.rs::linear_into`。
3. bind group 缓存（`nn::record` 是唯一入口）、按步预分配 scratch（DiT 形状固定 t=3）。
4. LLM 段 1.23 s：每 token 读 1.92 GiB 权重，是带宽地板；只有量化能再降。

### P1：6 个轨迹分岔用例

不是 bug，是 4.4 的必然。若要收窄，唯一的路是**把每步误差再降一个量级**：DiT / LLM 的
GEMV 改 f64 累加（或对齐 cuBLAS 的分块归约顺序），每步 1e-4 → 1e-5，分岔会显著推迟。
代价是算力和 RTF，与 P0 冲突，**建议先做 P0**。

### P2：功能缺口（原版有、这个移植没做）

- `--spk_emb_path`（.pt 文件）未实现：需要 torch 序列化格式，目前显式报错。
- pynini 文本规范化未移植：`--no_text_normalization` 关闭时的数字/符号规范化缺失。
  非中文语言走"绕过 TN"分支，所以 F 组不受影响。
- 多分片路径（`max_length` 切出 >1 个 fragment）存在但**没有验收样本**。
- 流式推理未做（原版非流式路径也没用它）。

### P3：内存

`Weights::load` 把 2.9 GB 权重全量拷进 RAM。改成 mmap 借用可省这段内存；不影响 RTF。

## 6. 坑清单（都实际踩过，别重犯）

1. **金标 bug 会自证**：M3 曾看起来"验证通过"，因为金标 std 就是用错公式导出的。
   交叉验证要靠"用真实权重反查"或"用下游产物反推"。
2. **torch 的 SDPA 对滑窗只打警告不报错**：窗口其实生效了（编码器/解码器都是 64）。
   以数字为准，别信日志。
3. **`.npy` 的 fortran_order**：`np.save` 对 F 连续数组会写 fortran 序，Rust 读不了。
4. **每个 dispatch 一个 command buffer 会拖垮 WDDM**（上万次提交）。批量化后必须定期
   `flush()`（submit + poll），否则 Vulkan 报 "Out of Memory"——那是分配数超限，不是真没显存。
5. **`queue.write_buffer` 与延迟提交语义冲突**：所以 uniform 走 arena + 动态偏移。
6. **DiT 的 GELU 是 tanh 变体**（`approximate="tanh"`），用 erf 近似会把 parity 卡在 1.2e-4。
7. **VAE 后验是 `softplus(scale)+1e-4`**，不是 `exp(0.5·logvar)`。
8. **k 的 ROPE 要用 `KV_HEADS` 不是 `Q_HEADS`**（否则 token 基址错位，误差随 token 增长）。
9. **重采样从原生采样率一跳**，不要 48k→44.1k→16k。
10. **prompt latent 抽样要转置**：torch 抽 `[1,64,T]`，我们的 mean/std 是 `[T,64]`。
11. **尾静音裁剪**：Python 先截到整帧再切尾巴，否则长度差一帧。
12. **参考脚本的缓存**：`test_params.py` 用一个 MingTTS 实例跑 36 例，prompt 编码被缓存，
    A4/I1-I4 的参考因此**不消耗后验抽样**；要用 `ref_nocache` 才可比。

## 7. 决策记录


- 权重：**bf16 原样存储 + shader 内解包 f32 计算**（bf16⊂f32 无损；总线减半）；
  不走 ASR 项目的 f16 权重路线。
- 激活：**全 f32**（LLM 残差流平台 7.44e4 超 f16 上限；indextts f16-CFM 漂移 9e-2 教训）。
- iSTFT：CPU rustfft（12.5 帧/s，不值得写 GPU FFT）。
- CAMPlus 说话人：ort CPU ONNX，不移植。
- 独立 git 仓库（本目录），与 Python 主仓分离，过程按里程碑提交。


## 7.1 设备假设清理（2026-11-01）

原来残留三处"像是为这张卡写的"的地方，已全部消掉（数值不变，52/52 测试通过）：

| 位置 | 改法 |
|---|---|
| `shaders.rs` ATTEND 里 `kbase = kvh * 512u * 64u` | cap 变成 uniform 参数（`AttendParams::cap`），换 `max_cache_len` 不同的模型不用改内核 |
| `gpu.rs` `Backends::VULKAN \| DX12` | 改成 Vulkan/Metal/DX12/GL 全开 + 显式优先级排序，macOS 不再无适配器 |
| `gpu.rs` `required_limits: Limits::default()` | 改成 `adapter.limits()`，大卡自动可用更大 workgroup / 更大 storage binding |

剩下的设备相关项只有两个常量（`MAX_PENDING=2048`、`ARENA_SLOTS=4096`，都可用环境变量覆盖）
和性能结论本身。金标是这台机器的 CUDA torch 生成的，换卡重导会有 ulp 级漂移（阈值留了 10-100x 余量）。

## 8. 复用来源


- `D:\qwen3-asr-wgpu`：wgpu 30 设备/BulkUpload/weights/safetensors、Qwen 解码器、
  WGSL 内核族（改 f32）、tokenizers 集成。
- `D:\index-tts\indextts-rs`：golden npy + rel_l2 parity 方法论、CPU oracle twin、
  f32 CFM 精度教训、预分配工作区模式。


## 9. 历史记录（按时间，含已过时的数字）

> 以下保留当时的**错误假设与修复细节**，它们解释了现在的代码为什么长这样。
> 数字可能已被第 4 节取代。

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

## M5 记录（续）：torch RNG 反解成功 —— 现在只凭 seed 就能复现参考音频

之前对拍只能靠 MING_NOISE_GOLDEN 注入金标抽样；一旦改用原生 RNG，时长都会变
（A2 我们 3.92 s vs 参考 3.20 s），"同 seed 对齐"无从谈起。本轮把 torch 的
CUDA randn 完全反解出来了，写法在 src/rng.rs 顶部：

- 算法：curand 的 **Philox4x32-10**（10 轮，key 每轮 +W0/+W1）
- key = (seed & 0xffffffff, seed >> 32)
- **counter.x = 抽签序号**（每调用一次 torch.randn 就 +1）
- **counter.z = 该次抽样内的元素下标**
- 值 = `sqrt(-2 ln u1) * sin(2π u2)`，其中 `u = x * 2^-32 + 2^-33`（curand_uniform，
  注意是 (0,1] 的那个映射，不是 `x/2^32`）

这条不是猜的：用参考实现自己录下来的抽样逐值比对——**80 次抽签 / 28160 个值，
最大绝对误差 1.43e-6**（torch 用的是 `__sincosf` 快速内建，我们用精确 sin，
差在最后 1-2 个 ulp）。测试固定在 `rng::tests::matches_torch_randn`。

**效果**：不注入任何金标、只给 `--seed 1895`，A1 的输出与 Python 参考
**同长度（141120 采样点）、同 40 步停止**，rel_l2 6.57e-3、log-mel 相关 0.999991
——和"注入金标"时的 6.46e-3 只差 RNG 的 float 舍入。也就是说移植版现在**自己就能
复现参考音频**，不再依赖任何录制数据。

顺带修掉一个真 bug：prompt latent 的后验抽样，torch 抽的是 `[1, 64, T]`
（通道优先），而我们的 mean/std 是 `[T, 64]`，必须把抽样转置后逐元素配对。
原本用"每次抽 64 个再取第一个"的写法，形状和顺序都不对。


### 全参数对齐过程中修掉的三个真 bug

1. **VAE 后验的 std 用错了公式**（影响所有克隆/提示音用例）。
   diffusers 的 `OobleckDiagonalGaussianDistribution` 是
   `std = softplus(scale) + 1e-4`，不是 `exp(0.5*logvar)`。
   M3 当时"验证通过"其实是**自证**：金标 std 也是我自己按错的公式导出的，
   两边当然一致。用参考实现算出来的 latent 反查才发现差了 1.4%。
   修正后 prompt latent 从 rel 1.4e-2 → **2.2e-6**。
2. **说话人 embedding 的重采样走了两跳**：Python 是从原生 48 kHz 直接重采样到
   16 kHz 喂 CAMPlus，我写成了 48k → 44.1k → 16k，滤波器不同、embedding 会变。
   已改成各自从原生采样率出发。
3. **prompt latent 的抽样布局**：torch 抽的是 `[1, 64, T]`（通道优先），
   而 mean/std 是 `[T, 64]`，必须转置后再逐元素配对。

### 关于"逐个采样点对齐"能做到什么程度（已量化）

解码环是自回归 + 扩散：把**同一份输入扰动 1e-7**，我们自己的两次运行在 40 步后
`z` 的相对差就到 **1.5e-2**（放大约 1e5 倍）。也就是说：

- 任何移植（包括同一份代码换个编译器）都不可能让 40 步后的波形逐采样点一致；
- 能保证的、也已经做到的是：**每步 ~1e-4 的数值一致** + **停止判据一致**
  （步数对） + **音色一致**（CAMPlus 余弦）+ **内容一致**（log-mel 相关）。
- 典型结果：A1 log-mel 0.999991、时长逐样本相同；克隆类因为条件更强，
  轨迹分岔更早，但音色余弦 0.9695（参考自身对提示音也只有 0.80）。

### 验收锚点（用户指定）

1. **初步对齐**：与 `D:\Ming-3.0-tts\output\` 下现有音频对齐（clone_demo / ab_graph 等）
2. **最终验收**：与 `D:\Ming-3.0-tts\output\params_test\` 36 个全参数样本全面对齐
3. **API 一致**：CLI 参数面与 `infer.py` 完全一致
   （--model_path/--task/--text/--device/--seed/--language_id/--instruction/
   --speed/--pitch/--volume/--emotion/--dialect/--style/--ip/--cfg/--sigma/
   --temperature/--max_length/--prompt_wav_path/--prompt_text/--use_spk_emb/
   --use_zero_spk_emb/--output）


## M0 记录（2026-10-31）

- `cargo build`/`test` 全绿；`ming-tts-wgpu.exe devices` 列出 [Vulkan] P104-100。
- golden dump：真实 clone-demo 输入（lingguang + A3 文本），T=109，h3 激活含 7.63e3 尖峰。
- 内核：`RMSNORM`（256 lane 行归约）、`GEMV_BF16W`（warp/行 + bf16 解包 f32 累加 + bias）。
- **教训**：① 本 checkpoint 的 q/k/v 投影带 bias（absmax ~17），gemv 必须融合 bias；
  ② checkpoint key 有 `audio.encoder/decoder` 前缀的同名层，匹配要用 `model.model.layers.*` 全前缀；
  ③ bf16 1.0 LE = `[0x80,0x3F]`（测试数据写错过一次）。


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


## M5 记录（历史：性能第一轮定位）

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