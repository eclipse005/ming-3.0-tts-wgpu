# Ming-3.0-tts → wgpu 移植（交接文档）

> **一句话状态**：Ming-3.0-tts-0.5b 已完整移植为纯 Rust + wgpu（无 Python / torch /
> CUDA 依赖），五个里程碑全部数值对齐；36 个参数用例 **36/36 时长逐采样点一致**，
> 26 例波形级重合（rel_l2 ≤ 5e-2，最好 3.6e-5），其余 10 例只有后段的相位差
> （log-mel ≥ 0.9959，见 4.4）。**唯一未达成的验收项是 RTF：长句解码环 5.35–5.91 s
> （5.00 s 音频，RTF 1.07–1.18），Python CUDA Graph 是 5.11 s（RTF 1.008）——
> 已基本追平，差距只剩 DiT 的 workgroup 并行度。见 0.5.8。**

接手顺序：**本文** → [REVIEW.md](REVIEW.md)（代码结构评审与重构记录）→
[WGPU_PORT_ANALYSIS.md](WGPU_PORT_ANALYSIS.md)（最初的拆解方案）。

## 0. 交接速览

### 它是什么

`wgpu/` 是一个**独立 git 仓库**（与 Python 主仓 `D:\Ming-3.0-tts` 分离），把
`Ming-3.0-tts-0.5b` 的全部推理链路重写为 Rust + wgpu：Qwen2.5-0.5B LLM、DiT + CFM
流匹配、VAE 编码/解码、iSTFT、文本前端、分词、CAMPlus 说话人。约 8 千行 Rust，
21 个源文件，64 个测试。

### 三条命令

```bat
cd D:\Ming-3.0-tts\wgpu
cargo build --release

:: 单条合成（参数面与 Python infer.py 逐字一致）
target\release\ming-tts-wgpu.exe synth ^
  --model_path D:/Ming-3.0-tts/Ming-3.0-tts-0.5b ^
  --task TTS --text "今天天气真不错，我们一起去公园散步吧。" ^
  --output D:/Ming-3.0-tts/output/rust/A1_bare.wav --no_text_normalization

:: 全部测试（约 170 s，需要 GPU；release 才有意义，debug 慢 5-10 倍）
cargo test --release -- --test-threads=1
```

### 接手前必须知道的五件事

1. **数值不可能逐位一致，也不该拿它做门限。** 解码环是自回归 + 扩散：把输入扰动
   1e-7，我们**自己的**两次运行 40 步后 `z` 就差 1.5e-2（放大 ~1e5 倍，见测试
   `loop_is_chaotic`）。现在每步误差 ~1e-6（已经是 f32 GEMM 归约序的底噪），
   所以 36 例里 26 例全程重合、10 例只在后段分岔。能保证的是**每步 ~1e-6 数值一致
   + 停止判据一致 + 音色一致 + 内容一致**。
2. **RNG 已经和 torch 完全对齐**（`src/rng.rs`，28160 个值误差 1.4e-6）。
   不要动它——它现在是"同 seed 能复现参考"的唯一依据。
3. **DiT 仍占约 66% 的时间**（prefill 10%，LLM+agg 24%），是 RTF 的主战场。
4. **权重按 bf16 原样上传、激活全 f32**。不要改 f16：LLM 残差流平台在 7.4e4，超 f16
   上限（Python 侧同样结论，见决策记录）。
5. **`goldens/` 是回归网的根**，不许为了过测试改金标；改代码。

### 下一步从哪开始

只剩性能。**先读 0.5 节**（本轮已确认的数字、成功方案、排除项和剩余缺口）。
DiT 的 t=6 线性层已从 24 GB/s 提到 40–47 GB/s，剩下的提升点在 K 分裂并行和调度开销。

## 0.5 RTF 攻坚交接（2026-11-01 第二轮）

> **一句话**：数值和听感对齐有 64 个测试 + 36 例 sweep 守着；本轮把解码环
> **5.56 s → 3.72 s（1.49×）**，DiT 单阶段 **4.30 s → 2.46 s（1.75×）**。
> RTF 从 1.74 降到 **1.16**，距 Python 0.88 还差约 24%。

### 0.5.1 时间账（A1_bare：3.2 s 音频，背靠背）

| 阶段 | 本轮前 | 现在 | Python（RUNBOOK 反推） |
|---|---|---|---|
| prefill（43 token） | 0.40 s | **0.38 s**（tile 路线省约 20 ms） | 含在 4.4 s 内 |
| **DiT + CFM（10 次合批前向 × 40 步）** | **4.30 s** | **2.46 s** | **3.34 s** |
| LLM 解码 + aggregator | 0.90 s | 0.89 s | 0.84 s |
| 解码环合计 | **5.56 s → RTF 1.74** | **3.72 s → RTF 1.16** | **4.4 s → RTF 0.88** |

上表是本轮同一分钟内交替运行 base / hybrid 的结果。更早文档里的 6.4–6.5 s 是热机后
的另一时段（见 4.12），不要和这里的冷机基線混用。**所有 A/B 必须背靠背。**

### 0.5.2 已确认的事实（不必重测）

| 事实 | 数字 | 出处 |
|---|---|---|
| 每次 DiT 前向 | **6.1 ms**（本轮前 12.1 ms） | MING_PROFILE |
| 每次前向的权重流量 | 200 MB bf16，有效带宽 **40–47 GB/s** | `linear_bandwidth` |
| 40 步解码的 dispatch / buffer | **84,308 / 61,535**（每前向 211 次 dispatch） | MING_MEM=1 |
| 单次 dispatch 成本（同卡实测） | ~5.7 µs → 84k 次 = **0.48 s**，不是主因 | qwen3-asr `docs/perf.md` |
| 同卡 M=1 GEMV 可达到 | **226–268 GB/s**（qwen3 的 f16/int8 路径） | 同上 |
| 同卡 tiled GEMM 相对基线 | **48.2×**（128×64×32 单缓冲） | ctc `src/gemm_kernel.rs` |
| 我们 M=1，小投影 | **72 GB/s**（896×896） | `linear_bandwidth` |
| 我们 M=1，宽投影 | **64–66 GB/s** | `linear_bandwidth` |

### 0.5.3 已试并排除（别重复）

| 尝试 | 结果 | 原因 |
|---|---|---|
| pred/null 合批成 6 行 | 6.45 → **4.75 s** ✅ | 权重读一遍顶两遍 |
| vec4 权重 + **单 token** | 20 → **62–65 GB/s** ✅ | 16 B 一次加载 |
| **配对激活 `vec2` + 6 token** | t6 24 → **30–41 GB/s** ✅ | 每次少 24 条激活 load |
| **按 K 调 lanes/行**（8 / 32） | t6 → **40–47 GB/s** ✅ | 减少归约步数且保持并行度 |
| **prefill 48×64 tile** | 全句省约 **20 ms** ✅ | 权重只装一次，43 token 复用 |
| DiT 的 64×6×K32 tile 原型 | **13 GB/s**（更差） | 每 token 单线程，共享读指令淹掉复用 |
| 6-token 的 4 lanes/行（K>2048） | 微基准 45 GB/s，但端到端 DiT 2.47→2.85 s ❌ | 1024 行只剩 16 个 workgroup，填不满 20 SM |
| prefill 按 8 token 分块复用权重 | 0.60–0.98×（更慢）❌ | 43 个 token 本来就能并发，分块反而少并行 |
| QKV 三合一（18 累加器、三输出） | 306 µs vs 分三次 299 µs ❌ | 18 KB 归约占用抵消激活复用 |
| `subgroupShuffleXor` 归约 | 输出错误 ❌ | 设备探针正常，但真实 GEMM 上驱动/编译路径不可用 |
| 每 lane 4 行的寄存器分块 | 4.51 s（+5%） | FMA 总数没变 |
| K 循环 4 路展开 | 无变化 | 不是加载延迟受限 |
| 激活整块进 workgroup 内存 | 9.0 s（更差） | 屏障把加载延迟串行化 |
| 2 行/warp（归约缓冲 12 KB） | 20 GB/s（更差） | occupancy 砍半 |

**关键经验**：Pascal 上 occupancy 与并行 workgroup 数优先；孤立微基准的赢家必须用整句
`MING_PROFILE` 复核（4-lane 的 K=4096 就是这样被否掉的）。

### 0.5.4 还没做 / 下一步（按预期收益排序）

1. **K 分裂（split-K）解决小 workgroup 数**：qkv/o 的 1024×1024 只有 32 个
   workgroup（约 1.6 个/SM）。按 K 切 2–4 份、再做一个融合 bias 的 merge，可提高
   SM 占用；K=4096 的宽层已有 128 个 workgroup，优先级较低。代价是每层多 1 次
   dispatch 和 partial buffer，需整句 A/B。
2. **RMSNorm 融进线性层**：每 block 省 2 次 dispatch 和一次 24 KB 读写；每前向 17 次
   dispatch，约 0.1 ms 量级。
3. **bind group / buffer 复用**：61k buffer/40 步，dispatch 上限 0.48 s，预计还能压
   0.2–0.3 s。
4. **aggregator（t=2）**：目前仍走通用 T4 路径，单次调用占比很小，优先级低于上面三项。
5. **prefill 更深一轮**：当前 tile 只省 20 ms；若把 896×4864 的下投影也做成更多 N tile
   并解决 14-workgroup 欠占用，还有空间。

### 0.5.5 工具、门限、纪律

```bat
cd D:\Ming-3.0-tts\wgpu
:: 带宽基准：改完 kernel 先看这个（t=1 应 >=60 GB/s，t=6 目标 40+）
cargo test --release linear_bandwidth -- --nocapture
:: 数值门限（64 个测试，约 155 s）
cargo test --release -- --test-threads=1
:: 36 例 + 10 例压力集（需要 videolingo 环境 + PYTHONPATH=.runtime\pylibs446）
%PY% wgpu\tools\check_params_align.py
%PY% wgpu\tools\stress_compare.py --skip-ref
```

- **任何性能改动都不许动数值**：64 个测试 + 36 例 sweep 是护栏，红了就是改错了。
- 性能改动必须**背靠背 A/B**（同一分钟内跑两个版本），否则会被热降频骗（4.12）。
- 不要为了让 benchmark 好看去改金标或放宽门限。

### 0.5.7 第三轮：持久 scratch + bind cache（2026-11-01 下午）

**做了什么**

- DiT 与 LLM 的激活改为**整段持久 scratch**。同一 CFM/解码序列里的张量形状固定，
  命令流保证顺序，因此一块 buffer 可重复使用。
- `Kernel` 增加按调用方 key 缓存的 bind group；key 只用于“buffer 集合永不变化”的路径，
  并用命名空间隔离 DiT block / aggregator / LLM，避免共享 `Ops` 时串权重。
- 每次 DiT pair 前向：buffer 申请 **109 → 11**，bind group **156 → 13**。

**同热背靠背（A1 短句，同一分钟）**

| 阶段 | 旧路径 | 新路径 |
|---|---:|---:|
| prefill | 0.45 s | 0.44 s |
| DiT | 3.09 s | **2.53 s** |
| LLM + agg | 1.15 s | **1.04 s** |
| 解码环 | 4.70 s | **3.82 s（−19%）** |

长句（5 s 音频、63 latents）同热 A/B：**9.39 s → 7.59 s（−19%）**；
温度稳定后再测为 6.62–6.75 s。数值：`cfg_pair` 2.9e-7、LLM replay worst 1.5e-6、
64/64 测试通过。

**Python 对照（同机、同文本）**

- `bench_rtf.py`：mean synth **5.11 s / 5.07 s 音频，RTF 1.008**（CUDA Graph 开启）。
- 阶段拆分：Flow **2.92 s**、LLM **1.49 s**、VAE **0.10 s**。
- 关闭 CUDA Graph 后 Python eager 是 **8.26 s**（Flow 5.92），所以当前 Rust
  （7.59 s）已经快过 eager，但 graph 仍快约 1.5×。

**差距在哪**

Python 的 CUDA Graph 把 13 万次 kernel 发射的 GPU 前端成本压到接近零；Rust 即便
把 dispatch 录进同一个 compute pass，每个 dispatch 仍有约 10–17 µs 的前端/调度成本。
下一步不是继续抠算术，而是**合并 dispatch**：qkv 三合一、rope+KV+attention 合一、
matmul 融合 residual/gelu epilogue。CUDA Graph 在 wgpu 没有等价物，不应再尝试用
“异步 submit”绕（已实测更慢）。

> **0.5.8 更正**：上面的"差距在 dispatch 数"判断是错的。0.5.8 用 cuBLAS 分形状实测
> 证明差距在 workgroup 并行度（t=6 只有 128 个 workgroup → 47 GB/s；t=117 的
> 13104 个 → 150 GB/s）。qkv 三合一当时也实测无收益。dispatch 合并保留为
> 第三优先级，split-K 才是第一优先级。

### 0.5.8 第四轮：向量化 + 多行复用 + prefill 分块（2026-11-01 傍晚）

**为什么会找错方向三周**

前三轮一直在"调度开销"上找答案，因为 `linear_bandwidth` 显示 t=6 只有 40–47 GB/s，
而同卡 M=1 GEMV 能到 226 GB/s。这一轮先测了 cuBLAS 的真实形状上限，结论把前面的推断
推翻了：

| 形状（t=6） | 我们（bf16 权重） | cuBLAS fp32 | cuBLAS bf16 |
|---|---:|---:|---:|
| 4096×1024 | 197 µs | **139 µs** | 294 µs |
| 1024×4096 | 208 µs | **156 µs** | 481 µs |
| 1024×1024 | 45 µs | 55 µs | 123 µs |

两个事实：① 这张 Pascal 卡**没有 bf16 原生支持**，cuBLAS 走 fp32 反而比 bf16 快
2–3×（Python 的 DiT 就是 fp32，所以它的 Flow 2.92 s 是算得出来的）；
② cuBLAS fp32 读 8 字节/元素的墙钟比我们读 4 字节还短，说明**它的有效读取带宽
（~120 GB/s）是我们的 2.8 倍**——差距是访存并行度，不是算术、不是 dispatch。

**做了什么**

1. **全向量化六 token kernel**（`shaders::gemv_t6_vec4`）。原来的 kernel 每个 8 元素
   K granule 要发 4 条标量 `u32` 权重加载 + 24 条标量激活加载（28 条加载 / 48 FMA）；
   改成 `vec4<u32>` 权重 + `vec4<f32>` 激活后，配合每 lane 组 2 行，变成 14 条加载 /
   96 FMA，即**每行每 granule 从 28 条降到 7 条加载**。
2. **K 方向展开 2 个 granule**（`v12`）。单个 granule 只有 2 条权重加载在飞，覆盖不了
   ~600 周期延迟；展开后 4 条并发，三个形状从 29/40/45 升到 **47/45/48 GB/s**。
   展开 4 个（`v13`）反而回落（寄存器压力），已实测排除。
3. **单 token 同样处理**（`gemv_t1_vec4`，默认 `8 lanes × 2 rows × 128 threads`）。
   `896×4864` 从 51 → **144 GB/s**，其余两个形状 +4%/+29%。
4. **prefill 按 6 token 分块**（`k_linear_t6win`，带 `n_tok`/`tok0` 的滑动窗口版）。
   通用多 token kernel 是"每 (行块, token) 一个 workgroup"，117 token 的 prompt 会
   把整套权重读 **117 遍**；它跑在 ~150 GB/s（带宽没问题，是冗余），分块后降到 20 遍。
   守卫版会让 DiT 从 3.45 退到 4.03 s，所以**分成两个 kernel**：DiT 用无守卫版，
   prefill 用带守卫版。

**同热背靠背（长句 5.00 s 音频 / 63 latents，t=117）**

| 阶段 | 本轮开始 | 现在 |
|---|---:|---:|
| prefill | 0.61 s | **0.46 s** |
| DiT | 4.29 s（旧标量 8.6 s） | **3.21–3.59 s** |
| LLM + agg | 1.82 s | **1.68–1.85 s** |
| 解码环 | 6.42 s | **5.35–5.91 s** |

`MING_T6=base`（旧标量 kernel）在同一时刻是 11.46 s，即 DiT 单阶段 **2.1×**。

收尾确认（同一二进制连续 3 次，机器已热透）：**5.73 / 5.51 / 5.90 s**
（prefill 0.46–0.47 s，DiT 3.31–3.59 s，LLM+agg 1.72–1.84 s）。对照 Python
CUDA Graph 同文本 5.11 s：**RTF 1.10–1.18 vs 1.008，差 5–15%**。

**验收**

- `cargo test --release -- --test-threads=1`：**64/64**，165 s。
- 压力集（`tools/stress_compare.py`）比文档基线**全面变好**：S6 mel 0.885 → 0.923、
  S8 mel 0.970 → 0.99991、S3 rel 4.7e-1 → 2.9e-2。长单段用例的提升来自 prefill 分块
  改变了 f32 累加分组。

**下一步（按收益排序）**

1. **split-K**：DiT 仍只有 32–128 个 workgroup。t=117 的 prefill 用同族 kernel、
   在 13104 个 workgroup 下跑到 150 GB/s，而 t=6 的 128 个 workgroup 只有 47 GB/s
   ——**并行度就是那个 3× 的缺口**。按 K 切 2–4 份再融合 bias，是现在最明确的一条。
2. **prefill 再用真 GEMM 收一次**：分块把冗余从 117 降到 20，但每个分块的 workgroup
   数仍然偏低（28–152），实际只有 ~32 GB/s；要做到 150 GB/s + 1 遍读，需要
   48–128 token × 64 行的 tile（现有 `GEMM_PREFILL_48X64` 实测无效，得重写）。
3. **合并 dispatch**（gelu 融进 ff_in epilogue、add 融进 rms）：每 block 省 2 次，
   每次 6-token 前向约 16 次。

### 0.5.6 关于"能不能到 0.88"

上一轮"DiT 屋顶 34–40 GB/s、0.88 到不了"的结论早已作废。本轮 t=6 已实测
**45–48 GB/s**（展开后），DiT 3.2–3.6 s，已经和文档里 Python 的 2.92 s 同量级。
**0.88 仍未达成，但差距已经从"看不到路"变成具体的 split-K 并行度问题。**

### 0.9 设备选择修正（2026-11-03）：以上性能数字全部来自核显

`--device` 之前只是参数面占位（`main.rs` 里 `let _ = device;`），而 `Gpu::new`
取的是**第一个枚举到的 Vulkan 适配器**，本机是 Intel 核显。所以 0.5 / 0.5.7 /
0.5.8 / 4.5 里的 Rust 数字，以及"P104-100 / cuBLAS 2.8× / 差距是 workgroup
并行度 / split-K 第一优先级"这一整套分析，都是在核显上得到的；拿来对比的
`cuBLAS fp32 120 GB/s`、`M=1 GEMV 226–268 GB/s`、`150 GB/s prefill` 则来自
P104（兄弟项目），**两边不是同一台设备**。

现在 `--device` 真正生效：`cuda:0`→独显（默认）、`igpu`→核显、`cpu`→软件
适配器（WARP/lavapipe）。同一份 `linear_bandwidth`：

| t=6 形状 | Intel 核显 | P104-100 |
|---|---:|---:|
| 4096×1024 | 46 GB/s | **103 GB/s** |
| 1024×4096 | 32 GB/s | **140 GB/s** |
| 1024×1024 | 50 GB/s | **89 GB/s** |

长句（5.00 s 音频、63 latents、带 prompt）端到端实测：

| 设备 | 解码环 | VAE+iSTFT | 合计 | RTF |
|---|---:|---:|---:|---:|
| Intel 核显 | 4.63 s | 0.80 s | 5.44 s | 1.09 |
| P104-100 | 2.96–3.06 s | 0.50 s | **3.46–3.56 s** | **0.70** |
| Python CUDA Graph | — | — | 3.91 s | 0.78 |

P104 上带 VAE encoder 的路径原先必崩（`Parent device is lost`）：单个 submit
里积到 1024+ dispatch 时，驱动会在这个 submit 里才编 SASS，触发 Windows TDR。
现在**所有设备**默认 `MING_MAX_PENDING=256`（单次 submit 保持在 TDR 窗口内，与厂商无关；
数值与 2048 **逐位一致**，已实测），崩溃消失。

跨设备数值（同一二进制、同一 seed）：A1 无 prompt 两卡 **rel_l2 5.96e-5**
（采样点与时长完全一致，逐秒频谱相关 1.000000），长句 7.8e-4——听感一致，
逐位不一致（驱动超越函数/FMA 收缩，见 4.6）。**但改 kernel 归约序就不是这个
量级了**：把 encoder 里 3528 宽的 fc1 从通用路径换成分块路径（数值 1.3e-6），
prompt latent 变 ~1e-6 后整句走到另一条合法轨迹（rel_l2 ~1.0、RMS 0.047→0.028）。
所以"对齐"只能在同一份二进制内讨论，验收 sweep 也必须同版本比。

### 0.9.1 大批量线性层：t12 默认、48×64 tile 可选（2026-11-03）

t>8 的线性层（VAE 编码/解码、长 prefill）原来一律按 6-token 窗口走
`gemv_t6_vec4`，t=252 的 VAE 解码要把每层权重读 42 遍。两条候选路线：

| 方案 | P104 prefill (t=117) | P104 VAE (252 帧) | 核显 prefill | 核显 VAE |
|---|---:|---:|---:|---:|
| 6-token（原） | 0.233 s | 0.50 s | 0.44 s | 0.80 s |
| 12-token 窗口（核显默认） | 0.197 s | 0.45 s | 0.40 s | 0.80 s |
| 48×64 tile（**独显默认**） | **0.102 s** | **0.23 s** | ❌ 0.66 s | ❌ 1.30 s |

结论：两条路径按**适配器类别 + 是否回流**选。tile 的 K 归约序和窗口 kernel 不同，
**只要它的输出会回到自回归环（prefill、VAE 编码），停止步就可能翻**（实测 S3/L=8.7 s：
tile prefill 下 dur 0.9885，窗口下 1.0000）。所以默认只让 **VAE 解码器**（输出直接
变成波形、不回流）走 tile；prefill/编码器一律窗口——这条会改轨迹的实验路径已经删除。
12-token 窗口则是**逐位中性**的（同一输入下 tok=6 与 tok=12 的 data 块字节相同），
所以它可以在所有路径上默认开。
tile 微内核仍值得重写（sb 用 vec4 共享布局、减少每 K tile 的 16 次标量 shared load），
重写后核显也能吃到它的收益。

长句端到端（5.00 s 音频、63 latents、带 prompt）：
- **P104（默认：decoder tile + 窗口 prefill）**：prefill 0.197 + DiT 1.89 + LLM/agg 0.66
  + VAE 0.25 + iSTFT 0.02 ≈ **3.00 s → RTF 0.60**（两次复测 2.746 s 解码环）；
- P104 强制窗口（`MING_PREFILL=base`）：3.47 s / 0.69；
- 核显（窗口）：5.2–5.7 s / 1.04–1.14。

**0.9.3 36 例的数字是核显口径；dGPU 上 F2 会翻 1 个 latent**

同一份二进制、同一 seed：F2_korean 在核显上 dur 1.0000 / rel 0.322 / mel 0.99779
（= 4.2 表），在 P104 上 dur 1.0333（38 vs 39 latents）/ rel 0.448 / mel 0.9868。
用**原始内核**（`MING_T12=0 MING_PREFILL=base`）和 `MING_IN_FLIGHT=1` 复测，
P104 仍是 1.0333 —— 所以这是“混沌轨迹 + 设备级超越函数差异”在停止判据上的
体现，不是本轮改动引入的。结论：**4.2 的 “36/36 时长一致” 是核显结论**；
dGPU 上一律是 35/36（F2 例外），A4/I4 另属 cache-free 参考口径（见 4.2）。

**0.9.2 提交重叠（2026-11-03）**：`flush()` 原来每提交 256 个 dispatch 就
`poll(wait)` 一次，GPU 每个批次都要排空等 CPU 录制。现在只保留“在飞提交”上限
（`MING_IN_FLIGHT`，默认 16）和 512 MiB 字节上限，读回时才阻塞；`flush_wait()`
留给计时基准。效果：DiT 2.05 → **1.88 s**、解码环 2.82 → **2.62 s**、整句
3.05 → **2.85 s**。**数值不受影响**：同一二进制 `MING_IN_FLIGHT=1/16` 的
音频采样**逐位相同**（只有 WAV 里 PEAK 块的 1 字节时间戳不同），64/64 测试通过。

### 0.10 CPU 后端（2026-11-03，手写 twin）

--device cpu 以前走 wgpu 的 DX12 WARP，能跑但**输出全是 NaN**，而且比原版慢约 70×。
现在 CPU 走的是**手写 Rust 后端**（src/cpu.rs，思路同 qwen3-asr-wgpu/src/cpu_decoder.rs）：
f32 激活 + bf16 权重加载时展宽 + rayon 行并行 + 手写分块 GEMM，没有 NN 框架、不碰 wgpu。

覆盖：Qwen2.5-0.5B（prefill + 增量解码 + stop head）、DiT+CFM（8 block、CFG 合批、10 步
Euler）、aggregator、VAE 解码器（24 层因果滑窗 64 + iSTFT）、VAE 编码器（24 层 + patch/cls
+ 4 层 aggregator + fc3 → mean/std）、spk_head，以及克隆条件（prompt 编码 + 后验抽样 +
linear_proj_audio 逐 patch 投影）。

逐模块对拍（全部对着 Python 金标，CPU 与 GPU 同为 1e-6 量级）：

| 阶段 | CPU 实测 | 门限 |
|---|---:|---:|
| LLM 40 步重放 worst z | **1.82e-6**（stop step 40 一致） | 1e-5 |
| DiT block0 逐算子 | **2.2e-7 – 9.0e-7** | 2e-6 |
| CFM 41 步 worst | **4.92e-6** | 1e-5 |
| VAE 解码波形 | **2.39e-6** | 1e-4 |
| VAE 编码 mean / std(abs) / latent | **1.35e-6 / 6.8e-8 / 1.35e-6** | 1e-4 |
| 克隆条件 embeds（CPU vs GPU） | **2.0e-6**（最差行 1.1e-5） | — |
| 端到端 A1（对 params_test 参考） | 时长 1.0000、rel 1.79e-4、**log-mel 1.000000** | — |
| 端到端 A3 克隆（对参考） | 时长 1.0312、mel 0.9136、音色 cos 0.9606 —— 条件已对齐到 2e-6，
但 40 步自回归把 2e-6 放大成另一条合法轨迹（4.4 的混沌类） | — |

性能（Ultra 7 265K，20 核）：3.2 s 音频 **模型加载 3.0 s + 解码 6.28 s + VAE 0.69 s ≈ 7.0 s
（RTF 2.2）**；同一句 Python CPU 是 **12.8–16.3 s（RTF 4.0–5.3）**，即 CPU 路径约 **2× 于原版**。

STYLE 多分片的首片重条件也已补齐（第 0 片生成后重新过 CAMPlus + VAE 编码，再作为后续
分片的参考），CPU 上 G2_empress 实测 **时长 1.0000 / rel 0.321 / mel 0.99733 / 音色 cos 0.9975**
（GPU 基线 0.29 / 0.9991 / 0.9994，同一混沌类）。至此 --device cpu 与 GPU 路径的功能面一致。

## 1. 现状

| 验收项 | 状态 | 证据 |
|---|---|---|
| M0 骨架（设备 / bf16 权重 / 内核） | ✅ | rmsnorm 1.0e-7、gemv 3.0e-7 |
| M1 LLM（prefill + KV 解码 + 停止头） | ✅ | 真实 40 步重放：z 8.1e-7、停止步一致 |
| M2 DiT + CFM 采样 | ✅ | 41 步对拍 worst **9.6e-7**、y 链 **3.0e-7**（2026-11-01 修好 DiT rope 之后；此前是 1.28e-4 / 5.8e-5） |
| M3 VAE 编码（波形→latent） | ✅ | mean 1.4e-6 / std 6e-8(abs) / latent 1.4e-6 |
| M4 VAE 解码 + iSTFT | ✅ | 最终波形 1.9e-6，长度逐采样点一致 |
| M5 文本前端 | ✅ | 695 条语料 / 10425 个字段、36/36 request dict 逐字节 |
| M5 音频 / 分词 / 说话人 | ✅ | wav 逐字节、resample 4.5e-8、tokenizer 946 token、CAMPlus 1.5e-5 |
| M5 解码环 | ✅ | 步数与参考一致，z 逐步 6e-7 → 1e-3 |
| M5 RNG（torch 对齐） | ✅ | 28160 个抽样值最大误差 1.4e-6 |
| M5 36 个参数用例 | ✅ 跑通 | **36/36 时长一致**；26 例波形级重合（rel_l2 ≤5e-2），10 例后段相位差（log-mel ≥0.9959）；中位 log-mel 1.00000 |
| 长文本 / 长音频压力集 | ✅ 10 例 | 分片路径精确（21.6 s、6 片，rel 1.3e-2）；单段 >6 s 会分岔（4.7）；挖出并修掉 3 个越界 bug（4.8） |
| CLI 参数面 = infer.py | ✅ | `rename_all = snake_case`，逐 flag 对齐 |
| **RTF** | ❌ | **1.16**（该数字来自核显，且不含 VAE；见 0.9：P104 端到端 **0.70**，Python 同文本 **0.78**） |

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
| `MING_MEM=1` | 打印每阶段的显存申请量（累计 + flush 窗口峰值），见 4.9 |
| `MING_NOISE_GOLDEN=<dir>` | 注入金标 torch.randn 抽样（`goldens/m5/` 或 `goldens/m5_params/<case>/`），用于逐样本对拍 |
| `MING_DUMP_IDS=<file>` | 把首条分片的 token id 写盘，用于和参考的 `build_input_ids` 比 |
| `MING_MAX_PENDING=<n>` | 一次提交的 dispatch 上限（默认 2048） |
| `MING_T6=base\|spec\|v2\|v3\|v4\|v5` | 回退到旧的标量六 token kernel（A/B 用） |
| `MING_T6=v8\|v9\|v10\|v11\|v12\|v13\|v14` | 向量化 kernel 的行宽/行数/展开组合，默认 `v12`（8 lanes × 2 行 × 展开 2） |
| `MING_T1=base\|half` | 回退单 token 路径（默认是向量化 2 行版；`half` 是旧的半 warp 版） |
| `MING_T1=v1\|v2\|v3\|v4` | 单 token 变体，(8,64,1)/(8,128,1)/(8,128,2)/(16,128,1) |
| `MING_PREFILL=base` | 关掉 decoder 的 48×64 tile（默认随适配器类别：独显开、核显/CPU 关），见 0.9.1 |
| `MING_T12=0` | 关闭 t>8 的 12-token 窗口，回到 6-token（A/B 用） |
| `MING_IN_FLIGHT=<n>` | 允许的在飞 submit 数（默认 16；=1 表示每批都 wait，等价旧行为） |
| `MING_STOP_TRACE=1` | 每步打印停止概率（排查时长差的唯一可靠手段） |
| `MING_ALIGN_DEVICE=igpu` | `tools/check_params_align.py` 钉设备（默认 cuda:0） |
| `MING_DEVICE=cuda:0\|igpu\|cpu` | 测试/基准里钉设备（CLI 用 `--device`） |
| `MING_LM_PREFILL_PROFILE=1` | 打印 prefill 单层分段耗时（只打印一次） |
| `MING_SCRATCH=0` | 关闭 DiT 持久 scratch/缓存快路径（A/B 用） |
| `MING_LM_FAST=0` | 关闭 LLM 持久 scratch/缓存快路径（A/B 用） |
| `MING_BIND_CACHE=0` | 关闭 bind group 缓存（定位数值问题用） |
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
| `src/parity.rs` | parity 测试（`#[cfg(test)]`） | 与各模块内测试合计 56 个 |

## 4. 验收数据

### 4.1 逐模块数值对齐（对 Python 金标）

| 阶段 | 指标 | 实测 |
|---|---|---|
| LLM 40 步重放 | z rel_l2 | 8.1e-7，停止步 40 = Python |
| DiT 单前向 | rel_l2 | **1.6e-7**（= 单个 GEMM 的底噪；rope 修好前是 2.6e-5） |
| DiT 单 block 逐算子 | rel_l2 | norm1/q/k/v 1-2e-7，**q_rope 1.9e-7**，attn 4.2e-7，out 1.0e-7 |
| CFM 41 步 | worst rel_l2 | **9.6e-7**（修好前 1.28e-4），y 链 **3.0e-7** |
| VAE 编码 | mean / std / latent | 1.4e-6 / 6e-8(abs) / 1.4e-6 |
| VAE 解码 | 最终波形 | 1.9e-6 |
| 克隆条件 | prompt latent / spk_head | 2.2e-6 / 1.2e-7 |
| torch RNG | 28160 个抽样值 | 1.4e-6 |

### 4.2 36 个 params_test 用例

A4 / I1-I4 的参考音频是批量脚本**复用缓存 prompt 编码**产出的（不消耗后验抽样），
单进程 CLI 复现不了；用 `tools/ref_nocache.py` 重渲染后对比（表里标 cache-free）。

| 用例 | 时长比 | 波形 rel_l2 | log-mel 相关 | 音色 cos | 备注 |
|---|---|---|---|---|---|
| A1_bare | 1.0000 | 3.6e-05 | 1.00000 | 1.0000 |  |
| A2_zero_spk | 1.0000 | 2.8e-04 | 1.00000 | 1.0000 |  |
| A3_clone | 1.0000 | 2.6e-03 | 1.00000 | 1.0000 | 修好前 1.03 |
| A4_clone_latent_only | 1.0000 | 1.9e-03 | 1.00000 | 1.0000 | cache-free 参考 |
| B1_speed_slow | 1.0000 | 4.9e-02 | 0.99994 | 0.9999 |  |
| B2_speed_mid | 1.0000 | 8.4e-01 | 0.97057 | 0.9731 | 后段分岔（本表最难一例） |
| B3_speed_fast | 1.0000 | 1.2e-03 | 1.00000 | 1.0000 |  |
| B4_pitch_low | 1.0000 | 3.1e-03 | 1.00000 | 1.0000 | 修好前 6.38e-01 |
| B5_pitch_high | 1.0000 | 4.3e-03 | 1.00000 | 1.0000 |  |
| B6_volume_low | 1.0000 | 5.2e-02 | 0.99977 | 0.9999 | 修好前 1.20 |
| B7_volume_high | 1.0000 | 6.6e-04 | 1.00000 | 1.0000 |  |
| C1_happy | 1.0000 | 6.0e-01 | 0.99900 | 0.9974 | 后段分岔 |
| C2_sad | 1.0000 | 1.5e-01 | 0.99980 | 0.9993 | 后段分岔 |
| C3_angry | 1.0000 | 8.9e-03 | 1.00000 | 1.0000 |  |
| C4_fearful | 1.0000 | 4.6e-01 | 0.99892 | 0.9983 | 后段分岔 |
| C5_disgusted | 1.0000 | 1.3e-03 | 1.00000 | 1.0000 |  |
| C6_surprised | 1.0000 | 9.2e-01 | 0.99785 | 0.9959 | 后段分岔（用户点名的用例） |
| C7_neutral | 1.0000 | 2.2e-01 | 0.99894 | 0.9973 | 修好前时长差 1 步（1.0400），现在一致 |
| D1_cantonese | 1.0000 | 8.2e-03 | 1.00000 | 1.0000 |  |
| D2_sichuanese | 1.0000 | 2.4e-02 | 0.99997 | 0.9999 |  |
| E1_ip_lingyuechuan | 1.0000 | 5.9e-04 | 1.00000 | 1.0000 |  |
| E2_ip_lingxiyue_happy | 1.0000 | 1.2e-01 | 0.99968 | 0.9996 | 后段分岔 |
| E3_ip_us_emma | 1.0000 | 1.3e-03 | 1.00000 | 1.0000 |  |
| F1_japanese | 1.0000 | 4.5e-04 | 1.00000 | 1.0000 |  |
| F2_korean | 1.0000 | 3.4e-01 | 0.99776 | 0.9914 | 后段分岔 |
| F3_german | 1.0000 | 1.6e-02 | 0.99997 | 1.0000 |  |
| F4_french | 1.0000 | 9.1e-03 | 0.99997 | 1.0000 |  |
| F5_spanish | 1.0000 | 6.5e-04 | 1.00000 | 1.0000 |  |
| G1_asmr_whisper | 1.0000 | 1.7e-04 | 1.00000 | 1.0000 |  |
| G2_empress | 1.0000 | 2.9e-01 | 0.99847 | 0.9973 | 修好前时长差 1 步；现在一致 |
| H1_whisper_secret | 1.0000 | 4.6e-01 | 0.99587 | 0.9941 | 修好前时长差 1 步；现在一致 |
| H2_news_anchor | 1.0000 | 3.3e-04 | 1.00000 | 1.0000 |  |
| I1_temperature_06 | 1.0000 | 4.1e-05 | 1.00000 | 1.0000 | cache-free 参考 |
| I2_temperature_10 | 1.0000 | 4.7e-04 | 1.00000 | 1.0000 | cache-free 参考 |
| I3_cfg_10 | 1.0000 | 7.6e-05 | 1.00000 | 1.0000 | cache-free 参考 |
| I4_cfg_35 | 1.0000 | 7.4e-04 | 1.00000 | 1.0000 | cache-free 参考 |

> 汇总（2026-11-01，DiT rope 修好之后）：**36/36 时长逐采样点一致**；
> **26 例波形级重合**（rel_l2 ≤ 5e-2，最好 3.6e-5）；剩下 **10 例**是后段相位差
> （rel_l2 0.12–0.92，log-mel 中位 1.00000、最差 0.9959，B2 例外 0.9706）——
> 前 1/3 逐帧几乎相同，差异随时间单调增大到段尾（见 4.4）。
> 对照修好前：只有 15 例 rel_l2 < 0.15、10 例 > 0.3，3 例连时长都对不上（C7/G2/H1）。

### 4.3 音色（克隆）链路

| A | B | CAMPlus cos |
|---|---|---|
| 参考 A1（无提示） | rust A1 | **1.0000** |
| 参考 A3（Python 克隆） | rust A3 | **0.9685** |
| 提示音 lingguang | 参考 A3 | 0.8019 |
| 提示音 lingguang | rust A3 | 0.8013 |

### 4.4 为什么做不到逐采样点一致（量化过，别再试）

测试 `m5_chaos_tests::loop_is_chaotic`：把同一份输入乘 (1+1e-7)，我们**自己的**
两次运行 40 步后 `z` 的相对差就到 **1.5e-2**，放大约 1e5。现在移植版每步误差
~1e-6（f32 GEMM 归约序的底噪，见 4.6），从 1e-6 放大到 O(1) 只要 ~15 步，
所以 36 例里必然有几例在后段分岔——**这不是"哪里写错了"**。

证据（2026-11-01 修好 DiT rope 之后重新量的）：

- C6 逐步：step0 `z` 1.15e-6 → step3 6.6e-5 → step11 5.2e-4 → step24 2.4e-1。
- 分岔的形状：把 10 例的 log-mel 帧对齐后按时间三段统计平均绝对差，第一段
  0.00–0.03（几乎逐帧相同）、第二段 0.02–0.67、第三段 0.15–1.16，峰值一律落在
  85%–99% 处——**同一句话、同一时长，只有后段的相位/细节不同**，没有局部热点，
  所以不是缺陷。
- 修好 rope 之前每步误差 ~1e-4，那时是 10 例 rel_l2 > 0.3、3 例连停止步都差 1 步；
  修好之后 36/36 时长一致，高差异只剩 10 例。

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

那"单算子 1.6e-7 → 单次前向 1.6e-7"为什么不再放大？（**2026-11-01 更正**：这里原本写的是
"单算子 3e-7 → block 输出 2.35e-5，4096 宽 MLP 把 ulp 级差异放大"——那个结论是错的。）
修好 DiT 的 rope 之后逐算子重测：norm1 1.3e-7 / q 1.8e-7 / **q_rope 1.9e-7** /
attn 4.2e-7 / norm2 1.8e-7 / ff_out 1.8e-7 / **out 1.0e-7**：一个 block 的输出误差和
它的输入同一量级，没有放大。之前的 2.35e-5 全部来自 rope 的 base 猜错（见坑清单 13）——
1.4e-4 的角频率偏差经 q/k 放大 450 倍进 attention，再被后续层稀释成 2.35e-5。

GELU 用 erf 而不是 tanh 那件事仍然成立（每元素差 1e-3，是真实的公式差异），只是当时
被更大的 rope 误差盖住了；两个都修好之后 block0 是 1.05e-7。

要做到逐位一致，必须同时满足：

1. 每个 GEMM 的归约树与 cuBLAS 逐位相同——算法未公开，且随 shape / 架构变化；
2. 每个超越函数（exp / tanh / sin）与 CUDA libdevice 逐位相同——**WGSL 的 exp/tanh/sin
   由 Vulkan 驱动实现，我们无法指定、也无法得知它用哪个近似**；
3. softmax / RMSNorm 的归约同理。

这不是"努力不够"，是**在 wgpu 层面拿不到那个控制权**。同一台设备确实消除了硬件差异
（我们自己的两次运行逐位一致，否则测试早红了），但消除不了"与另一套实现逐位相同"
所需的顺序与超越函数信息。

> 结论：能保证的是**每步 ~1e-6 量级数值一致 + 停止判据一致 + 音色一致 + 内容一致**；
> 逐采样点一致需要复刻 cuBLAS 与 CUDA 数学库的逐位行为，不在可及范围内。
> 剩下的 10 例后段分岔就来自这里（4.4），**不再当作待办**：每步误差已经从 1e-4 压到
> 1e-6（100 倍），再往下就是逐位复刻，不可达。

### 4.7 覆盖矩阵：36 例验收集没覆盖什么

36 个 params_test 用例覆盖的是**内容面**（8 种 task / 7 种语言 / 方言 / 情感 / 音色 / style / instruction / cfg / temperature / 克隆），但结构面基本是空的：**35/36 是单分片，文本 10–62 字，音频 0.9–7 s，全是干净文本**。也就是说"模型说得对不对"验过了，"说很长一段"从来没验过。

| 维度 | 36 例验收集 | 2026-11-01 补的压力集 |
|---|---|---|
| task | 8/8 | — |
| 语言 | zh/en/ja/ko/de/fr/es + 粤语/川渝 | 英文长文（266 字符） |
| 文本长度 | 10–62 字 | 2 / 6 / 45 / 75 / 207 字 / 266 字符英文 |
| 音频时长 | 0.9–7 s | 1.1 / 5.3 / 8.7 / 11.8 / 21.6 / 23.2 s |
| 分片 | 只有 G2（STYLE, 2 片） | 非 STYLE 2 片 / 6 片 + `--raw_text` |
| 数字/符号/混排 | 无 | `GPT-4 / 3.5 / A/B / 12%` 混排 |
| 文本规范化 ON | 未测（本机跑不了，见本节末） | 仍缺 |
| 参考音频长度 | lingguang（~5 s） | 12 s 参考（编码器） |

压力集 = `tools/stress_compare.py`（10 例，报告在 `output/stress/report.json`）+ 4 个新单测。最终数字：

| 用例 | 时长比 | rel_l2 | log-mel | 判读 |
|---|---|---|---|---|
| S1 嗯。 | 1.0000 | 1.8e-04 | 1.00000 | 极短文本 ✓ |
| S2 好的，谢谢。 | 1.0000 | 4.7e-03 | 0.99999 | ✓ |
| S4 60 字分 2 片 | 1.0000 | 4.3e-03 | 1.00000 | **分片路径精确** |
| S5 150 字分 6 片（21.6 s） | 1.0000 | 1.3e-02 | 0.99985 | 分片路径与长度无关 |
| S9 中英混排 | 1.0000 | 4.4e-02 | 0.99994 | ✓ |
| S3 / S10 60 字**单段** | 1.0115 | 7.4e-01 | 0.94452 | 第 30 步起分岔，停止步差 1 |
| S8 266 字符英文单段 | 1.0085 | 2.9e-01 | 0.97045 | 前 87% 逐窗口重合 |
| S6 207 字单段（23.2 s） | 1.0000 | 6.3e-01 | 0.88484 | 前 26% 逐窗口重合 |

**结论**：长文本不会"崩"（越界那类错误已经修掉，见 4.8），但**单段越长越会跑到另一条合法轨迹上**——这就是 4.4 的放大，只是窗口更长。**分片路径是精确的**，想逐点对齐就用默认 `--max_length 24` 让它分片；`--raw_text` 只适合短句。

### 4.8 长文本压力集挖出来的三个越界 bug（2026-11-01，已修）

36 例一直全绿，而 >6.4 s 的音频一直是坏的——因为没有任何验收样本跨过这条线。

1. **VAE 解码器/编码器复用了 LLM 的 512 位置窗口。** `vae_dec::stack` / `vae::stack` 把整段音频一次性写进 `KV_CAP = 512` 的 K/V 缓冲，rope 表也是 512 个位置。于是 >6.4 s 音频（解码）/>10.2 s 参考音频（编码）就是**越界写**。wgpu 不做边界检查，所以既不报错也不崩，只是音质坏掉：S6 的 log-mel 0.82、音色 cos 0.79，而且**从第一个窗口就开始坏**。修法：新增 `ops::KvCache`，按实际长度 `ensure()` 扩容（rope 表一起重建，floor 仍是 512）。
2. **ATTEND 的 workgroup scratch 用绝对位置索引。** `sc[pos]` / `wgt[pos]` 只有 512 槽，滑动窗口下 pos 会跑到 640+，同样越界——而且写的是 workgroup 共享内存，踩了就整组错。修法：改成 `sc[pos - lo]`：窗口内最多 window 个活跃位置，索引恒小于 512；`window = 0`（LLM/DiT）时 lo = 0，索引与原来完全一致，所以**36 例的数字一个没变**。
3. **`(512 - t)` 的 usize 下溢。** `--raw_text` 把长篇一次送进去时 t 可以 >512，预算回绕成天文数字，于是继续往 KV 里写到 512 之后。修法：`saturating_sub` + 明确报错（原版在这种情况下返回空音频）。

修完的验证：`vae_decodes_past_the_llm_window`（160 latents = 640 帧 = 12.8 s，门限 1e-4，实测 **2.9e-6**）、`vae_encodes_past_the_llm_window`（12 s 参考 = 600 帧，实测 **1.2e-6**）、`long_prompt_prefill_matches_python`（65 / 88 / 161 token 的 prefill，实测 7e-7 / 8e-7 / 1e-6）。

> **文本规范化（TN）的现状**：参考实现走 pynini/WeTextProcessing 的 FST，而本机 videolingo 环境里**没有 pynini**（RUN_LOCAL.md 第 13 行），所以 `infer.py` 只要不带 `--no_text_normalization` 就会在第一个分片上抛 `ModuleNotFoundError`；36 例和本文所有对拍都是带 `--no_text_normalization` 跑的。移植版实现的是 pynini 之后的那一步 `normalize_text_kit`，所以**带数字/符号的文本一旦开 TN，两边没有可比的基准**——这是 P2 的功能缺口，不是回归。

### 4.9 显存随生成时长怎么涨（量过，别再猜）

`MING_MEM=1` 会打印每个阶段向 `device.create_buffer` 申请了多少字节，以及一次 flush 窗口内的峰值（flush 之后上一批 recording 才能回收，所以窗口峰值 ≈ 该阶段的显存峰值）。单段实测（P104-100，权重常驻 1782 MiB）：

| 场景 | 采样阶段峰值 | 采样阶段累计申请 | VAE 解码累计 | **整体峰值** |
|---|---|---|---|---|
| 3.2 s 单段（40 latents） | 1869 MiB | 1603 MiB | 328 MiB | **1869 MiB** |
| 8.8 s 单段（110） | 1913 MiB | 4307 MiB | 902 MiB | **1913 MiB** |
| 23.2 s 单段（291） | 2107 MiB | 11377 MiB | 2390 MiB | **2373 MiB** |
| 21.6 s **分 6 片** | 1873 MiB | 每片 1.3–2.6 GiB | 每片 262–550 MiB | **1873 MiB** |

两条完全不同的规律：

1. **LLM + DiT 采样阶段：显存不随时长增长。** 每个 decode step 会申请约 39 MiB 临时 buffer（291 步 = 11 GiB 累计申请），但它们活到下一次 `flush()` 就被回收；flush 每 2048 个 dispatch（约 3 步）发生一次，所以窗口峰值恒定在 ~1.9 GB。**时长只影响累计申请量（≈39 MiB/latent ≈ 0.49 GB/秒音频），不影响驻留量。**

2. **VAE 解码阶段：显存随音频时长线性增长。** 解码器是“整段一次过”（与参考实现一致），24 层的中间 buffer 全部在同一次 flush 之前申请完，所以 ≈ **2.05 MiB/帧 = 102 MiB/秒音频**，一帧都不回收。23.2 s 单段就是 2.39 GB —— 这一阶段的峰值第一次超过了采样窗口。

两条线的交点约 **18.6 s**：短于它，峰值 1.9 GB（权重 + 采样窗口）；长于它，峰值 ≈ 0.102 GB × 音频秒数。单段最多受 512-token 预算限制在 ~37 s，对应 ~3.8 GB，8 GB 卡够用；**分片能把它压平**（21.6 s 分 6 片，每片单独解码，峰值 1873 MiB，与 3.2 s 单段相同）——这是默认 `max_length=24` 的另一个好处。

给 RTF / 批处理优化的提示：pred/null 合批不会让驻留量明显上涨（窗口仍由 flush 控制）；但如果要改“一次 flush 里塞更多 dispatch”，窗口峰值会跟着涨。想让长单段的 VAE 峰值消失，得做分块解码（参考实现的 streaming `use_cache=True` 路径，用 `stream_state` 复用窗口）。

### 4.10 一次性能喂多长的文本

硬约束只有一条（来自参考实现的静态 KV cache）：**prompt tokens `t` + 生成的 latents ≤ 512**，即 `latents ≤ 512 - t`，每个 latent = 80 ms。中文 `t ≈ 33 + 0.58 × 字数`，英文 `t ≈ 33 + 0.19 × 字符数`（tokenizer 实测，含模板）。

| 文本 | t | 预算 latents | 预算时长 | 实测结果 |
|---|---|---|---|---|
| 207 汉字 | 161 | 351 | 28.1 s | 291 latents / **23.2 s**，`stopped_at=Some` 正常收尾 |
| 236 汉字 | ~170 | 342 | 27.4 s | 315 latents / **25.2 s**，正常收尾 |
| 290 汉字 | 215 | 297 | 23.8 s | 297 latents / **23.7 s，`stopped_at=None`：撞预算截断，话没说完** |
| 290 汉字（默认分片） | 每片 ~45 | 每片 ~460 | — | **13 片 / 50.2 s，13 片全部 `stopped_at=Some`** |

所以：

- **一片的安全区 ≈ 200 汉字（≈25 s 音频）/ 约 500 字符英文**。再长就进入"预算比话短"的区间，会在半句处截断（不报错，`stopped_at=None`）。

- 预算归零的位置：中文约 **830 字**、英文约 **2600 字符**——到那里会直接报错（4.8 第 3 条的 saturating 保护）。

- **默认分片（`max_length=24`）没有上限**：290 字 → 13 片 50.2 s；2000 字 → 94 片。这是推荐路径，也是 4.9 里显存/质量都最稳的路径。

- `--raw_text` 等于关掉分片，只适合短句；长文用它就会遇到上面的截断。

- STYLE 任务分片时会拿第 0 片重新做参考（`reference_first_fragment`），后面每片的 `t` 会额外带上第 0 片的 latent 行数（几十个），预算相应再少一点。

### 4.11 GEMM 屋顶线：DiT 为什么卡在 24 GB/s（2026-11-01 量）

> **⚠️ 本节结论已作废，只保留测量数据。** 当时把"我们 kernel 的发出率屋顶"当成了
> "这张卡的屋顶"。本轮已用**配对 `vec2` 激活 + 按 K 选择 lanes/行**把 M=6 做到
> **40–47 GB/s**，DiT 2.46 s，见 0.5 节。剩余差距不是算术量，而是 split-K 并行度
> 和 dispatch/内存调度。
同类移植项目的实测（`D:\qwen3-asr-wgpu\docs\perf.md`，同一张 P104-100）：`t_us = 5.7 + MB×1000/268`，即 **M=1 GEMV 能到 226-268 GB/s**。但那是单 token 的形状。DiT 的 CFG 合批是 **M=6**，每个权重 granule（8 个 bf16）要配 12 个激活 `vec4` + 48 个 FMA：
| 形态 | 每 granule 指令 | 实测带宽 |
|---|---|---|
| M=1（LLM 解码） | 1 权重 + 2 激活 + 8 FMA ≈ 11 | **62-65 GB/s** |
| M=6（DiT 合批） | 1 权重 + 12 激活 + 48 FMA ≈ 69 | **24 GB/s** |
48 个 FMA 是**不可压缩**的（6 token × 8 值，激活必须 f32）。按 20 SM × 4 scheduler × 1.6 GHz 估，524k 个 granule 的*指令发出*下限约 250 µs，实测 348 µs —— 说明"每 warp 一行"这个形态已经接近自己的发出率上限。

**但这只是这种写法的上限，不是这张卡的上限**：cuBLAS 在同样的卡、同样 6 token 上做到 ~37 GB/s（即每个 granule 用更少的指令，靠寄存器分块摊薄激活读和循环开销）。要突破必须换形态（0.5.4 第 1 条）。
已试过并排除的：寄存器分块 4 行/lane（4.51 s，指令没变少）、K 循环 4 路展开（无变化）、激活整块进共享内存（9.0 s，屏障串行化）、2 行/warp（20 GB/s，归约缓冲 12 KB 把 occupancy 砍半）。最后一条和 ctc 移植项目的 tile 扫描结论一致：**Pascal 上 occupancy 优先于预取**。
### 4.12 计时会在热机后漂移 ~18%（做 A/B 时注意）
同一份二进制、同一句话，一小时前 DiT 是 4.84 s，连跑数小时压测后变成 5.65 s（两轮复测一致）。P104-100 是被动散热，长时间满载会降频。
**所以任何 kernel A/B 都必须背靠背进行**（同一分钟内交替跑两个版本，或至少用 `linear_bandwidth` 这种短基准做相对比较），不要拿隔了半小时的绝对数字比。

### 4.5 性能（release，P104-100）

> **4.5 的表是历史快照，只保留阶段结构。最新数字见 0.5.8**（长句 63 latents：
> prefill 0.46 s / DiT 3.2–3.6 s / LLM+agg 1.7–1.9 s，整句 5.35–5.91 s）。

| 阶段 | 耗时（历史） |
|---|---|
| prefill | 0.38 s |
| DiT + CFM | 2.46 s |
| LLM 解码 + aggregator | 0.89 s |
| 合计（解码环） | 3.72 s |
| 模型加载 + VAE 解码 | ~5.7 s（一次性） |

## 5. 未完成与已知问题

### P0：RTF（dGPU 已达 0.70，核显 1.09）→ 目标 < 0.88

> **2026-11-03 更正**：本节原来的"1.07–1.18"是 Intel 核显上的解码环数字，
> 且不含 VAE。见 **0.9**：`--device` 现在真正选设备，P104 上端到端 **0.70**
> （Python 同文本 0.78），已达标；剩余空间在 VAE decode 分块和 dispatch 合并，
> 不是 split-K。

**详细交接见 0.5 节**（时间账 / 已确认事实 / 成功与排除方案 / 剩余三条 / 工具门限）。
**先读 0.5.8**：它推翻了"瓶颈是 dispatch 数"这个判断。

进度：pred/null 合批（2.51 → 2.03）→ t=6 配对激活 + lanes/行（1.74 → 1.16）→
持久 scratch + bind cache（再降 19%，0.5.7）→ 全向量化 + 2 行复用 + K 展开
（DiT 4.29 → 3.2–3.6 s，0.5.8）→ prefill 按 6 token 分块（0.61 → 0.46 s）。

长句整句已到 **5.35–5.91 s**（5.00 s 音频），Python CUDA Graph 是 **5.11 s**，
差 5–15%。下一步**不是**合并 dispatch，而是 **split-K**：DiT 每层只有 32–128 个
workgroup，而 prefill 里同族 kernel 在 13104 个 workgroup 下跑到 150 GB/s
（t=6 只有 47 GB/s）——并行度就是那个 3× 缺口。

### P1：轨迹分岔（已收口：6 例连时长都错 → 10 例"只在后段"）

2026-11-01 修好 DiT rope 之后，每步误差从 ~1e-4 降到 ~1e-6，36/36 时长一致，
只剩 10 例后段的相位差（4.4）。要再收窄只能把每步压到 1e-7 以下，即逐位复刻
cuBLAS 归约 + CUDA 超越函数——4.6 已经用测量证明不可达。**不再是待办。**

### P2：功能缺口（原版有、这个移植没做）

- `--spk_emb_path`（.pt 文件）未实现：需要 torch 序列化格式，目前显式报错。
- pynini 文本规范化未移植：`--no_text_normalization` 关闭时的数字/符号规范化缺失。
  非中文语言走"绕过 TN"分支，所以 F 组不受影响。
- 多分片路径（`max_length` 切出 >1 个 fragment）已补验收（4.7 的 S4/S5：2 片、6 片、21.6 s，rel 1.3e-2）。
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
13. **rope 的 base 不能猜，要读 checkpoint 的 `inv_freq`。** DiT / aggregator 的
    `rotary_embed.inv_freq` 就存在权重里（32 个 f32），它不是 `1e4**(-2i/64)`：
    差 1.4e-4 相对值。进 rope 前 q 的误差 1.8e-7，出 rope 变 8.1e-5（450 倍），
    attention 1.1e-4、block 输出 2.35e-5、41 步 CFM 1.3e-4 —— 这就是"移植版听感更差"
    的唯一原因。凡 state_dict 里有的配置，一律读，不重算。
14. **VAE 的 K/V 和 rope 不能沿用 LLM 的 512 位置窗口。** LLM 是自回归、有 512 的预算
    （4.8 第 3 条），而 VAE 解码器/编码器是**整段一次过**，长度没有上界：>6.4 s 音频
    解码、>10.2 s 参考音频编码，都会越过 512 去写 K/V 和读 rope 表。wgpu 不报错，
    只是音质坏掉。现在走 `ops::KvCache`，按 t 扩容。
15. **ATTEND 的 workgroup scratch 必须用窗口内相对下标。** `sc[pos]` 是 512 槽，
    滑动窗口下绝对位置能到几百上千——越界写共享内存不会崩，只会整组算错。
    现在用 `sc[pos - lo]`。
16. **usize 的减法先 saturating。** `512 - t` 在 t > 512 时回绕成天文数字，
    等于把"没有预算"变成"无限预算"，然后写穿 KV。凡是从容量算剩余量的地方都要先想清楚。

## 7. 决策记录


- 权重：**bf16 原样存储 + shader 内解包 f32 计算**（bf16⊂f32 无损；总线减半）；
  不走 ASR 项目的 f16 权重路线。
- **DiT / aggregator 的 rope 用 checkpoint 里的 `inv_freq` 缓冲，不重算 base**（2026-11-01）。
  M2 时按"x_transformers 默认 theta=1e4"硬编码，相对频率差 1.4e-4 —— 这是整个移植里
  唯一一处"猜配置"而不是"读配置"，代价是每步 1e-4 的系统误差、以及 6 例轨迹分岔。
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

## M2 收口（2026-11-01）：DiT 的 rope base 是猜的 —— 一改全绿

**症状**：用户听出"移植版效果更差"，36 例里 6 例轨迹分岔、3 例连时长都对不上。
把 block0 的算子逐个对拍（`parity_m5::block0_op_by_op`）后，只有 rope 一个环节异常：

| block0 算子 | 修前 rel_l2 | 修后 |
|---|---|---|
| norm1 / q / k / v | 1.2–2.1e-7 | 不变 |
| **q_rope / k_rope** | **8.1e-5** | **1.9e-7** |
| attn（o_proj 输出） | 1.1e-4 | 4.2e-7 |
| norm2 / ff_out | 2.1e-5 / 1.8e-5 | 1.8e-7 |
| block out | 2.35e-5 | 1.05e-7 |

**根因**：`src/dit.rs` 写的是 `RopeTables::new(&gpu, 10_000.0, 64, 512)`，即
`theta ** (-2i/64)`；但 checkpoint 里就存着真正的
`flowloss.cfm.model.rotary_embed.inv_freq`（32 个 f32；aggregator 的
`linear_proj_audio.rotary_embed.inv_freq` 与它完全相同）。它不是 `1e4 ** (-2i/64)`：
前几项是 1, 0.75, 0.5625, 0.421875…，和公式差 1.4e-4 相对值；而且它本身也不是精确的
`0.75^i`（最大差 3.05e-4），像是被更低精度算过一遍再存成 f32。**来路不用管——存的
就是参考实现用的那 32 个数。**

numpy 复算（pos=1/2，直接和 torch 的 q_rope/k_rope 比）：

| rope 频率来源 | q 误差 | k 误差 |
|---|---|---|
| `theta=1e4`（原来的写法） | 8.92e-5 / 2.11e-4 | 8.04e-5 / 1.33e-4 |
| **checkpoint 的 inv_freq** | **5.5e-8 / 4.4e-8** | **4.7e-8 / 4.5e-8** |

**修法**：新增 `RopeTables::from_inv_freq()`，`dit.rs` 加载时读
`flowloss.cfm.model.rotary_embed.inv_freq`（校验 32 项），DiT 与 aggregator 共用；
`RopeTables::new(theta, …)` 只留给 LLM / VAE——它们的 rope 来自 config 的
`rope_theta=1e6`，checkpoint 里没有对应缓冲（只有上面两个 inv_freq），而且 M1 / M3
的门限本来就过。

**效果**（同 seed 1895，36 例全量重跑 + 全量测试）：

- DiT 单前向 2.6e-5 → **1.6e-7**；CFM 41 步 1.28e-4 → **9.6e-7**；y 链 5.8e-5 → **3.0e-7**。
- **36/36 时长逐采样点一致**（修前 C7 1.0400 / G2 1.0704 / H1 1.0270）；26 例波形级重合：
  A1 6.6e-3 → **3.6e-5**，A3 1.03 → **2.6e-3**，B4 6.4e-1 → **3.1e-3**，
  I1-I4（cache-free）2.3e-3/6.5e-2/3.3e-3/4.6e-1 → **4.1e-5/4.7e-4/7.6e-5/7.4e-4**。
- 剩下 10 例只有后段相位差（4.4），前 1/3 逐帧相同。
- 门限顺势收紧（10 倍余量，换卡 ulp 漂移不会误报）：`flowloss < 1e-5`、
  `m2_bisect < 1e-5`、`euler y < 1e-5`、`block0_op_by_op` 每算子 `< 2e-6`。

**教训**：AGENTS.md 原来写着"DiT 的 rope θ=1e4 且交错"——后半句（交错）对，
前半句是猜的。这个 bug 藏了整整一轮：所有下游门限都是按"1e-4 底噪"解释的，
于是 2.35e-5 的 block 误差被写成了"f32 累积序的必然"（旧的 4.6 就是这么写的），
反而把真因盖住了。**当某个算子的误差比它的输入大 400 倍时，那不是噪声。**

## M5 覆盖补齐（2026-11-01）：长文本压力集，和它挖出来的三个越界 bug

**为什么做**：用户问"覆盖全了吗？时长长一点会不会出错"。36 例验收覆盖的是内容面
（task/语言/情感/音色/cfg/温度/克隆），结构面是空的：35/36 单分片、10–62 字、0.9–7 s、
无数字无符号。长文本这条路上**一个样本都没有**。

**压力集**（`tools/stress_compare.py`，10 例，同一 seed 1895 对拍）：

| 用例 | 修前 | 修后 |
|---|---|---|
| S1 嗯。（2 字） | — | rel 1.8e-4，mel 1.00000 |
| S2 好的，谢谢。 | — | rel 4.7e-3，mel 0.99999 |
| S4 60 字分 2 片 | rel 4.3e-3 | 不变（分片路径本来就对） |
| S5 150 字分 6 片 / 21.6 s | rel 1.3e-2 | 不变 |
| S8 266 字符英文单段 | rel 5.9e-1 / mel 0.944 | rel 2.9e-1 / mel 0.970 |
| S6 207 字单段 / 23.2 s | rel 9.2e-1 / mel 0.822 / 音色 0.79 | rel 6.3e-1 / mel 0.885 / 音色 0.90 |

**三个 bug**（详见 4.8）：VAE 解码器/编码器复用 LLM 的 512 窗口（>6.4 s 越界写）、
ATTEND 的 workgroup scratch 用绝对位置索引（滑动窗口下越界）、`512 - t` 的 usize 下溢。

**方法论教训**：这三个 bug 都属于"不报错、不崩、只是变难听"的那一类，只有跨过那条长度的
对拍才能发现。验收集越像"内容测试"，越容易漏掉这种**结构性边界**；补样例时要按
"长度/分片/输入格式"这些正交维度去补，而不是再加十个类似的中文短句。

**新增回归**：`long_prompt_prefill_matches_python`（65/88/161 token prefill）、
`vae_decodes_past_the_llm_window`（640 帧 / 12.8 s）、
`vae_encodes_past_the_llm_window`（600 帧 / 12 s）。三个门限都是 1e-4，实测 1e-6 量级。
