# AGENTS.md — 接手这个仓库前先读

这是 **Ming-3.0-tts-0.5b 的纯 Rust + wgpu 移植**（TTS 推理，无 Python / torch / CUDA
运行依赖）。完整状态、验收数据、未完成项见 [PROGRESS.md](PROGRESS.md)；代码结构评审与
重构记录见 [REVIEW.md](REVIEW.md)。

## 构建 / 运行 / 测试

```bat
cd D:\Ming-3.0-tts\wgpu
cargo build --release
cargo test --release -- --test-threads=1        :: 80 个测试，约 230 s，需要 GPU
```

单条合成（参数面与 Python 的 `infer.py` 逐字一致）：

```bat
target\release\ming-tts-wgpu.exe synth ^
  --model_path D:/Ming-3.0-tts/Ming-3.0-tts-0.5b ^
  --task TTS --text "今天天气真不错，我们一起去公园散步吧。" ^
  --output D:/Ming-3.0-tts/output/rust/A1_bare.wav --no_text_normalization
```

`--device cuda:0 | igpu | cpu` 分别走独显、核显和手写 CPU 后端（`src/cpu/`，无 wgpu）。

模型权重在 `D:/Ming-3.0-tts/Ming-3.0-tts-0.5b/`（不在本仓库）。金标在 `goldens/`
（已入库，67 MB；文本规范化金标见 PROGRESS 0.12）。

## 硬性规则

1. **不要为了让测试变绿去改金标。** 金标来自 Python 参考实现（`tools/dump_*.py`）；
   测试红了说明代码错了。反过来如果确证金标本身错了，先在 PROGRESS.md 里写明证据再重导。
2. **不要动 `src/rng.rs`** 的算法。它是 torch CUDA `randn` 的精确复刻
   （Philox4x32-10 + curand_uniform + box-muller sin），是"同 seed 复现参考音频"的唯一依据，
   已用 28160 个值验证到 1.4e-6。
3. **激活保持 f32，权重保持 bf16 原样上传。** 不要引入 f16：LLM 残差流平台在 7.4e4，
   超出 f16 上限（这不是猜测，PROGRESS.md 决策记录里有逐层实测）。
4. **模型几何来自 checkpoint，不要硬编码猜测**：DiT 的注意力是双向、rope **交错**
   且频率**直接读 checkpoint 的 `rotary_embed.inv_freq`**（它不是 θ=1e4 的解析式，
   两者差 1.4e-4，足以把每步误差从 1e-7 抬到 1e-4 —— 见 PROGRESS 坑清单 13）；
   LLM/VAE 是 Qwen2 rotate-half θ=1e6；VAE 编码/解码有**滑窗 64**。
   搞错任何一个，parity 会以 1e-1 量级失败。
5. **Python 主仓 `D:\Ming-3.0-tts` 是只读参考**，不要改它。
6. **单次改动后必须跑 `cargo test --release`**；碰到音频相关的改动再跑
   `python wgpu/tools/check_params_align.py`（在 `D:\Ming-3.0-tts` 下执行，见文末的验收
   脚本说明）。CPU 与 GPU 都要保持能跑：分片编排已经共用 `src/backend.rs`（0.15），但两个
   decode 环仍是各自实现，改条件（CFM history、seed、停止判据）必须两边一起改 ——
   0.14/0.15 各修过一次漂移。

## 怎么判断"改对了"

- 数值：`cargo test --release` 全绿。每个阶段的门限见 PROGRESS.md 第 4.1 节。
- 音频：`tools/check_params_align.py` 会逐例打印 时长比 / rel_l2 / log-mel 相关 /
  音色余弦；判定标准见 PROGRESS.md 第 4.2 节。
- **改动碰到"长度/缓存/分片"时**跑 `python tools/stress_compare.py`（10 例压力集：
  长文、多分片、极短文本、混排、raw_text）。`cargo test --release` 里另有三个门限专门
  守这条线：长 prefill（161 token）、长解码（640 帧 / 12.8 s）、长编码（600 帧 / 12 s）。
- **不要追求逐采样点一致**：解码环是混沌的，1e-7 的输入扰动会被放大 1e5 倍
  （PROGRESS.md 4.4 已量化）。目标是每步 ~1e-6 数值一致 + 停止判据一致 + 听感一致。
- **P104 掉设备已经修了，别再当"偶发"忽略**：`Buffer with 'readback' label is invalid`
  是进程的第一个命令缓冲超过 ~50 个 dispatch 时驱动复位（nvlddmkm event 153），
  `gpu.rs` 的 `FIRST_SUBMIT_MAX` 已经把它压到 24 个一次 + 立即 drain。改批量/提交逻辑
  时别把这个 prime 弄丢，否则 A4/G2/整套测试会在独显上随机挂（PROGRESS 0.18）。

## 现状（2026-11-04）

- 三条后端（`cuda:0` / `igpu` / `cpu`）与 `infer.py` 的参数面、功能面一致：8 种 task、
  分片、克隆（prompt wav 或 `.pt` 说话人向量）、zero-spk、STYLE 首片重条件、中英文本
  规范化（规则层 + FST + inflect 措辞）。唯一没做的是 `infer.py` 本身没暴露的流式。
- 性能：P104 上 A1 解码（3.2 s 音频）在 10-31 的热状态里是 2.75–2.81 s（RTF 0.60），
  CPU 后端 RTF ≈ 2.2–2.7（原版 Python CPU 4.0–5.3）。**这两个数只对那天的卡状态成立**：
  11-04 用同一批代码（连 11-01 的旧二进制）测出来都是 3.9–4.3 s，卡本身就是差 1.4×，
  不是代码回退。引用 RTF 前必须同窗口重测，A/B 也必须在同一窗口内做。
- 代码结构：`src/backend.rs` 是三条后端共用的分片编排（`Backend` trait + `synthesize`），
  设备相关的算子留在各自的 `Pipeline` 里；`src/cpu/` 与 GPU 各阶段一一对应；
  `src/tn/` 是文本规范化移植；`src/pt.rs` 是无 torch 的 `.pt` 读取器；
  `src/shaders/` 按 kernel 家族分模块（gemv / gemv_t1 / gemv_t6 / prefill / attention /
  elementwise），模型侧仍然只写 `shaders::NAME`。

## 待查与性能决策

代码层面没有已知的清理项，也没有已知的数值缺陷。

（A4 在 P104 上掉设备已查清并修好 —— 进程第一个命令缓冲的容量问题，见 PROGRESS 0.18。
F2_korean 的 1 步时长差也查完了：id 和 RNG 都和参考一致，音频前 1.3 s 吻合到 1e-3 之后按
混沌放大；同一份代码在核显上就和参考同步收尾，三个后端两两之间差异也有 0.4 —— 是混沌，
不是 bug，见 0.19。别再去"修"它。）

性能决策（已复核，结论是不做）：t=6 的 32-lane warp-per-row kernel 在独显上端到端快
9.5%（核显上慢 46%）。现跑参考 36 例里它让 30 例时长全 1.0000、还把 F2 的 1 步时长差"修好"
（只是把混沌挪了个位置），代价是 H1_whisper_secret 的 log-mel 0.99587 → 0.96889，
C6/B2/C1 各降一点。按"不为提速改轨迹"的规矩不收。真要改就是 `ops.rs::Ops::new` 的
`k_linear_t6` 一行 + `t6_plan` 的 `per_wg`，验收清单见 PROGRESS 0.17。

回退口 / 调试开关：`MING_PREFILL=base`（强制 12-token 窗口 kernel）、`MING_SCRATCH=0`、
`MING_LM_FAST=0`、`MING_BIND_CACHE=0`；`MING_PROFILE=1` 分段耗时、
`MING_LM_PREFILL_PROFILE=1` prefill 单层、`MING_MEM=1` 显存、
`MING_DUMP_WGSL=<dir>` 把编译进去的 WGSL 按内容哈希落盘（核对 shader 重构用）。

验收脚本：`tools/check_params_align.py` 比对 `output/params_test/*.wav`（4.2 表那套参考，
**不需要 torch，只要 numpy + soundfile**；但必须在 `D:\Ming-3.0-tts` 下运行，克隆用例的
prompt wav 是相对路径）。`tools/_goldens_align.py` 回放金标里的 `noise_*.npy` 对比
`ref_waveform.npy`，不需要 Python 参考，但它的绝对值带系统偏移（金标里波形与抽样不同源），
**只能做 A/B，不能对门限**；门限级结论一律用前者。
