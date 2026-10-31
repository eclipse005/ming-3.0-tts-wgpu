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

**RNG 备注（M5 会用）**：prompt latent 与 CFM 噪声都走 torch.randn，
对拍时由金标注入 eps（已存 m3_eps.npy）；要做到 bit-exact 需复刻 torch Philox，
留到 M5 再决策。
