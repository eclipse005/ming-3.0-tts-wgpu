# wgpu 移植代码评审 + 重构记录

> 评审视角：这份代码要在后面几轮里**被反复改性能**（RTF 3.08 → <0.88）。
> 所以评审标准不是"能跑"，而是"改一处能只改一处、热点路径上没有意外成本、
> 模块边界跟性能实验的边界一致"。

## 一、评审发现（按"阻碍 RTF 推进"的程度排序）

### P0 — 四个模型模块各自复制了一套设备原语（已修）

`qwen_lm.rs` / `dit.rs` / `vae.rs` / `vae_dec.rs` 里各自都有：

| 重复项 | 份数 |
|---|---|
| `struct Lin`（权重+bias 的 lineare） | 3 |
| `uni1` / `uni2`（uniform 打包） | 3 |
| `fn rms` / `fn mrow` / `fn add` / `fn silu_mul` | 4 |
| `fn stack`（Qwen2 残差栈） | 2（`vae` 还有第三份 copy 在 `stack_range_check` 里） |
| `fn blocks_forward` / `fn blocks_range` | 2（逐字重复） |
| 每引擎一套 `k_rms/k_mrow/k_rope/k_attend/...` 字段 | 4 |

**后果**：任何内核层面的优化（tile、展开、向量化、批处理）都要在 4 处同步；
goldens 只能保证"跑得对"，保证不了"改一处只影响一处"。这正是本轮重构的主线。

### P1 — 每个 dispatch 一个 uniform buffer + 一个 bind group（部分已修，见下）

`nn::Kernel::run` 每次调用都会 `create_buffer` + `queue.write_buffer` 做 uniform。
实测 10.5 µs/次；DiT 单次前向约 110 个 dispatch，于是光 uniform 就 ~1.2 ms/前向
（占总耗时 15%）。而且 `queue.write_buffer` 与"延迟提交"是**语义冲突**的
（写入会先于已录制但未提交的批次生效），所以不能简单地"复用一个 UBO 反复写"，
必须走 **uniform arena + dynamic offset**。

### P1 — `Gpu` 职责过载、且 `Clone` 后共享隐藏队列

`Gpu` 同时负责：适配器/设备创建、buffer 分配/上传/回读、dispatch 录制。
`Clone` 后 `dispatches: Arc<RefCell<Vec<..>>>` 是**跨克隆共享**的——正确但隐晦，
`flush()` 的时机靠调用方自觉。建议（未做）：拆成 `Device`（句柄/分配）+ `Recorder`（录制/冲刷）。

### P2 — `parity.rs` 880 行既是测试工具又是测试集

`read_npy` / `rel_l2` 是**生产代码也需要**的（CLI 的 `MING_NOISE_GOLDEN` 读金标 npy），
放在 `parity.rs` 里名不副实；11 个 `mod xxx_tests` 挤在一个文件里，
定位"哪个里程碑挂了"要靠搜函数名。建议：`golden.rs`（npy + rel_l2）+ 各模块的 `#[cfg(test)] mod`。

### P2 — 魔数散落

`882`（hop）、`44100`（采样率）、`896`（hidden）、`1024`/`4096`（DiT）、`64`（latent/head_dim）、
`512`（KV cap）、`0.5`（stop 阈值）在各模块里以字面量出现，其中 `882`/`44100` 在
`vae.rs`/`vae_dec.rs`/`synth.rs`/`pipeline.rs` 各出现一次。建议集中到 `geometry.rs`，
并由 `config.json` 反查校验（避免"改了一处没改另一处"）。

### P3 — 其他

- 51 条编译警告（未用 import、`_zero_c_host`、`assert_f32_multiple`、`bf16_widen` 等死代码）。
- 环境变量开关分散（`MING_NOISE_GOLDEN` / `MING_PROFILE` / `MING_MAX_PENDING` /
  `MING_KV_TRACE` / `MING_STEPS` / `MING_GOLDEN` / `MING_MODEL`），没有集中说明。
- `synth.rs` 在一个文件里做了"参数→request、音色条件、分片循环、写盘"四件事。

## 二、本轮已做的重构

### 1. 新增 `src/ops.rs`：设备原语唯一入口

- `Lin`（含 `Lin::load`，带形状校验）、`QwenLayer`、`RopeTables`、`AttendParams`
  （`causal()` / `full()` / `with_window()` 三个构造器把 32 字节 uniform 的位布局**收进一个地方**）。
- `Ops`：**一套编译好的 kernel**（rms / linear(+T4) / rope / attend / kv_write /
  silu_mul / gelu / add / pack3 / cfm_step），全 pipeline 共用一份。
- 四个引擎改为持有 `ops: Arc<Ops>`，本地只保留"自己的架构"（残差栈、采样器、前端），
  原来的 `fn rms/mrow/add/silu_mul` 变成 3 行委托。

效果：`uni1/uni2` 从 3 份→1 份；`Lin` 3→1；`rms/mrow/add/silu_mul` 4→1；
`k_*` 字段四套→一套；`stack` 3 份→1 份（`stack_range_check` 现在就是 `stack(&self.enc[r], ..)`）；
`blocks_range` 逐字重复直接删除。

### 2. VAE/DIT 的 KV scratch 改为引擎内共用一对

原来每层每前向都 `storage()` 一对 KV（DiT 是 2 MB×2/块/前向，单步 640 MB 存活缓冲，
批量化后直接把驱动逼到 "Out of Memory"）。现在一个引擎一对，按"写→读→下一层重写"
的顺序复用（WebGPU 保证 dispatch 间接可见性）。

### 3. 编译期消灭重复测试辅助

`parity.rs` 里 11 处 `XxxEngine::load(gpu, &weights)` 统一走 `ops_for(&gpu)`，
测试不再各自造一套 kernel。

### 4. uniform arena + dynamic offset（把 P1 干掉）

`nn.rs` 重写：所有 kernel 的 binding 0 改成 `has_dynamic_offset: true`，
uniform 数据写进设备级 **arena**（1 MB = 4096 个 256 B 槽位），
每个 dispatch 只拿一个 offset，flush 时一次性 `write_buffer` 上传。

这条不只是"省 10.5 µs/次"——原来的 `queue.write_buffer` 与"延迟提交"是**语义冲突**的
（写入会先于已录制未提交的批次生效，uniform 会被后写的值覆盖）。所以 arena 是
"既保留批量提交、又保留每 dispatch 独立 uniform"的唯一正解。

`nn.rs` 里三处重复的"建 pipeline + 建 uniform + 建 bind group + run"也合并成一个
`record()`。

### 5. `golden.rs`：把"金标读取"从测试堆里拆出来

`read_npy` / `rel_l2` 移出 `parity.rs` 成为生产模块（CLI 的 `MING_NOISE_GOLDEN`
本来就要读金标 npy），`parity.rs` 标记为 `#[cfg(test)]`。命名终于对上了。

### 6. 告警清零

51 → **0**。做法：删掉迁移残留（`dit.rs` 里第二套 `uni1/uni2/vec_add`、
`gpu::assert_f32_multiple`、`Gpu::run`、`pipeline::load`），
把只在测试里用的诊断接口标 `#[cfg(test)]`，
对"测试用/预留 API"的模块加模块级 `allow(dead_code)` 并写明原因。

## 三、验证

重构前后**数值完全不变**（同一组 goldens，同一组断言）：47 个测试全绿，
M5 端到端仍 `waveform rel_l2 = 6.463e-3`、CLI 输出与 `A1_bare.wav` 的 log-mel 相关
**0.999991**（和重构前一模一样），说明这是纯结构改动。

顺带的首轮收益（`MING_PROFILE=1` 实测）：

| 指标 | 重构前 | 重构后 |
|---|---|---|
| DiT 单前向 | 8.09 ms | **7.58 ms** |
| 40 步解码环 | 9.87 s | **9.39 s** |
| 其中 DiT 段 | 7.92 s | **7.38 s** |
| RTF | 3.08 | **2.93** |
| 编译告警 | 51 | **0** |

## 四、下一轮（RTF）该动的地方，已经就位

1. **uniform arena + dynamic offset**（`ops.rs` 是唯一入口，改一处全覆盖）：
   每个 dispatch 只占 256 B 的 arena 槽位，flush 时一次性上传，去掉 10.5 µs/次 的
   buffer 创建 + write；预计省 1.2 ms/前向。
2. **bind group 缓存**：缓冲区稳定后按 (kernel, buffer 集合) 缓存，省掉每次 dispatch 的
   `create_bind_group`。
3. **pred/null 合批**：给 `AttendParams` 加"分组注意力"（前 3 行与后 3 行互不可见），
   权重只读一遍 → DiT 7.9 s → ~4 s。
4. **tiled GEMM**：`Ops::linear` 换成"每 workgroup 把权重 tile 读进 shared memory、
   对所有 token 复用"，把实测 24 GB/s 拉向 100+ GB/s。