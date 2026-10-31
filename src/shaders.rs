//! WGSL kernel generators (f32 activations; bf16 weights unpacked in-shader).

/// RMSNorm over the last dim, one row per workgroup (256 lanes), f32 I/O.
/// Layout matches Qwen2/RMSNorm: x * w * rsqrt(mean(x^2) + eps).
pub const RMSNORM: &str = r#"
struct Params { n: u32, eps: f32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg.x;
    let base = row * p.n;
    var acc = 0.0;
    var i = lid.x;
    loop {
        if (i >= p.n) { break; }
        let v = x[base + i];
        acc = acc + v * v;
        i = i + 256u;
    }
    red[lid.x] = acc;
    workgroupBarrier();
    var s = 0.0;
    // tree reduce 256 partials by the first warp
    if (lid.x < 128u) { red[lid.x] = red[lid.x] + red[lid.x + 128u]; }
    workgroupBarrier();
    if (lid.x < 64u) { red[lid.x] = red[lid.x] + red[lid.x + 64u]; }
    workgroupBarrier();
    if (lid.x < 32u) { red[lid.x] = red[lid.x] + red[lid.x + 32u]; }
    workgroupBarrier();
    if (lid.x < 16u) { red[lid.x] = red[lid.x] + red[lid.x + 16u]; }
    workgroupBarrier();
    if (lid.x < 8u) { red[lid.x] = red[lid.x] + red[lid.x + 8u]; }
    workgroupBarrier();
    if (lid.x < 4u) { red[lid.x] = red[lid.x] + red[lid.x + 4u]; }
    workgroupBarrier();
    if (lid.x < 2u) { red[lid.x] = red[lid.x] + red[lid.x + 2u]; }
    workgroupBarrier();
    s = red[0u] + red[1u];
    let inv = 1.0 / sqrt(s / f32(p.n) + p.eps);
    i = lid.x;
    loop {
        if (i >= p.n) { break; }
        y[base + i] = x[base + i] * inv * w[i];
        i = i + 256u;
    }
}
"#;

/// y = W @ x + b for W stored bf16 [rows, in] packed 2-per-u32 (low half = even col).
/// One warp (32 lanes) per row; lane steps over u32 words with stride 32.
/// f32 accumulate. Workgroup 256 = 8 rows per workgroup.
pub const GEMV_BF16W: &str = r#"
struct Params { rows: u32, in_words: u32, }  // in_words = in_features / 2
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<f32, 32>, 8>;

fn bf_lo(u: u32) -> f32 { return bitcast<f32>((u & 0xFFFFu) << 16u); }
fn bf_hi(u: u32) -> f32 { return bitcast<f32>(u & 0xFFFF0000u); }

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let warp = li / 32u;
    let lane = lid.x % 32u;
    let row = wg.x * 8u + warp;
    var acc = 0.0;
    if (row < p.rows) {
        let wbase = row * p.in_words;
        var wi = lane;
        loop {
            if (wi >= p.in_words) { break; }
            let u = w[wbase + wi];
            acc = acc + bf_lo(u) * x[wi * 2u] + bf_hi(u) * x[wi * 2u + 1u];
            wi = wi + 32u;
        }
    }
    // warp reduction via shared memory
    red[warp][lane] = acc;
    workgroupBarrier();
    var stride = 16u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            red[warp][lane] = red[warp][lane] + red[warp][lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lane == 0u && row < p.rows) {
        y[row] = red[warp][0] + b[row];
    }
}
"#;

/// Multi-token linear: y[t, r] = W[r, :] . X[t, :] + b[r].
/// W bf16 packed [rows, in_words]; X f32 [T, in]; Y f32 [T, rows].
/// One warp per (row, token); dispatch (rows/8, T).
pub const GEMV_MROW: &str = r#"
struct Params { rows: u32, in_words: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;   // [T, in]
@group(0) @binding(3) var<storage, read> b: array<f32>;   // [rows]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [T, rows]
var<workgroup> red: array<array<f32, 32>, 8>;

fn bf_lo(u: u32) -> f32 { return bitcast<f32>((u & 0xFFFFu) << 16u); }
fn bf_hi(u: u32) -> f32 { return bitcast<f32>(u & 0xFFFF0000u); }

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let warp = li / 32u;
    let lane = lid.x % 32u;
    let row = wg.x * 8u + warp;
    let t = wg.y;
    var acc = 0.0;
    if (row < p.rows) {
        let wbase = row * p.in_words;
        let xbase = t * (p.in_words * 2u);
        var wi = lane;
        loop {
            if (wi >= p.in_words) { break; }
            let u = w[wbase + wi];
            acc = acc + bf_lo(u) * x[xbase + wi * 2u] + bf_hi(u) * x[xbase + wi * 2u + 1u];
            wi = wi + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    var s = 16u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) { red[warp][lane] = red[warp][lane] + red[warp][lane + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lane == 0u && row < p.rows) {
        y[t * p.rows + row] = red[warp][0] + b[row];
    }
}
"#;

/// Multi-token linear with the weight row read ONCE for all tokens.
/// y[t, r] = W[r,:] . X[t,:] + b[r]
/// W bf16 packed [rows, in_words]; X f32 [T<=4, in]; Y f32 [T, rows].
/// One warp per row, dispatch (rows/8, 1).
///
/// The plain GEMV_MROW dispatches one workgroup per (row, token), which
/// re-reads every weight row once per token. On this card that is 3x the
/// weight bytes per DiT forward (the DiT always runs t=3: cond row + latent
/// history + current latent), and the kernel is bandwidth bound, so it cost
/// ~3x more than it had to.
pub const GEMV_MROW_T4: &str = r#"
struct Params { rows: u32, in_words: u32, n_tok: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;   // [T, in]
@group(0) @binding(3) var<storage, read> b: array<f32>;   // [rows]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [T, rows]
var<workgroup> red: array<array<array<f32, 32>, 4>, 8>;

fn bf_lo(u: u32) -> f32 { return bitcast<f32>((u & 0xFFFFu) << 16u); }
fn bf_hi(u: u32) -> f32 { return bitcast<f32>(u & 0xFFFF0000u); }

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let warp = li / 32u;
    let lane = lid.x % 32u;
    let row = wg.x * 8u + warp;
    let in_len = p.in_words * 2u;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    if (row < p.rows) {
        let wbase = row * p.in_words;
        var wi = lane;
        loop {
            if (wi >= p.in_words) { break; }
            let u = w[wbase + wi];
            let lo = bf_lo(u);
            let hi = bf_hi(u);
            let x0 = wi * 2u;
            if (p.n_tok > 0u) { a0 = a0 + lo * x[x0] + hi * x[x0 + 1u]; }
            if (p.n_tok > 1u) { a1 = a1 + lo * x[in_len + x0] + hi * x[in_len + x0 + 1u]; }
            if (p.n_tok > 2u) { a2 = a2 + lo * x[2u * in_len + x0] + hi * x[2u * in_len + x0 + 1u]; }
            if (p.n_tok > 3u) { a3 = a3 + lo * x[3u * in_len + x0] + hi * x[3u * in_len + x0 + 1u]; }
            wi = wi + 32u;
        }
    }
    red[warp][0][lane] = a0;
    red[warp][1][lane] = a1;
    red[warp][2][lane] = a2;
    red[warp][3][lane] = a3;
    workgroupBarrier();
    var s = 16u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) {
            red[warp][0][lane] = red[warp][0][lane] + red[warp][0][lane + s];
            red[warp][1][lane] = red[warp][1][lane] + red[warp][1][lane + s];
            red[warp][2][lane] = red[warp][2][lane] + red[warp][2][lane + s];
            red[warp][3][lane] = red[warp][3][lane] + red[warp][3][lane + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lane == 0u && row < p.rows) {
        let bias = b[row];
        if (p.n_tok > 0u) { y[row] = red[warp][0][0] + bias; }
        if (p.n_tok > 1u) { y[p.rows + row] = red[warp][1][0] + bias; }
        if (p.n_tok > 2u) { y[2u * p.rows + row] = red[warp][2][0] + bias; }
        if (p.n_tok > 3u) { y[3u * p.rows + row] = red[warp][3][0] + bias; }
    }
}
"#;

/// RoPE applied in place to q or k, one token per workgroup.
/// interlected=0: Qwen2 rotate-half (pairs hd, hd+32); =1: x_transformers
/// interleaved (pairs 2hd, 2hd+1). Angle tables [512, 32] f32 host-side.
/// pos = pos_base + wg.x; dispatch (1, 1) for decode, (T, 1) for prefill.
pub const ROPE: &str = r#"
struct Params { n_heads: u32, pos_base: u32, interleaved: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;  // [512, 32]
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>; // [T, n_heads*64]
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let token = wg.x;
    let pos = p.pos_base + token;
    let c = cos_t[pos * 32u + (lid.x % 32u)];
    let s = sin_t[pos * 32u + (lid.x % 32u)];
    var idx = lid.x;
    loop {
        if (idx >= p.n_heads * 32u) { break; }
        let head = idx / 32u;
        let hd = idx % 32u;
        let base = token * p.n_heads * 64u + head * 64u;
        let d1 = select(base + hd, base + hd * 2u, p.interleaved > 0u);
        let d2 = select(d1 + 32u, d1 + 1u, p.interleaved > 0u);
        let x1 = q[d1];
        let x2 = q[d2];
        q[d1] = x1 * c - x2 * s;
        q[d2] = x2 * c + x1 * s;
        idx = idx + 256u;
    }
}
"#;

/// GQA attention for one query token per (head, token) workgroup.
/// K/V caches: [n_kv_heads * cap * 64] f32, head-major (cap comes from the
/// uniform, so the same kernel serves every stack regardless of its cache size).
/// Causal: query at position q_pos_base + t attends 0..=q_pos_base + t.
pub const ATTEND: &str = r#"
struct Params {
    t_q: u32,          // number of query tokens
    q_pos_base: u32,
    heads_per_kv: u32,
    scale: f32,
    full: u32,         // 1 = bidirectional: every row attends 0..=qpos_max
    qpos_max: u32,
    stride: u32,       // q/y row stride in f32 (hidden size)
    window: u32,       // sliding window (0 = full causal from position 0)
    cap: u32,          // per-head KV cache capacity in tokens
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;   // [T, 896]
@group(0) @binding(2) var<storage, read> kc: array<f32>;  // [n_kv*512*64]
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [T, 896]
var<workgroup> sc: array<f32, 512>;
var<workgroup> wgt: array<f32, 512>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wg.x;
    let t = wg.y;
    let kvh = head / p.heads_per_kv;
    var qpos = p.q_pos_base + t;
    if (p.full > 0u) { qpos = p.qpos_max; }
    let lo = select(0u, max(qpos, p.window - 1u) - (p.window - 1u), p.window > 0u);
    let qbase = t * p.stride + head * 64u;
    let kbase = kvh * p.cap * 64u;
    let d = lid.x;

    // scores for positions lo..=qpos (each lane owns pos d, d+64, ...)
    var pmax = -3.0e38;
    var pos = d;
    loop {
        if (pos > qpos) { break; }
        if (pos >= lo) {
            var s = 0.0;
            var dd = 0u;
            loop {
                if (dd >= 64u) { break; }
                s = s + q[qbase + dd] * kc[kbase + pos * 64u + dd];
                dd = dd + 1u;
            }
            s = s * p.scale;
            sc[pos] = s;
            pmax = max(pmax, s);
        }
        pos = pos + 64u;
    }
    red[d] = pmax;
    workgroupBarrier();
    var s2 = 32u;
    loop {
        if (s2 == 0u) { break; }
        if (d < s2) { red[d] = max(red[d], red[d + s2]); }
        workgroupBarrier();
        s2 = s2 / 2u;
    }
    let m = red[0];

    // weights + sum
    var psum = 0.0;
    pos = d;
    loop {
        if (pos > qpos) { break; }
        if (pos >= lo) {
            let wv = exp(sc[pos] - m);
            wgt[pos] = wv;
            psum = psum + wv;
        }
        pos = pos + 64u;
    }
    red[d] = psum;
    workgroupBarrier();
    s2 = 32u;
    loop {
        if (s2 == 0u) { break; }
        if (d < s2) { red[d] = red[d] + red[d + s2]; }
        workgroupBarrier();
        s2 = s2 / 2u;
    }
    let denom = red[0];

    // weighted V sum: lane d owns output dims d (d < 64)
    var acc = 0.0;
    pos = lo;
    loop {
        if (pos > qpos) { break; }
        acc = acc + wgt[pos] * vc[kbase + pos * 64u + d];
        pos = pos + 1u;
    }
    y[qbase + d] = acc / denom;
}
"#;

/// One CFM Euler step, kept on the device so the sampler needs no host
/// round-trip between the (pred, null) DiT forwards and the update:
///   y[i] += (pred[off+i] + (pred[off+i] - null[off+i]) * cfg) * dt + k * sde[i]
/// pred/null are [3, n] (the DiT emits 3 rows and only the last one is the
/// prediction), hence `off`.
pub const CFM_STEP: &str = r#"
struct Params {
    n: u32,
    dt: f32,
    cfg: f32,
    k: f32,
    off: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pred: array<f32>;
@group(0) @binding(2) var<storage, read> nullp: array<f32>;
@group(0) @binding(3) var<storage, read> sde: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wg.x * 256u + lid.x;
    if (i < p.n) {
        let pc = pred[p.off + i];
        let nc = nullp[p.off + i];
        y[i] = y[i] + (pc + (pc - nc) * p.cfg) * p.dt + p.k * sde[i];
    }
}
"#;

/// out[i] = silu(g[i]) * u[i], elementwise over `n` elements.
pub const SILU_MUL: &str = r#"
struct Params { n: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wg.x * 256u + lid.x;
    if (i < p.n) {
        let v = g[i];
        y[i] = v / (1.0 + exp(-v)) * u[i];
    }
}
"#;

/// y[i] = a[i] + b[i].
pub const ADD: &str = r#"
struct Params { n: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wg.x * 256u + lid.x;
    if (i < p.n) { y[i] = a[i] + b[i]; }
}
"#;

/// Copy k or v for one token per workgroup into the head-major cache:
/// cache[((h * cap) + pos) * 64 + d] = src[token * (n_kv*64) + h * 64 + d].
pub const KV_WRITE: &str = r#"
struct Params { pos_base: u32, cap: u32, n_kv: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;   // [T, n_kv*64]
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let token = wg.x;
    let pos = p.pos_base + token;
    var idx = lid.x;
    let total = p.n_kv * 64u;
    loop {
        if (idx >= total) { break; }
        let h = idx / 64u;
        let d = idx % 64u;
        cache[(h * p.cap + pos) * 64u + d] = src[token * total + idx];
        idx = idx + 256u;
    }
}
"#;

/// Assemble a [3, W] seq from three [W] rows.
pub const PACK3: &str = r#"
struct Params { w: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [3*W]
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wg.x * 256u + lid.x;
    if (i < p.w * 3u) {
        let row = i / p.w;
        let col = i % p.w;
        let v = select(select(c[col], b[col], row == 1u), a[col], row == 0u);
        y[i] = v;
    }
}
"#;

/// GELU, tanh approximation -- the form the DiT/Aggregator actually use:
/// fm/modules.py builds FeedForward with nn.GELU(approximate="tanh"), i.e.
///   0.5 * x * (1 + tanh(beta * x * (1 + kappa * x^2)))
/// with beta = sqrt(2/pi) and kappa = 0.044715, exactly as PyTorch's kernel.
/// The erf form differs from this by up to ~1e-3, which is what capped M2
/// parity at 1.2e-4 and then amplified along the autoregressive decode chain.
pub const GELU_TANH: &str = r#"
struct Params { n: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wg.x * 256u + lid.x;
    if (i < p.n) {
        let v = x[i];
        let beta = 0.7978845608028654;
        let kappa = 0.044715;
        y[i] = 0.5 * v * (1.0 + tanh(v * beta * (1.0 + v * v * kappa)));
    }
}
"#;