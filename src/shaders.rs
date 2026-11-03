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

/// 8-token linear, K loop unrolled 4x.
///
/// The 8-token kernel keeps only ONE weight load in flight per lane; at 8 rows
/// per workgroup that is ~1 KB per workgroup, well under what the memory
/// pipeline needs to cover DRAM latency. Unrolling gives four independent
/// 128-byte row loads per iteration. Requires in_words % 128 == 0 for the fast
/// path; anything else falls through to the same scalar tail the other kernels
/// use.
pub const GEMV_MROW_T8U4: &str = r#"
struct Params { rows: u32, in_words: u32, n_tok: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<array<f32, 32>, 8>, 8>;

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
    var a4 = 0.0;
    var a5 = 0.0;
    var a6 = 0.0;
    var a7 = 0.0;
    let wbase = row * p.in_words;
    var wi = lane;
    loop {
        if (wi + 96u < p.in_words) {
                let u0 = w[wbase + wi + 0u];
                let u1 = w[wbase + wi + 32u];
                let u2 = w[wbase + wi + 64u];
                let u3 = w[wbase + wi + 96u];
                let lo0 = bf_lo(u0);
                let hi0 = bf_hi(u0);
                let lo1 = bf_lo(u1);
                let hi1 = bf_hi(u1);
                let lo2 = bf_lo(u2);
                let hi2 = bf_hi(u2);
                let lo3 = bf_lo(u3);
                let hi3 = bf_hi(u3);
                let x0 = (wi + 0u) * 2u;
                let x1 = (wi + 32u) * 2u;
                let x2 = (wi + 64u) * 2u;
                let x3 = (wi + 96u) * 2u;
                if (p.n_tok > 0u) { a0 = a0 + lo0 * x[x0] + hi0 * x[x0 + 1u]; }
                if (p.n_tok > 0u) { a0 = a0 + lo1 * x[x1] + hi1 * x[x1 + 1u]; }
                if (p.n_tok > 0u) { a0 = a0 + lo2 * x[x2] + hi2 * x[x2 + 1u]; }
                if (p.n_tok > 0u) { a0 = a0 + lo3 * x[x3] + hi3 * x[x3 + 1u]; }
                if (p.n_tok > 1u) { a1 = a1 + lo0 * x[1u * in_len + x0] + hi0 * x[1u * in_len + x0 + 1u]; }
                if (p.n_tok > 1u) { a1 = a1 + lo1 * x[1u * in_len + x1] + hi1 * x[1u * in_len + x1 + 1u]; }
                if (p.n_tok > 1u) { a1 = a1 + lo2 * x[1u * in_len + x2] + hi2 * x[1u * in_len + x2 + 1u]; }
                if (p.n_tok > 1u) { a1 = a1 + lo3 * x[1u * in_len + x3] + hi3 * x[1u * in_len + x3 + 1u]; }
                if (p.n_tok > 2u) { a2 = a2 + lo0 * x[2u * in_len + x0] + hi0 * x[2u * in_len + x0 + 1u]; }
                if (p.n_tok > 2u) { a2 = a2 + lo1 * x[2u * in_len + x1] + hi1 * x[2u * in_len + x1 + 1u]; }
                if (p.n_tok > 2u) { a2 = a2 + lo2 * x[2u * in_len + x2] + hi2 * x[2u * in_len + x2 + 1u]; }
                if (p.n_tok > 2u) { a2 = a2 + lo3 * x[2u * in_len + x3] + hi3 * x[2u * in_len + x3 + 1u]; }
                if (p.n_tok > 3u) { a3 = a3 + lo0 * x[3u * in_len + x0] + hi0 * x[3u * in_len + x0 + 1u]; }
                if (p.n_tok > 3u) { a3 = a3 + lo1 * x[3u * in_len + x1] + hi1 * x[3u * in_len + x1 + 1u]; }
                if (p.n_tok > 3u) { a3 = a3 + lo2 * x[3u * in_len + x2] + hi2 * x[3u * in_len + x2 + 1u]; }
                if (p.n_tok > 3u) { a3 = a3 + lo3 * x[3u * in_len + x3] + hi3 * x[3u * in_len + x3 + 1u]; }
                if (p.n_tok > 4u) { a4 = a4 + lo0 * x[4u * in_len + x0] + hi0 * x[4u * in_len + x0 + 1u]; }
                if (p.n_tok > 4u) { a4 = a4 + lo1 * x[4u * in_len + x1] + hi1 * x[4u * in_len + x1 + 1u]; }
                if (p.n_tok > 4u) { a4 = a4 + lo2 * x[4u * in_len + x2] + hi2 * x[4u * in_len + x2 + 1u]; }
                if (p.n_tok > 4u) { a4 = a4 + lo3 * x[4u * in_len + x3] + hi3 * x[4u * in_len + x3 + 1u]; }
                if (p.n_tok > 5u) { a5 = a5 + lo0 * x[5u * in_len + x0] + hi0 * x[5u * in_len + x0 + 1u]; }
                if (p.n_tok > 5u) { a5 = a5 + lo1 * x[5u * in_len + x1] + hi1 * x[5u * in_len + x1 + 1u]; }
                if (p.n_tok > 5u) { a5 = a5 + lo2 * x[5u * in_len + x2] + hi2 * x[5u * in_len + x2 + 1u]; }
                if (p.n_tok > 5u) { a5 = a5 + lo3 * x[5u * in_len + x3] + hi3 * x[5u * in_len + x3 + 1u]; }
                if (p.n_tok > 6u) { a6 = a6 + lo0 * x[6u * in_len + x0] + hi0 * x[6u * in_len + x0 + 1u]; }
                if (p.n_tok > 6u) { a6 = a6 + lo1 * x[6u * in_len + x1] + hi1 * x[6u * in_len + x1 + 1u]; }
                if (p.n_tok > 6u) { a6 = a6 + lo2 * x[6u * in_len + x2] + hi2 * x[6u * in_len + x2 + 1u]; }
                if (p.n_tok > 6u) { a6 = a6 + lo3 * x[6u * in_len + x3] + hi3 * x[6u * in_len + x3 + 1u]; }
                if (p.n_tok > 7u) { a7 = a7 + lo0 * x[7u * in_len + x0] + hi0 * x[7u * in_len + x0 + 1u]; }
                if (p.n_tok > 7u) { a7 = a7 + lo1 * x[7u * in_len + x1] + hi1 * x[7u * in_len + x1 + 1u]; }
                if (p.n_tok > 7u) { a7 = a7 + lo2 * x[7u * in_len + x2] + hi2 * x[7u * in_len + x2 + 1u]; }
                if (p.n_tok > 7u) { a7 = a7 + lo3 * x[7u * in_len + x3] + hi3 * x[7u * in_len + x3 + 1u]; }
                wi = wi + 128u;
        } else {
            if (wi >= p.in_words) { break; }
            let uT = w[wbase + wi];
            let loT = bf_lo(uT);
            let hiT = bf_hi(uT);
            let xT = wi * 2u;
                if (p.n_tok > 0u) { a0 = a0 + loT * x[xT] + hiT * x[xT + 1u]; }
                if (p.n_tok > 1u) { a1 = a1 + loT * x[1u * in_len + xT] + hiT * x[1u * in_len + xT + 1u]; }
                if (p.n_tok > 2u) { a2 = a2 + loT * x[2u * in_len + xT] + hiT * x[2u * in_len + xT + 1u]; }
                if (p.n_tok > 3u) { a3 = a3 + loT * x[3u * in_len + xT] + hiT * x[3u * in_len + xT + 1u]; }
                if (p.n_tok > 4u) { a4 = a4 + loT * x[4u * in_len + xT] + hiT * x[4u * in_len + xT + 1u]; }
                if (p.n_tok > 5u) { a5 = a5 + loT * x[5u * in_len + xT] + hiT * x[5u * in_len + xT + 1u]; }
                if (p.n_tok > 6u) { a6 = a6 + loT * x[6u * in_len + xT] + hiT * x[6u * in_len + xT + 1u]; }
                if (p.n_tok > 7u) { a7 = a7 + loT * x[7u * in_len + xT] + hiT * x[7u * in_len + xT + 1u]; }
            wi = wi + 32u;
        }
    }
    red[warp][0][lane] = a0;
    red[warp][1][lane] = a1;
    red[warp][2][lane] = a2;
    red[warp][3][lane] = a3;
    red[warp][4][lane] = a4;
    red[warp][5][lane] = a5;
    red[warp][6][lane] = a6;
    red[warp][7][lane] = a7;
    workgroupBarrier();
    var s = 16u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) {
            red[warp][0][lane] = red[warp][0][lane] + red[warp][0][lane + s];
            red[warp][1][lane] = red[warp][1][lane] + red[warp][1][lane + s];
            red[warp][2][lane] = red[warp][2][lane] + red[warp][2][lane + s];
            red[warp][3][lane] = red[warp][3][lane] + red[warp][3][lane + s];
            red[warp][4][lane] = red[warp][4][lane] + red[warp][4][lane + s];
            red[warp][5][lane] = red[warp][5][lane] + red[warp][5][lane + s];
            red[warp][6][lane] = red[warp][6][lane] + red[warp][6][lane + s];
            red[warp][7][lane] = red[warp][7][lane] + red[warp][7][lane + s];
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
        if (p.n_tok > 4u) { y[4u * p.rows + row] = red[warp][4][0] + bias; }
        if (p.n_tok > 5u) { y[5u * p.rows + row] = red[warp][5][0] + bias; }
        if (p.n_tok > 6u) { y[6u * p.rows + row] = red[warp][6][0] + bias; }
        if (p.n_tok > 7u) { y[7u * p.rows + row] = red[warp][7][0] + bias; }
    }
}
"#;

/// [`GEMV_MROW_T8U4`] specialised for the DiT's fixed six-token batch.
///
/// Folding `n_tok` at shader-generation time removes 48 uniform predicate
/// checks per unrolled K step and makes the two unused token groups dead code.
/// The kernel is only selected when t == 6; other row counts keep the generic
/// version.
pub fn gemv_t6u4() -> String {
    GEMV_MROW_T8U4
        .lines()
        // Drop the a6/a7 token groups entirely: every remaining use is guarded
        // by `n_tok > 6`, which cannot hold for this kernel.
        .filter(|line| {
            !(line.contains("a6")
                || line.contains("a7")
                || line.contains("[6]")
                || line.contains("[7]"))
        })
        .collect::<Vec<_>>()
        .join("\n")
        .replace("array<array<array<f32, 32>, 8>, 8>", "array<array<array<f32, 32>, 6>, 8>")
        .replace("p.n_tok", "6u")
}

/// Six-token linear with paired activation loads.
///
/// The scalar kernel loads each token's two adjacent activation values with two
/// 32-bit loads. Viewing X as `[T, K/2]` of `vec2<f32>` turns those into one
/// 64-bit load, cutting 24 load instructions per unrolled K step. Weight loads
/// stay strided so coalescing is unchanged. Requires `in_words % 128 == 0`.
pub const GEMV_T6_PAIRED: &str = r#"
struct Params { rows: u32, in_words: u32, n_tok: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec2<f32>>;   // [T, in_words]
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<array<f32, 32>, 6>, 8>;

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
    let inw = p.in_words;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    var a4 = 0.0;
    var a5 = 0.0;
    if (row < p.rows) {
        let wb = row * inw;
        var wi = lane;
        loop {
            if (wi + 96u >= inw) { break; }
            let u0 = w[wb + wi];
            let u1 = w[wb + wi + 32u];
            let u2 = w[wb + wi + 64u];
            let u3 = w[wb + wi + 96u];
            let l0 = bf_lo(u0); let h0 = bf_hi(u0);
            let l1 = bf_lo(u1); let h1 = bf_hi(u1);
            let l2 = bf_lo(u2); let h2 = bf_hi(u2);
            let l3 = bf_lo(u3); let h3 = bf_hi(u3);
            let q0 = x[wi];
            let q1 = x[inw + wi];
            let q2 = x[2u * inw + wi];
            let q3 = x[3u * inw + wi];
            let q4 = x[4u * inw + wi];
            let q5 = x[5u * inw + wi];
            a0 = a0 + l0*q0.x + h0*q0.y;
            a1 = a1 + l0*q1.x + h0*q1.y;
            a2 = a2 + l0*q2.x + h0*q2.y;
            a3 = a3 + l0*q3.x + h0*q3.y;
            a4 = a4 + l0*q4.x + h0*q4.y;
            a5 = a5 + l0*q5.x + h0*q5.y;
            let r0 = x[wi + 32u];
            let r1 = x[inw + wi + 32u];
            let r2 = x[2u * inw + wi + 32u];
            let r3 = x[3u * inw + wi + 32u];
            let r4 = x[4u * inw + wi + 32u];
            let r5 = x[5u * inw + wi + 32u];
            a0 = a0 + l1*r0.x + h1*r0.y;
            a1 = a1 + l1*r1.x + h1*r1.y;
            a2 = a2 + l1*r2.x + h1*r2.y;
            a3 = a3 + l1*r3.x + h1*r3.y;
            a4 = a4 + l1*r4.x + h1*r4.y;
            a5 = a5 + l1*r5.x + h1*r5.y;
            let s0 = x[wi + 64u];
            let s1 = x[inw + wi + 64u];
            let s2 = x[2u * inw + wi + 64u];
            let s3 = x[3u * inw + wi + 64u];
            let s4 = x[4u * inw + wi + 64u];
            let s5 = x[5u * inw + wi + 64u];
            a0 = a0 + l2*s0.x + h2*s0.y;
            a1 = a1 + l2*s1.x + h2*s1.y;
            a2 = a2 + l2*s2.x + h2*s2.y;
            a3 = a3 + l2*s3.x + h2*s3.y;
            a4 = a4 + l2*s4.x + h2*s4.y;
            a5 = a5 + l2*s5.x + h2*s5.y;
            let v0 = x[wi + 96u];
            let v1 = x[inw + wi + 96u];
            let v2 = x[2u * inw + wi + 96u];
            let v3 = x[3u * inw + wi + 96u];
            let v4 = x[4u * inw + wi + 96u];
            let v5 = x[5u * inw + wi + 96u];
            a0 = a0 + l3*v0.x + h3*v0.y;
            a1 = a1 + l3*v1.x + h3*v1.y;
            a2 = a2 + l3*v2.x + h3*v2.y;
            a3 = a3 + l3*v3.x + h3*v3.y;
            a4 = a4 + l3*v4.x + h3*v4.y;
            a5 = a5 + l3*v5.x + h3*v5.y;
            wi = wi + 128u;
        }
    }
    red[warp][0][lane] = a0;
    red[warp][1][lane] = a1;
    red[warp][2][lane] = a2;
    red[warp][3][lane] = a3;
    red[warp][4][lane] = a4;
    red[warp][5][lane] = a5;
    workgroupBarrier();
    var s = 16u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) {
            red[warp][0][lane] = red[warp][0][lane] + red[warp][0][lane + s];
            red[warp][1][lane] = red[warp][1][lane] + red[warp][1][lane + s];
            red[warp][2][lane] = red[warp][2][lane] + red[warp][2][lane + s];
            red[warp][3][lane] = red[warp][3][lane] + red[warp][3][lane + s];
            red[warp][4][lane] = red[warp][4][lane] + red[warp][4][lane + s];
            red[warp][5][lane] = red[warp][5][lane] + red[warp][5][lane + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lane == 0u && row < p.rows) {
        let bias = b[row];
        y[row] = red[warp][0][0] + bias;
        y[p.rows + row] = red[warp][1][0] + bias;
        y[2u * p.rows + row] = red[warp][2][0] + bias;
        y[3u * p.rows + row] = red[warp][3][0] + bias;
        y[4u * p.rows + row] = red[warp][4][0] + bias;
        y[5u * p.rows + row] = red[warp][5][0] + bias;
    }
}
"#;

/// Six-token linear with `lanes` threads per output row.
///
/// The 32-lane kernel spends five workgroup barriers reducing each row while a
/// K=1024 matrix only has four unrolled loop steps. Using 16 lanes doubles the
/// rows per workgroup and cuts the reduction to four steps; 8 lanes reaches
/// three steps. Paired activation loads are retained from
/// [`GEMV_T6_PAIRED`].
fn gemv_t6_lanes(lanes: u32, threads: u32) -> String {
    assert!(32 % lanes == 0 && threads % 32 == 0 && threads % lanes == 0);
    let tok: u32 = 6;
    let rows_per_wg = threads / lanes;
    let offsets = [0u32, lanes, 2 * lanes, 3 * lanes];
    let last_off = 3 * lanes;
    let stride = 4 * lanes;
    let first_step = lanes / 2;
    let mut body = String::new();
    for off in offsets {
        body.push_str(&format!(
            "            let u{off} = w[wb + wi + {off}u];\n\
             \x20           let l{off} = bf_lo(u{off});\n\
             \x20           let h{off} = bf_hi(u{off});\n"
        ));
        for t in 0..tok {
            let idx = if t == 0 {
                format!("wi + {off}u")
            } else {
                format!("{t}u * inw + wi + {off}u")
            };
            body.push_str(&format!(
                "            let q{off}_{t} = x[{idx}];\n\
                 \x20           a{t} = a{t} + l{off}*q{off}_{t}.x + h{off}*q{off}_{t}.y;\n"
            ));
        }
    }
    let mut reduce = String::new();
    for t in 0..tok {
        reduce.push_str(&format!("            red[rl][{t}][lane] = red[rl][{t}][lane] + red[rl][{t}][lane + s];\n"));
    }
    let mut store = String::new();
    for t in 0..tok {
        let idx = if t == 0 {
            "row".to_string()
        } else {
            format!("{t}u * p.rows + row")
        };
        store.push_str(&format!("        y[{idx}] = red[rl][{t}][0] + bias;\n"));
    }
    format!(
        r#"struct Params {{ rows: u32, in_words: u32, n_tok: u32, _pad: u32, }}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<array<f32, {lanes}>, {tok}>, {rows_per_wg}>;

fn bf_lo(u: u32) -> f32 {{ return bitcast<f32>((u & 0xFFFFu) << 16u); }}
fn bf_hi(u: u32) -> f32 {{ return bitcast<f32>(u & 0xFFFF0000u); }}

@compute @workgroup_size({threads})
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let lane = lid.x % {lanes}u;
    let rl = lid.x / {lanes}u;
    let row = wg.x * {rows_per_wg}u + rl;
    let inw = p.in_words;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    var a4 = 0.0;
    var a5 = 0.0;
    if (row < p.rows) {{
        let wb = row * inw;
        var wi = lane;
        loop {{
            if (wi + {last_off}u >= inw) {{ break; }}
{body}            wi = wi + {stride}u;
        }}
    }}
    red[rl][0][lane] = a0;
    red[rl][1][lane] = a1;
    red[rl][2][lane] = a2;
    red[rl][3][lane] = a3;
    red[rl][4][lane] = a4;
    red[rl][5][lane] = a5;
    workgroupBarrier();
    var s = {first_step}u;
    loop {{
        if (s == 0u) {{ break; }}
        if (lane < s) {{
{reduce}        }}
        workgroupBarrier();
        s = s / 2u;
    }}
    if (lane == 0u && row < p.rows) {{
        let bias = b[row];
{store}    }}
}}
"#
    )
}

pub fn gemv_t6_halfwarp() -> String {
    gemv_t6_lanes(16, 256)
}

pub fn gemv_t6_quarterwarp() -> String {
    gemv_t6_lanes(8, 256)
}

pub fn gemv_t6_fourlane() -> String {
    gemv_t6_lanes(4, 256)
}

/// Six-token linear with fully vectorised loads and `rpl` output rows per lane
/// group.
///
/// [`gemv_t6_lanes`] issues four scalar `u32` weight loads and 24 scalar
/// activation loads per 8-wide K granule, i.e. 28 load instructions for 48 FMAs.
/// This card has no native bf16, so the kernel is instruction- and
/// latency-bound long before it is bandwidth-bound: cuBLAS in fp32 streams
/// ~120 GB/s here while the scalar kernel sustains 29-43 GB/s.
///
/// Here one `vec4<u32>` covers the same 8 K elements (16 B) and the activations
/// come in as `vec4<f32>`, two per token. With two rows per lane group that is
/// 2 + 12 = 14 loads for 96 FMAs, or 7 loads per row per granule instead of 28.
///
/// `unroll` granules are processed per loop iteration and their loads are all
/// emitted before the first FMA. One weight load in flight per lane group is not
/// enough to cover ~600 cycles of DRAM latency from 8-16 warps per SM, which is
/// why the single-granule body plateaus near 46 GB/s.
///
/// `in4` must be a multiple of `unroll * lanes`; callers guarantee
/// `in_features % 256 == 0`, i.e. `in4 % 32 == 0`, so this holds for
/// `lanes in {8, 32}` with `unroll in {1, 2, 4}`.
pub fn gemv_t6_vec4(lanes: u32, threads: u32, rpl: u32, unroll: u32, guarded: bool, tok: u32) -> String {
    assert!(lanes > 0 && 32 % lanes == 0, "lanes must divide a warp");
    assert!(threads % 32 == 0 && threads % lanes == 0);
    assert!(rpl >= 1 && unroll >= 1);
    let rowgroups = threads / lanes;
    let rows_per_wg = rowgroups * rpl;
    let first_step = lanes / 2;
    let stride = lanes * unroll;

    let mut decl = String::new();
    for m in 0..rpl {
        for t in 0..tok {
            decl.push_str(&format!("    var a{m}_{t} = 0.0;\n"));
        }
    }

    let mut bases = String::from("        let wb0 = r0 * in4;\n");
    for m in 1..rpl {
        bases.push_str(&format!(
            "        let wb{m} = select(wb0, (r0 + {m}u) * in4, r0 + {m}u < p.rows);\n"
        ));
    }

    let mut body = String::new();
    body.push_str("            let g4 = 2u * g;\n");
    // Issue every weight load of the unrolled group first so the memory
    // pipeline has `rpl * unroll` requests outstanding before the first FMA.
    for m in 0..rpl {
        for u in 0..unroll {
            let tail = if u == 0 {
                String::new()
            } else {
                format!(" + {}u", u * lanes)
            };
            body.push_str(&format!(
                "            let uw{m}_{u} = w[wb{m} + g{tail}];\n"
            ));
        }
    }
    for m in 0..rpl {
        for u in 0..unroll {
            for (j, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                body.push_str(&format!(
                    "            let lo{m}_{u}_{j} = bf_lo(uw{m}_{u}.{comp});\n\
                     \x20           let hi{m}_{u}_{j} = bf_hi(uw{m}_{u}.{comp});\n"
                ));
            }
        }
    }
    // One token at a time: its activations stay live just long enough to feed
    // every row accumulator, so peak pressure is `2 * unroll` vec4 of x rather
    // than all six tokens at once.
    for t in 0..tok {
        // The guarded variant walks a sliding six-token window across a long
        // prompt; the unguarded one is the DiT's fixed six-row batch and keeps
        // the loop body free of both the token-offset add and the n_tok tests.
        let tok = if guarded {
            format!("(tok0 + {t}u) * x4 + ")
        } else if t == 0 {
            String::new()
        } else {
            format!("{t}u * x4 + ")
        };
        for u in 0..unroll {
            let tail = if u == 0 {
                String::new()
            } else {
                format!(" + {}u", 2 * u * lanes)
            };
            body.push_str(&format!(
                "            let xa{t}_{u} = x[{tok}g4{tail}];\n\
                 \x20           let xb{t}_{u} = x[{tok}g4{tail} + 1u];\n"
            ));
        }
        if guarded {
            body.push_str(&format!("            if (p.n_tok > {t}u) {{\n"));
        }
        for m in 0..rpl {
            for u in 0..unroll {
                body.push_str(&format!(
                    "            a{m}_{t} = a{m}_{t} + lo{m}_{u}_0*xa{t}_{u}.x + hi{m}_{u}_0*xa{t}_{u}.y \
                     + lo{m}_{u}_1*xa{t}_{u}.z + hi{m}_{u}_1*xa{t}_{u}.w \
                     + lo{m}_{u}_2*xb{t}_{u}.x + hi{m}_{u}_2*xb{t}_{u}.y \
                     + lo{m}_{u}_3*xb{t}_{u}.z + hi{m}_{u}_3*xb{t}_{u}.w;\n"
                ));
            }
        }
        if guarded {
            body.push_str("            }\n");
        }
    }

    let mut reduce = String::new();
    for m in 0..rpl {
        for t in 0..tok {
            let i = format!("red[rl * {rpl}u + {m}u][{t}]");
            reduce.push_str(&format!(
                "            {i}[lane] = {i}[lane] + {i}[lane + s];\n"
            ));
        }
    }

    let mut store = String::new();
    for m in 0..rpl {
        let guard = if m == 0 {
            "r0 < p.rows".to_string()
        } else {
            format!("r0 + {m}u < p.rows")
        };
        store.push_str(&format!("        if ({guard}) {{\n"));
        store.push_str(&format!("            let row = r0 + {m}u;\n"));
        store.push_str("            let bias = b[row];\n");
        for t in 0..tok {
            let (open, close, idx) = if guarded {
                (
                    format!("            if (p.n_tok > {t}u) {{\n"),
                    "            }\n".to_string(),
                    format!("(tok0 + {t}u) * p.rows + row"),
                )
            } else if t == 0 {
                (String::new(), String::new(), "row".to_string())
            } else {
                (String::new(), String::new(), format!("{t}u * p.rows + row"))
            };
            store.push_str(&open);
            store.push_str(&format!(
                "            y[{idx}] = red[rl * {rpl}u + {m}u][{t}][0] + bias;\n"
            ));
            store.push_str(&close);
        }
        store.push_str("        }\n");
    }

    let mut store_in = String::new();
    for m in 0..rpl {
        for t in 0..tok {
            store_in.push_str(&format!(
                "    red[rl * {rpl}u + {m}u][{t}][lane] = a{m}_{t};\n"
            ));
        }
    }

    format!(
        r#"struct Params {{ rows: u32, in_words: u32, n_tok: u32, tok0: u32, }}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<array<f32, {lanes}>, {tok}>, {rows_per_wg}>;

fn bf_lo(u: u32) -> f32 {{ return bitcast<f32>((u & 0xFFFFu) << 16u); }}
fn bf_hi(u: u32) -> f32 {{ return bitcast<f32>(u & 0xFFFF0000u); }}

@compute @workgroup_size({threads})
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let lane = lid.x % {lanes}u;
    let rl = lid.x / {lanes}u;
    let r0 = wg.x * {rows_per_wg}u + rl * {rpl}u;
    let inw = p.in_words;
    let in4 = inw / 4u;
    let x4 = inw / 2u;
    let tok0 = p.tok0;
{decl}    if (r0 < p.rows) {{
{bases}        var g = lane;
        loop {{
            if (g >= in4) {{ break; }}
{body}            g = g + {stride}u;
        }}
    }}
{store_in}    workgroupBarrier();
    var s = {first_step}u;
    loop {{
        if (s == 0u) {{ break; }}
        if (lane < s) {{
{reduce}        }}
        workgroupBarrier();
        s = s / 2u;
    }}
    if (lane == 0u) {{
{store}    }}
}}
"#
    )
}

/// Single-token linear with half a warp per row.
///
/// Same vec4 weight/activation granules as [`GEMV_V4_T1`], but two rows share a
/// warp and the reduction needs four steps instead of five. K=896 matrices in
/// the LLM decoder only run 3-4 K steps per lane, so the barrier tree is a
/// visible fraction of the kernel.
pub fn gemv_t1_halfwarp() -> String {
    let reduce = "            red[rl][lane] = red[rl][lane] + red[rl][lane + s];\n";
    format!(
        r#"struct Params {{ rows: u32, kg: u32, }}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<f32, 16>, 16>;

fn bf_lo(u: u32) -> f32 {{ return bitcast<f32>((u & 0xFFFFu) << 16u); }}
fn bf_hi(u: u32) -> f32 {{ return bitcast<f32>(u & 0xFFFF0000u); }}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let warp = li / 32u;
    let half = (lid.x % 32u) / 16u;
    let lane = lid.x % 16u;
    let rl = warp * 2u + half;
    let row = wg.x * 16u + rl;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    if (row < p.rows) {{
        let wb = row * p.kg;
        var i = lane;
        loop {{
            if (i >= p.kg) {{ break; }}
            let wv = w[wb + i];
            let xa = x[2u * i];
            let xb = x[2u * i + 1u];
            a0 = a0 + bf_lo(wv.x) * xa.x + bf_hi(wv.x) * xa.y;
            a1 = a1 + bf_lo(wv.y) * xa.z + bf_hi(wv.y) * xa.w;
            a2 = a2 + bf_lo(wv.z) * xb.x + bf_hi(wv.z) * xb.y;
            a3 = a3 + bf_lo(wv.w) * xb.z + bf_hi(wv.w) * xb.w;
            i = i + 16u;
        }}
    }}
    red[rl][lane] = a0 + a1 + a2 + a3;
    workgroupBarrier();
    var s = 8u;
    loop {{
        if (s == 0u) {{ break; }}
        if (lane < s) {{
{reduce}        }}
        workgroupBarrier();
        s = s / 2u;
    }}
    if (lane == 0u && row < p.rows) {{
        y[row] = red[rl][0] + b[row];
    }}
}}
"#
    )
}

/// Single-token linear with a configurable number of lanes per output row and
/// `rpl` rows per lane group.
///
/// Same vec4 weight/activation granules as [`GEMV_V4_T1`]. The parameters
/// matter because the LLM decoder mixes two very different shapes: the
/// 4864x896 projections can fill the machine with any row width, while the
/// 896x4864 down-projection has only 896 rows and needs a narrow row width to
/// emit enough workgroups. Lane count also sets the barrier-tree depth (log2).
pub fn gemv_t1_vec4(lanes: u32, threads: u32, rpl: u32) -> String {
    assert!(lanes > 0 && 32 % lanes == 0, "lanes must divide a warp");
    assert!(threads % 32 == 0 && threads % lanes == 0);
    assert!(rpl >= 1);
    let rowgroups = threads / lanes;
    let rows_per_wg = rowgroups * rpl;

    let mut decl = String::new();
    for m in 0..rpl {
        for j in 0..4u32 {
            decl.push_str(&format!("    var a{m}_{j} = 0.0;\n"));
        }
    }

    let mut bases = String::from("        let wb0 = r0 * p.kg;\n");
    for m in 1..rpl {
        bases.push_str(&format!(
            "        let wb{m} = select(wb0, (r0 + {m}u) * p.kg, r0 + {m}u < p.rows);\n"
        ));
    }

    let mut body = String::new();
    body.push_str("            let xa = x[2u * i];\n            let xb = x[2u * i + 1u];\n");
    for m in 0..rpl {
        body.push_str(&format!("            let wv{m} = w[wb{m} + i];\n"));
        // granule halves: (weight component, activation vector, lo, hi)
        let terms = [
            ("x", "xa", "x", "y"),
            ("y", "xa", "z", "w"),
            ("z", "xb", "x", "y"),
            ("w", "xb", "z", "w"),
        ];
        for (j, (comp, vec, lo, hi)) in terms.iter().enumerate() {
            body.push_str(&format!(
                "            a{m}_{j} = a{m}_{j} + bf_lo(wv{m}.{comp}) * {vec}.{lo} \
                 + bf_hi(wv{m}.{comp}) * {vec}.{hi};\n"
            ));
        }
    }

    let mut store_in = String::new();
    for m in 0..rpl {
        store_in.push_str(&format!(
            "    red[rl * {rpl}u + {m}u][lane] = a{m}_0 + a{m}_1 + a{m}_2 + a{m}_3;\n"
        ));
    }

    let mut reduce = String::new();
    for m in 0..rpl {
        let i = format!("red[rl * {rpl}u + {m}u]");
        reduce.push_str(&format!(
            "            {i}[lane] = {i}[lane] + {i}[lane + s];\n"
        ));
    }

    let mut store = String::new();
    for m in 0..rpl {
        let guard = if m == 0 {
            "r0 < p.rows".to_string()
        } else {
            format!("r0 + {m}u < p.rows")
        };
        store.push_str(&format!(
            "        if ({guard}) {{\n            let row = r0 + {m}u;\n\
             \x20           y[row] = red[rl * {rpl}u + {m}u][0] + b[row];\n        }}\n"
        ));
    }

    format!(
        r#"struct Params {{ rows: u32, kg: u32, }}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<f32, {lanes}>, {rows_per_wg}>;

fn bf_lo(u: u32) -> f32 {{ return bitcast<f32>((u & 0xFFFFu) << 16u); }}
fn bf_hi(u: u32) -> f32 {{ return bitcast<f32>(u & 0xFFFF0000u); }}

@compute @workgroup_size({threads})
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let lane = lid.x % {lanes}u;
    let rl = lid.x / {lanes}u;
    let r0 = wg.x * {rows_per_wg}u + rl * {rpl}u;
    let kg = p.kg;
{decl}    if (r0 < p.rows) {{
{bases}        var i = lane;
        loop {{
            if (i >= kg) {{ break; }}
{body}            i = i + {lanes}u;
        }}
    }}
{store_in}    workgroupBarrier();
    var s = {first}u;
    loop {{
        if (s == 0u) {{ break; }}
        if (lane < s) {{
{reduce}        }}
        workgroupBarrier();
        s = s / 2u;
    }}
    if (lane == 0u) {{
{store}    }}
}}
"#,
        first = lanes / 2
    )
}

/// Single-token linear with vec4 weight loads (minimal form, no unrolling).
///
/// One warp per row; a granule is 8 bf16 in one `vec4<u32>` (16 B) plus the
/// matching 8 activation floats (two `vec4<f32>`). Value pair j of a granule
/// feeds accumulator j, so the four accumulators stay independent and the
/// reduction is a plain xor butterfly over the 32 lanes.
///
/// This exists because the u32-at-a-time kernel sustains 18-26 GB/s on this
/// card while the qwen3-asr port measures 226-268 GB/s for the same shape.
pub const GEMV_V4_T1: &str = r#"
struct Params { rows: u32, kg: u32, }   // kg = in_features / 8
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;   // 8 bf16 each
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;   // 4 f32 each
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> bt0: array<f32, 256>;
var<workgroup> bt1: array<f32, 256>;

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
    let wb = row * p.kg;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    if (row < p.rows) {
        var i = lane;
        loop {
            if (i >= p.kg) { break; }
            let wv = w[wb + i];
            let xa = x[i * 2u];
            let xb = x[i * 2u + 1u];
            a0 = a0 + bf_lo(wv.x) * xa.x + bf_hi(wv.x) * xa.y;
            a1 = a1 + bf_lo(wv.y) * xa.z + bf_hi(wv.y) * xa.w;
            a2 = a2 + bf_lo(wv.z) * xb.x + bf_hi(wv.z) * xb.y;
            a3 = a3 + bf_lo(wv.w) * xb.z + bf_hi(wv.w) * xb.w;
            i = i + 32u;
        }
    }
    let wid = li & 0xFFFFFFE0u;
    bt0[lid.x] = a0 + a1 + a2 + a3;
    workgroupBarrier();
    var s = bt0[lid.x] + bt0[wid + (lane ^ 16u)];
    bt1[lid.x] = s;
    workgroupBarrier();
    s = bt1[lid.x] + bt1[wid + (lane ^ 8u)];
    bt0[lid.x] = s;
    workgroupBarrier();
    s = bt0[lid.x] + bt0[wid + (lane ^ 4u)];
    bt1[lid.x] = s;
    workgroupBarrier();
    s = bt1[lid.x] + bt1[wid + (lane ^ 2u)];
    bt0[lid.x] = s;
    workgroupBarrier();
    s = bt0[lid.x] + bt0[wid + (lane ^ 1u)];
    if (lane == 0u && row < p.rows) {
        y[row] = s + b[row];
    }
}
"#;

/// Six-token linear with vec4 weight loads: the DiT's CFG batch.
///
/// The same granule arithmetic as the single-token kernel (value pair j feeds
/// accumulator j), repeated over the six rows of the batched sequence while the
/// weight granule is read once for all of them. Six independent accumulator
/// groups keep the dependency depth at two FMAs.
pub const GEMV_V4_T6: &str = r#"
struct Params { rows: u32, kg: u32, }   // kg = in_features / 8
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> bt: array<array<f32, 256>, 6>;

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
    let wid = li & 0xFFFFFFE0u;
    let row = wg.x * 8u + warp;
    let wb = row * p.kg;
    var a00 = 0.0;
    var a01 = 0.0;
    var a02 = 0.0;
    var a03 = 0.0;
    var a10 = 0.0;
    var a11 = 0.0;
    var a12 = 0.0;
    var a13 = 0.0;
    var a20 = 0.0;
    var a21 = 0.0;
    var a22 = 0.0;
    var a23 = 0.0;
    var a30 = 0.0;
    var a31 = 0.0;
    var a32 = 0.0;
    var a33 = 0.0;
    var a40 = 0.0;
    var a41 = 0.0;
    var a42 = 0.0;
    var a43 = 0.0;
    var a50 = 0.0;
    var a51 = 0.0;
    var a52 = 0.0;
    var a53 = 0.0;
    if (row < p.rows) {
        var i = lane;
        loop {
            if (i >= p.kg) { break; }
            let wv = w[wb + i];
            let xa0 = x[2u * i];
            let xb0 = x[2u * i + 1u];
            a00 = a00 + bf_lo(wv.x) * xa0.x + bf_hi(wv.x) * xa0.y;
            a01 = a01 + bf_lo(wv.y) * xa0.z + bf_hi(wv.y) * xa0.w;
            a02 = a02 + bf_lo(wv.z) * xb0.x + bf_hi(wv.z) * xb0.y;
            a03 = a03 + bf_lo(wv.w) * xb0.z + bf_hi(wv.w) * xb0.w;
            let xa1 = x[2u * i + 2u * p.kg];
            let xb1 = x[2u * i + 2u * p.kg + 1u];
            a10 = a10 + bf_lo(wv.x) * xa1.x + bf_hi(wv.x) * xa1.y;
            a11 = a11 + bf_lo(wv.y) * xa1.z + bf_hi(wv.y) * xa1.w;
            a12 = a12 + bf_lo(wv.z) * xb1.x + bf_hi(wv.z) * xb1.y;
            a13 = a13 + bf_lo(wv.w) * xb1.z + bf_hi(wv.w) * xb1.w;
            let xa2 = x[2u * i + 4u * p.kg];
            let xb2 = x[2u * i + 4u * p.kg + 1u];
            a20 = a20 + bf_lo(wv.x) * xa2.x + bf_hi(wv.x) * xa2.y;
            a21 = a21 + bf_lo(wv.y) * xa2.z + bf_hi(wv.y) * xa2.w;
            a22 = a22 + bf_lo(wv.z) * xb2.x + bf_hi(wv.z) * xb2.y;
            a23 = a23 + bf_lo(wv.w) * xb2.z + bf_hi(wv.w) * xb2.w;
            let xa3 = x[2u * i + 6u * p.kg];
            let xb3 = x[2u * i + 6u * p.kg + 1u];
            a30 = a30 + bf_lo(wv.x) * xa3.x + bf_hi(wv.x) * xa3.y;
            a31 = a31 + bf_lo(wv.y) * xa3.z + bf_hi(wv.y) * xa3.w;
            a32 = a32 + bf_lo(wv.z) * xb3.x + bf_hi(wv.z) * xb3.y;
            a33 = a33 + bf_lo(wv.w) * xb3.z + bf_hi(wv.w) * xb3.w;
            let xa4 = x[2u * i + 8u * p.kg];
            let xb4 = x[2u * i + 8u * p.kg + 1u];
            a40 = a40 + bf_lo(wv.x) * xa4.x + bf_hi(wv.x) * xa4.y;
            a41 = a41 + bf_lo(wv.y) * xa4.z + bf_hi(wv.y) * xa4.w;
            a42 = a42 + bf_lo(wv.z) * xb4.x + bf_hi(wv.z) * xb4.y;
            a43 = a43 + bf_lo(wv.w) * xb4.z + bf_hi(wv.w) * xb4.w;
            let xa5 = x[2u * i + 10u * p.kg];
            let xb5 = x[2u * i + 10u * p.kg + 1u];
            a50 = a50 + bf_lo(wv.x) * xa5.x + bf_hi(wv.x) * xa5.y;
            a51 = a51 + bf_lo(wv.y) * xa5.z + bf_hi(wv.y) * xa5.w;
            a52 = a52 + bf_lo(wv.z) * xb5.x + bf_hi(wv.z) * xb5.y;
            a53 = a53 + bf_lo(wv.w) * xb5.z + bf_hi(wv.w) * xb5.w;
            i = i + 32u;
        }
    }
    bt[0][lid.x] = a00 + a01 + a02 + a03;
    bt[1][lid.x] = a10 + a11 + a12 + a13;
    bt[2][lid.x] = a20 + a21 + a22 + a23;
    bt[3][lid.x] = a30 + a31 + a32 + a33;
    bt[4][lid.x] = a40 + a41 + a42 + a43;
    bt[5][lid.x] = a50 + a51 + a52 + a53;
    workgroupBarrier();
    workgroupBarrier();
    bt[0][lid.x] = bt[0][lid.x] + bt[0][wid + (lane ^ 16u)];
    bt[1][lid.x] = bt[1][lid.x] + bt[1][wid + (lane ^ 16u)];
    bt[2][lid.x] = bt[2][lid.x] + bt[2][wid + (lane ^ 16u)];
    bt[3][lid.x] = bt[3][lid.x] + bt[3][wid + (lane ^ 16u)];
    bt[4][lid.x] = bt[4][lid.x] + bt[4][wid + (lane ^ 16u)];
    bt[5][lid.x] = bt[5][lid.x] + bt[5][wid + (lane ^ 16u)];
    workgroupBarrier();
    bt[0][lid.x] = bt[0][lid.x] + bt[0][wid + (lane ^ 8u)];
    bt[1][lid.x] = bt[1][lid.x] + bt[1][wid + (lane ^ 8u)];
    bt[2][lid.x] = bt[2][lid.x] + bt[2][wid + (lane ^ 8u)];
    bt[3][lid.x] = bt[3][lid.x] + bt[3][wid + (lane ^ 8u)];
    bt[4][lid.x] = bt[4][lid.x] + bt[4][wid + (lane ^ 8u)];
    bt[5][lid.x] = bt[5][lid.x] + bt[5][wid + (lane ^ 8u)];
    workgroupBarrier();
    bt[0][lid.x] = bt[0][lid.x] + bt[0][wid + (lane ^ 4u)];
    bt[1][lid.x] = bt[1][lid.x] + bt[1][wid + (lane ^ 4u)];
    bt[2][lid.x] = bt[2][lid.x] + bt[2][wid + (lane ^ 4u)];
    bt[3][lid.x] = bt[3][lid.x] + bt[3][wid + (lane ^ 4u)];
    bt[4][lid.x] = bt[4][lid.x] + bt[4][wid + (lane ^ 4u)];
    bt[5][lid.x] = bt[5][lid.x] + bt[5][wid + (lane ^ 4u)];
    workgroupBarrier();
    bt[0][lid.x] = bt[0][lid.x] + bt[0][wid + (lane ^ 2u)];
    bt[1][lid.x] = bt[1][lid.x] + bt[1][wid + (lane ^ 2u)];
    bt[2][lid.x] = bt[2][lid.x] + bt[2][wid + (lane ^ 2u)];
    bt[3][lid.x] = bt[3][lid.x] + bt[3][wid + (lane ^ 2u)];
    bt[4][lid.x] = bt[4][lid.x] + bt[4][wid + (lane ^ 2u)];
    bt[5][lid.x] = bt[5][lid.x] + bt[5][wid + (lane ^ 2u)];
    workgroupBarrier();
    bt[0][lid.x] = bt[0][lid.x] + bt[0][wid + (lane ^ 1u)];
    bt[1][lid.x] = bt[1][lid.x] + bt[1][wid + (lane ^ 1u)];
    bt[2][lid.x] = bt[2][lid.x] + bt[2][wid + (lane ^ 1u)];
    bt[3][lid.x] = bt[3][lid.x] + bt[3][wid + (lane ^ 1u)];
    bt[4][lid.x] = bt[4][lid.x] + bt[4][wid + (lane ^ 1u)];
    bt[5][lid.x] = bt[5][lid.x] + bt[5][wid + (lane ^ 1u)];

    if (lane == 0u && row < p.rows) {
        y[0u * p.rows + row] = bt[0][lid.x] + b[row];
        y[1u * p.rows + row] = bt[1][lid.x] + b[row];
        y[2u * p.rows + row] = bt[2][lid.x] + b[row];
        y[3u * p.rows + row] = bt[3][lid.x] + b[row];
        y[4u * p.rows + row] = bt[4][lid.x] + b[row];
        y[5u * p.rows + row] = bt[5][lid.x] + b[row];
    }
}
"#;

/// Multi-token linear with each weight row read ONCE for up to 8 tokens.
/// y[t, r] = W[r,:] . X[t,:] + b[r]
/// W bf16 packed [rows, in_words]; X f32 [T<=8, in]; Y f32 [T, rows].
/// One warp per row, dispatch (rows/8, 1).
///
/// The plain GEMV_MROW dispatches one workgroup per (row, token), which
/// re-reads every weight row once per token. The DiT now runs one 6-row pass
/// per CFG step (cond + null batched), so without this kernel every forward
/// would read the whole 192 MB of weights six times instead of once.
pub const GEMV_MROW_T8: &str = r#"
struct Params { rows: u32, in_words: u32, n_tok: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;   // [T, in]
@group(0) @binding(3) var<storage, read> b: array<f32>;   // [rows]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [T, rows]
var<workgroup> red: array<array<array<f32, 32>, 8>, 8>;

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
    var a4 = 0.0;
    var a5 = 0.0;
    var a6 = 0.0;
    var a7 = 0.0;
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
            if (p.n_tok > 4u) { a4 = a4 + lo * x[4u * in_len + x0] + hi * x[4u * in_len + x0 + 1u]; }
            if (p.n_tok > 5u) { a5 = a5 + lo * x[5u * in_len + x0] + hi * x[5u * in_len + x0 + 1u]; }
            if (p.n_tok > 6u) { a6 = a6 + lo * x[6u * in_len + x0] + hi * x[6u * in_len + x0 + 1u]; }
            if (p.n_tok > 7u) { a7 = a7 + lo * x[7u * in_len + x0] + hi * x[7u * in_len + x0 + 1u]; }
            wi = wi + 32u;
        }
    }
    red[warp][0][lane] = a0;
    red[warp][1][lane] = a1;
    red[warp][2][lane] = a2;
    red[warp][3][lane] = a3;
    red[warp][4][lane] = a4;
    red[warp][5][lane] = a5;
    red[warp][6][lane] = a6;
    red[warp][7][lane] = a7;
    workgroupBarrier();
    var s = 16u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) {
            red[warp][0][lane] = red[warp][0][lane] + red[warp][0][lane + s];
            red[warp][1][lane] = red[warp][1][lane] + red[warp][1][lane + s];
            red[warp][2][lane] = red[warp][2][lane] + red[warp][2][lane + s];
            red[warp][3][lane] = red[warp][3][lane] + red[warp][3][lane + s];
            red[warp][4][lane] = red[warp][4][lane] + red[warp][4][lane + s];
            red[warp][5][lane] = red[warp][5][lane] + red[warp][5][lane + s];
            red[warp][6][lane] = red[warp][6][lane] + red[warp][6][lane + s];
            red[warp][7][lane] = red[warp][7][lane] + red[warp][7][lane + s];
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
        if (p.n_tok > 4u) { y[4u * p.rows + row] = red[warp][4][0] + bias; }
        if (p.n_tok > 5u) { y[5u * p.rows + row] = red[warp][5][0] + bias; }
        if (p.n_tok > 6u) { y[6u * p.rows + row] = red[warp][6][0] + bias; }
        if (p.n_tok > 7u) { y[7u * p.rows + row] = red[warp][7][0] + bias; }
    }
}
"#;

/// Tiled prefill GEMM: 48 tokens x 64 weight rows x 32 K per workgroup.
///
/// The generic prefill kernel dispatches one workgroup per (weight row, token).
/// It has excellent token parallelism, but each workgroup reads the row's
/// weights itself; the reads only stay cheap while they hit L2. Here a 16x16
/// workgroup owns three token rows and four output columns, and a full K tile
/// of weights is copied to shared memory once. Weight rows in the checkpoint
/// are [rows, K], so the copy transposes into shared [K, 64] for vec4 reads.
///
/// Requires rows % 64 == 0 and K % 32 == 0; other shapes keep the generic path.
pub const GEMM_PREFILL_48X64: &str = r#"
struct Params { rows: u32, k: u32, n_tok: u32, kg: u32, tok0: u32, }  // kg = K / 8
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;  // [rows, kg], 8 bf16 each
@group(0) @binding(2) var<storage, read> x: array<f32>;        // [T, K]
@group(0) @binding(3) var<storage, read> b: array<f32>;        // [rows]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;  // [T, rows]

var<workgroup> sb: array<f32, 2176>;       // [32, 68] = [K tile, 64 N], one pad
var<workgroup> sa: array<vec4<f32>, 432>;  // [48, 9] vec4  = [tokens, K tile], one pad

fn bf_lo(u: u32) -> f32 { return bitcast<f32>((u & 0xFFFFu) << 16u); }
fn bf_hi(u: u32) -> f32 { return bitcast<f32>(u & 0xFFFF0000u); }

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n0 = wg.x * 64u;
    let tok0 = p.tok0 + wg.y * 48u;
    let tid = li;
    let ktiles = p.k / 32u;
    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);

    for (var kt = 0u; kt < ktiles; kt = kt + 1u) {
        let k0 = kt * 32u;
        // 64 rows x 4 vec4-per-row = 256 vectors: one per invocation.
        let nloc = tid / 4u;
        let kv = tid % 4u;
        let wv = w[(n0 + nloc) * p.kg + k0 / 8u + kv];
        let base = kv * 8u;
        sb[(base + 0u) * 68u + nloc] = bf_lo(wv.x);
        sb[(base + 1u) * 68u + nloc] = bf_hi(wv.x);
        sb[(base + 2u) * 68u + nloc] = bf_lo(wv.y);
        sb[(base + 3u) * 68u + nloc] = bf_hi(wv.y);
        sb[(base + 4u) * 68u + nloc] = bf_lo(wv.z);
        sb[(base + 5u) * 68u + nloc] = bf_hi(wv.z);
        sb[(base + 6u) * 68u + nloc] = bf_lo(wv.w);
        sb[(base + 7u) * 68u + nloc] = bf_hi(wv.w);

        // 48 tokens x 8 vec4-per-token = 384 vectors.
        for (var i = tid; i < 384u; i = i + 256u) {
            let tok = i / 8u;
            let ak = i % 8u;
            let gtok = tok0 + tok;
            var v = vec4<f32>(0.0);
            if (gtok < p.n_tok) {
                let src = gtok * p.k + k0 + ak * 4u;
                v = vec4<f32>(x[src], x[src + 1u], x[src + 2u], x[src + 3u]);
            }
            sa[tok * 9u + ak] = v;
        }
        workgroupBarrier();

        for (var kv = 0u; kv < 8u; kv = kv + 1u) {
            let kb = kv * 4u;
            let ni = lid.x * 4u;
            let b0 = vec4<f32>(sb[(kb + 0u) * 68u + ni], sb[(kb + 0u) * 68u + ni + 1u], sb[(kb + 0u) * 68u + ni + 2u], sb[(kb + 0u) * 68u + ni + 3u]);
            let b1 = vec4<f32>(sb[(kb + 1u) * 68u + ni], sb[(kb + 1u) * 68u + ni + 1u], sb[(kb + 1u) * 68u + ni + 2u], sb[(kb + 1u) * 68u + ni + 3u]);
            let b2 = vec4<f32>(sb[(kb + 2u) * 68u + ni], sb[(kb + 2u) * 68u + ni + 1u], sb[(kb + 2u) * 68u + ni + 2u], sb[(kb + 2u) * 68u + ni + 3u]);
            let b3 = vec4<f32>(sb[(kb + 3u) * 68u + ni], sb[(kb + 3u) * 68u + ni + 1u], sb[(kb + 3u) * 68u + ni + 2u], sb[(kb + 3u) * 68u + ni + 3u]);
            let a0 = sa[lid.y * 9u + kv];
            let a1 = sa[(lid.y + 16u) * 9u + kv];
            let a2 = sa[(lid.y + 32u) * 9u + kv];
            c0 = c0 + a0.x*b0 + a0.y*b1 + a0.z*b2 + a0.w*b3;
            c1 = c1 + a1.x*b0 + a1.y*b1 + a1.z*b2 + a1.w*b3;
            c2 = c2 + a2.x*b0 + a2.y*b1 + a2.z*b2 + a2.w*b3;
        }
        workgroupBarrier();
    }

    let gn = n0 + lid.x * 4u;
    let bv = vec4<f32>(b[gn], b[gn + 1u], b[gn + 2u], b[gn + 3u]);
    let dst = gn;
    let t0 = tok0 + lid.y;
    let t1 = tok0 + lid.y + 16u;
    let t2 = tok0 + lid.y + 32u;
    if (t0 < p.n_tok) {
        let o = c0 + bv;
        let d = t0 * p.rows + dst;
        y[d] = o.x; y[d + 1u] = o.y; y[d + 2u] = o.z; y[d + 3u] = o.w;
    }
    if (t1 < p.n_tok) {
        let o = c1 + bv;
        let d = t1 * p.rows + dst;
        y[d] = o.x; y[d + 1u] = o.y; y[d + 2u] = o.z; y[d + 3u] = o.w;
    }
    if (t2 < p.n_tok) {
        let o = c2 + bv;
        let d = t2 * p.rows + dst;
        y[d] = o.x; y[d + 1u] = o.y; y[d + 2u] = o.z; y[d + 3u] = o.w;
    }
}
"#;

/// RoPE applied in place to q or k, one token per workgroup.
/// interlected=0: Qwen2 rotate-half (pairs hd, hd+32); =1: x_transformers
/// interleaved (pairs 2hd, 2hd+1). Angle tables [512, 32] f32 host-side.
/// pos = pos_base + wg.x; dispatch (1, 1) for decode, (T, 1) for prefill.
pub const ROPE: &str = r#"
// row_base shifts only the ROW addressed in the activation (and the table for
// the position stays pos_base + token): batched branches need positions
// 0..n-1 each while living at different rows of one buffer.
struct Params { n_heads: u32, pos_base: u32, interleaved: u32, row_base: u32, }
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
    let row = p.row_base + token;
    let c = cos_t[pos * 32u + (lid.x % 32u)];
    let s = sin_t[pos * 32u + (lid.x % 32u)];
    var idx = lid.x;
    loop {
        if (idx >= p.n_heads * 32u) { break; }
        let head = idx / 32u;
        let hd = idx % 32u;
        let base = row * p.n_heads * 64u + head * 64u;
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
    group: u32,        // 0 = one sequence; n = rows are n independent n-token
                       //     sequences packed back to back (CFG batching)
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;   // [T, 896]
@group(0) @binding(2) var<storage, read> kc: array<f32>;  // [n_kv*512*64]
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [T, 896]
// Scratch is indexed by (pos - lo), NOT by absolute position: with a sliding
// window only `window` positions are ever live, while absolute positions run
// to the length of the utterance. Indexing by pos made the VAE decoder (window
// 64, but 640+ frames of audio) write past these arrays -- silently, since
// wgpu does not bounds-check workgroup storage.
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
    var lo = select(0u, max(qpos, p.window - 1u) - (p.window - 1u), p.window > 0u);
    if (p.group > 0u) {
        // row t belongs to group (t / group); it attends only its own slice
        let g = t / p.group;
        lo = g * p.group;
        qpos = lo + p.group - 1u;
    }
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
            sc[pos - lo] = s;
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
            let wv = exp(sc[pos - lo] - m);
            wgt[pos - lo] = wv;
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
        acc = acc + wgt[pos - lo] * vc[kbase + pos * 64u + d];
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
    off: u32,       // where the cond prediction starts
    off_null: u32,  // where the null prediction starts
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
        let nc = nullp[p.off_null + i];
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

/// Assemble a [6, W] seq from four [W] rows: [a, b, c, d, b, c]. That is the
/// two-branch (cond, null) layout the CFG batching needs: each branch is
/// [y, x_hist, x_now] and only y differs between them.
pub const PACK6: &str = r#"
struct Params { w: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read> d: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>; // [6*W]
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wg.x * 256u + lid.x;
    if (i < p.w * 6u) {
        let row = i / p.w;
        let col = i % p.w;
        var v = b[col];
        if (row == 0u) { v = a[col]; }
        else if (row == 2u || row == 5u) { v = c[col]; }
        else if (row == 3u) { v = d[col]; }
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