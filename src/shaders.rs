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