//! Multi-token (2..8) linears: warp-per-row, bf16 unpacked in-shader.

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
