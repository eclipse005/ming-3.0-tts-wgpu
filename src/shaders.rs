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
