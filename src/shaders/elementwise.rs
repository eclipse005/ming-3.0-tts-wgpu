//! Norm, activations, packing and the CFM Euler step.

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

/// GELU for software adapters. Same formula, but tanh is written as
/// 1 - 2/(exp(2|x|)+1) with the sign restored: D3D12 WARP lowers `tanh` to a
/// form that overflows for |argument| > ~44 (inf/inf), which showed up as a
/// 0.34 rel_l2 in the DiT block and as NaNs downstream. GPU drivers have a
/// saturating native tanh, so they keep [GELU_TANH] unchanged.
pub const GELU_TANH_STABLE: &str = r#"
struct Params { n: u32, }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

fn tanh_stable(x: f32) -> f32 {
    let ax = abs(x);
    let e = exp(-2.0 * ax);
    let t = (1.0 - e) / (1.0 + e);
    return select(-t, t, x >= 0.0);
}

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
        y[i] = 0.5 * v * (1.0 + tanh_stable(v * beta * (1.0 + v * v * kappa)));
    }
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
