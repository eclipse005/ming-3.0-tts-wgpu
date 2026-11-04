//! RoPE, attention and the KV-cache write.

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

/// Scalar attention for software adapters (DX12 WARP / lavapipe).
///
/// Same maths as [ATTEND], but it deliberately avoids the two tree
/// reductions: some software rasterisers mis-execute workgroupBarrier inside
/// a loop and then read uninitialised scratch, which showed up as all-NaN
/// attention rows on WARP while q/k/v and the RMSNorm (unrolled barriers) were
/// fine. Each lane scans the score scratch itself -- 64x redundant max/sum, but
/// correct, and this path only ever runs on CPUs.
pub const ATTEND_SIMPLE: &str = r#"
struct Params {
    t_q: u32, q_pos_base: u32, heads_per_kv: u32, scale: f32,
    full: u32, qpos_max: u32, stride: u32, window: u32,
    cap: u32, group: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> sc: array<f32, 512>;

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
        let g = t / p.group;
        lo = g * p.group;
        qpos = lo + p.group - 1u;
    }
    let qbase = t * p.stride + head * 64u;
    let kbase = kvh * p.cap * 64u;
    let d = lid.x;

    // Scores: lane d owns positions d, d+64, ...
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
            sc[pos - lo] = s * p.scale;
        }
        pos = pos + 64u;
    }
    workgroupBarrier();

    // Every lane scans the window itself: no reduction, no barriers in loops.
    var m = -3.0e38;
    pos = lo;
    loop {
        if (pos > qpos) { break; }
        m = max(m, sc[pos - lo]);
        pos = pos + 1u;
    }
    var denom = 0.0;
    pos = lo;
    loop {
        if (pos > qpos) { break; }
        denom = denom + exp(sc[pos - lo] - m);
        pos = pos + 1u;
    }
    var acc = 0.0;
    pos = lo;
    loop {
        if (pos > qpos) { break; }
        acc = acc + (exp(sc[pos - lo] - m) / denom) * vc[kbase + pos * 64u + d];
        pos = pos + 1u;
    }
    y[qbase + d] = acc;
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
