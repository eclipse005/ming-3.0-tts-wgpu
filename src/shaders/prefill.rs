//! Tiled prefill GEMM: 64 rows x 48 tokens per workgroup.

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
