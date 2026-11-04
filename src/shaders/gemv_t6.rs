//! Six-token batch linears (the DiT's CFG pair).

/// Six-token linear with fully vectorised loads and `rpl` output rows per lane
/// group.
///
/// The scalar kernels issue four `u32` weight loads and 24 scalar activation
/// loads per 8-wide K granule, i.e. 28 load instructions for 48 FMAs.
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
    assert!(threads.is_multiple_of(32) && threads.is_multiple_of(lanes));
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
