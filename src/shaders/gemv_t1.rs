//! Single-token decode linears.

/// Single-token linear with a configurable number of lanes per output row and
/// `rpl` rows per lane group.
///
/// The parameters matter because the LLM decoder mixes two very different
/// shapes: the 4864x896 projections can fill the machine with any row width,
/// while the 896x4864 down-projection has only 896 rows and needs a narrow row
/// width to emit enough workgroups. Lane count also sets the barrier-tree depth
/// (log2). Measured on a P104 and on an Intel iGPU, this family is within 2% of
/// every other single-token layout, so one tuning serves both.
pub fn gemv_t1_vec4(lanes: u32, threads: u32, rpl: u32) -> String {
    assert!(lanes > 0 && 32 % lanes == 0, "lanes must divide a warp");
    assert!(threads.is_multiple_of(32) && threads.is_multiple_of(lanes));
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
