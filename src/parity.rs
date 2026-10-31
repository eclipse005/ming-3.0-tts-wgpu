//! Minimal .npy reader (numpy.save v1.0, C order) + rel_l2 metric + the M0
//! single-layer parity tests against Python dumps.

use anyhow::{bail, Context, Result};

pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn read_npy(path: &std::path::Path) -> Result<Npy> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if &bytes[0..6] != b"\x93NUMPY" {
        bail!("not an npy file: {}", path.display());
    }
    let major = bytes[6];
    let (hdr_len, off) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[off..off + hdr_len])?;
    let descr = extract(header, "'descr': '")?;
    let fortran = extract(header, "'fortran_order': ")?;
    if fortran.trim() != "False" {
        bail!("fortran-order npy not supported");
    }
    let shape = parse_shape(&extract(header, "'shape': (")?);
    let dtype = match descr.as_str() {
        "<f4" => D::F32,
        "<f8" => D::F64,
        "<i8" => D::I64,
        "|b1" => D::Bool,
        other => bail!("unsupported npy dtype {other}"),
    };
    let payload = &bytes[off + hdr_len..];
    let data = match dtype {
        D::F32 => payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        D::F64 => payload
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        D::I64 => payload
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        D::Bool => payload.iter().map(|&b| b as f32).collect(),
    };
    Ok(Npy { shape, data })
}

enum D {
    F32,
    F64,
    I64,
    Bool,
}

fn extract<'a>(header: &'a str, key: &'a str) -> Result<String> {
    let start = header
        .find(key)
        .with_context(|| format!("header missing {key}"))?
        + key.len();
    let rest = &header[start..];
    let end = rest
        .find(|c| c == ',' || c == '}')
        .with_context(|| format!("unterminated {key}"))?;
    Ok(rest[..end]
        .trim()
        .trim_matches('\'')
        .trim_matches('"')
        .to_string())
}

fn parse_shape(s: &str) -> Vec<usize> {
    s.trim_end_matches(',')
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect()
}

pub fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    (num.sqrt() / den.sqrt().max(1e-30)) as f32
}

fn model_dir() -> std::path::PathBuf {
    std::env::var("MING_MODEL")
        .unwrap_or_else(|_| "D:/Ming-3.0-tts/Ming-3.0-tts-0.5b".into())
        .into()
}

fn golden(name: &str) -> Result<crate::parity::Npy> {
    let dir = std::env::var("MING_GOLDEN").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string()
    });
    read_npy(&std::path::Path::new(&dir).join(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::nn;
    use crate::weights::Weights;

    const N: usize = 896; // hidden size
    const EPS: f32 = 1e-6;

    #[test]
    fn rmsnorm_layer4_matches_python() {
        let h3 = golden("h3_hidden.npy").unwrap();
        let w = golden("layer4_norm_w.npy").unwrap();
        let out = golden("layer4_norm_out.npy").unwrap();
        assert_eq!(h3.data.len() % N, 0);
        let gpu = Gpu::new().unwrap();
        let got = nn::rmsnorm(&gpu, &h3.data, &w.data, N, EPS).unwrap();
        let e = rel_l2(&got, &out.data);
        println!("rmsnorm rel_l2 = {e:.3e}");
        assert!(e < 1e-5, "rmsnorm rel_l2 {e:.3e} >= 1e-5");
    }

    #[test]
    fn gemv_known_values() {
        let gpu = Gpu::new().unwrap();
        // W = [1 2 3 4] bf16, x = ones(4), b = 0 -> expect 10
        let w_bytes: Vec<u8> = vec![
            0x80, 0x3F, 0x00, 0x40, // bf16 1.0, 2.0
            0x40, 0x40, 0x80, 0x40, // bf16 3.0, 4.0
        ];
        let entry = crate::weights::TensorEntry {
            shape: vec![1, 4],
            dtype: crate::weights::DType::Bf16,
            data: w_bytes,
        };
        let packed = entry.to_bf16_packed_u32().unwrap();
        let got = nn::gemv_bf16w(&gpu, &packed, &[1.0, 1.0, 1.0, 1.0], &[0.0], 1, 4)
            .unwrap();
        println!("known gemv: got {:?} want [10.0]", got);
        assert!((got[0] - 10.0).abs() < 1e-4, "got {}", got[0]);

        // bias test: b = [5] -> expect 15
        let got_b = nn::gemv_bf16w(&gpu, &packed, &[1.0, 1.0, 1.0, 1.0], &[5.0], 1, 4)
            .unwrap();
        assert!((got_b[0] - 15.0).abs() < 1e-4, "got {}", got_b[0]);

        // scale test: 896 rows x 896 cols, W=1, x=1 -> every y = 896
        let big_w = crate::weights::TensorEntry {
            shape: vec![896, 896],
            dtype: crate::weights::DType::Bf16,
            data: std::iter::repeat([0x80u8, 0x3Fu8]) // bf16 1.0
                .take(896 * 896)
                .flatten()
                .collect(),
        };
        let big_packed = big_w.to_bf16_packed_u32().unwrap();
        let big_x = vec![1.0f32; 896];
        let big_bias = vec![0.0f32; 896];
        let big_got = nn::gemv_bf16w(&gpu, &big_packed, &big_x, &big_bias, 896, 896)
            .unwrap();
        assert!(
            big_got.iter().all(|&v| (v - 896.0).abs() < 1e-2),
            "y[0]={} y[447]={} y[895]={}",
            big_got[0],
            big_got[447],
            big_got[895]
        );

        // identity-W probe: y[r] = x[r] + 0, exposing any x/W index bug
        let mut idata = vec![0u8; 896 * 896 * 2];
        for r in 0..896 {
            let off = (r * 896 + r) * 2;
            idata[off] = 0x80; // bf16 1.0 LE = [0x80, 0x3F]
            idata[off + 1] = 0x3F;
        }
        let ident = crate::weights::TensorEntry {
            shape: vec![896, 896],
            dtype: crate::weights::DType::Bf16,
            data: idata,
        };
        let ident_packed = ident.to_bf16_packed_u32().unwrap();
        let x_inc: Vec<f32> = (0..896).map(|i| (i + 1) as f32).collect();
        let ident_got =
            nn::gemv_bf16w(&gpu, &ident_packed, &x_inc, &big_bias, 896, 896).unwrap();
        let mism: Vec<usize> = (0..896)
            .filter(|&r| (ident_got[r] - x_inc[r]).abs() > 1e-3)
            .collect();
        assert!(mism.is_empty(), "identity probe mismatches: {:?}", &mism[..mism.len().min(10)]);
    }

    #[test]
    fn q_proj_gemv_bf16w_matches_python() {
        let q_in = golden("layer4_norm_out.npy").unwrap();
        let q_out = golden("layer4_q_out.npy").unwrap();
        let q_w = golden("layer4_q_w.npy").unwrap();
        let q_b = golden("layer4_q_b.npy").unwrap();

        // Cross-check: bf16 weight unpacked from the checkpoint must equal the
        // Python fp32 dump bit-for-bit (both are bf16 values widened to f32).
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let key = weights
            .names_matching("model.model.layers.4.self_attn.q_proj.weight")
            .into_iter()
            .next()
            .unwrap()
            .clone();
        let packed = weights.get(&key).unwrap().to_bf16_packed_u32().unwrap();
        let widened = weights.get(&key).unwrap().to_f32().unwrap();
        assert_eq!(widened, q_w.data, "bf16 widen != python fp32 dump");

        let gpu = Gpu::new().unwrap();
        let rows = q_w.shape[0];
        let t = q_in.shape[0];
        let mut worst = 0.0f32;
        for r in 0..t {
            let x = &q_in.data[r * N..(r + 1) * N];
            let want = &q_out.data[r * N..(r + 1) * N];
            let got = nn::gemv_bf16w(&gpu, &packed, x, &q_b.data, rows, N).unwrap();
            let e = rel_l2(&got, want);
            if e > worst {
                worst = e;
            }
        }
        println!("q_proj gemv worst row rel_l2 = {worst:.3e} over {t} rows");
        assert!(worst < 1e-5, "gemv rel_l2 {worst:.3e} >= 1e-5");
    }
}

#[cfg(test)]
mod m1_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::qwen_lm::LlmEngine;
    use crate::weights::Weights;

    #[test]
    fn llm_replay_matches_python() {
        let pre = golden("m1_prefill_embeds.npy").unwrap();
        let step_in = golden("m1_step_inputs.npy").unwrap();
        let step_z = golden("m1_step_z.npy").unwrap();
        let step_p = golden("m1_step_prob.npy").unwrap();
        let t = pre.shape[0];
        let s = step_in.shape[0];
        assert_eq!(step_z.shape[0], s + 1, "z goldens = prefill + steps");

        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let mut eng = LlmEngine::load(gpu, &weights).unwrap();

        // prefill
        let prob0 = eng.prefill(&pre.data, t).unwrap();
        let z0 = golden_last_z(); // helper reads step_z row 0
        let _ = z0;
        println!("prefill stop prob = {:.6} vs python {:.6}", prob0[1], step_p.data[0]);
        assert!((prob0[1] - step_p.data[0]).abs() < 1e-4, "prefill prob mismatch");

        // decode steps
        let mut worst_z = 0.0f32;
        let mut worst_p = 0.0f32;
        let mut stop_step = None;
        for i in 0..s {
            let z = eng.step(&step_in.data[i * 896..(i + 1) * 896], t + i).unwrap();
            let want = &step_z.data[(i + 1) * 896..(i + 2) * 896];
            let e = rel_l2(&z.0, want);
            worst_z = worst_z.max(e);
            worst_p = worst_p.max((z.1[1] - step_p.data[i + 1]).abs());
            if stop_step.is_none() && z.1[1] > 0.5 && i + 1 > 3 {
                stop_step = Some(i + 1);
            }
            if i < 3 || i + 1 == s {
                println!("step {}: z rel_l2={e:.3e} prob={:.6} (py {:.6})",
                         i + 1, z.1[1], step_p.data[i + 1]);
            }
        }
        println!("worst z rel_l2={worst_z:.3e} worst prob abs={worst_p:.3e} stop_step={stop_step:?}");
        assert!(worst_z < 1e-5, "decode z rel_l2 {worst_z:.3e} >= 1e-5");
        assert!(worst_p < 1e-4, "stop prob abs {worst_p:.3e} >= 1e-4");
        assert_eq!(stop_step, Some(40), "stop step must match python");
    }

    fn golden_last_z() -> Vec<f32> {
        let z = golden("m1_step_z.npy").unwrap();
        z.data[0..896].to_vec()
    }
}
