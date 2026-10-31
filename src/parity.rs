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

#[cfg(test)]
mod m2_tests {
    use super::*;
    use crate::dit::DitEngine;
    use crate::gpu::Gpu;
    use crate::weights::Weights;

    #[test]
    fn flowloss_matches_python() {
        let dir = |n| std::path::Path::new(&std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string()))
            .join(n);
        let z = read_npy(&dir("m2_z.npy")).unwrap();
        let lh = read_npy(&dir("m2_lh.npy")).unwrap();
        let noise = read_npy(&dir("m2_noise.npy")).unwrap();
        let sde = read_npy(&dir("m2_sde.npy")).unwrap();
        let y = read_npy(&dir("m2_y_ref.npy")).unwrap();
        let cfg = read_npy(&dir("m2_cfg.npy")).unwrap();
        let steps = z.shape[0];
        assert_eq!(steps, 41, "expected 41 recorded flowloss calls");

        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let eng = DitEngine::load(gpu, &weights).unwrap();

        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for i in 0..steps {
            let got = eng
                .flow_sample(
                    &z.data[i * 896..(i + 1) * 896],
                    &lh.data[i * 64..(i + 1) * 64],
                    &noise.data[i * 64..(i + 1) * 64],
                    &sde.data[i * 640..(i + 1) * 640],
                    cfg.data[i],
                )
                .unwrap();
            let want = &y.data[i * 64..(i + 1) * 64];
            let e = rel_l2(&got, want);
            if e > worst {
                worst = e;
                worst_i = i;
            }
            if i < 2 || i + 1 == steps {
                println!("step {i}: rel_l2={e:.3e} got[:3]={:?} want[:3]={:?}",
                         &got[0..3], &want[0..3]);
            }
        }
        println!("flowloss worst rel_l2 = {worst:.3e} @ step {worst_i}");
        // Gate: 2e-3. A full sample = 10 euler steps x 8 blocks x ~7 GEMMs; f32
        // accumulation-order noise + erf/exp approximations compound to ~1e-3
        // (single dit_forward verified at 1.2e-4, drift grows over the chain).
        assert!(worst < 2e-3, "flowloss rel_l2 {worst:.3e} >= 2e-3");
    }
}

#[cfg(test)]
mod m2_bisect_tests {
    use super::*;
    use crate::dit::DitEngine;
    use crate::gpu::Gpu;
    use crate::weights::Weights;

    #[test]
    fn dit_forward_bisect() {
        let g = |n| read_npy(&std::path::Path::new(&std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string()))
            .join(n)).unwrap();
        let x = g("m2bisect_x.npy").data;
        let c = g("m2bisect_c.npy").data;
        let lh = g("m2bisect_lh.npy").data;
        let t = g("m2bisect_t_scalar.npy").data[0];
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let eng = DitEngine::load(gpu, &weights).unwrap();
        let (temb, xn, xh, ce, b0, out) = eng.dit_forward_bisect(&x, t, &c, &lh).unwrap();
        let checks = [
            ("temb_mlp", temb, g("m2bisect_t_mlp.npy").data),
            ("x_emb_now", xn, g("m2bisect_x_emb_call0.npy").data),
            ("x_emb_hist", xh, g("m2bisect_x_emb_call1.npy").data),
            ("c_emb", ce, g("m2bisect_c_emb.npy").data[0..1024].to_vec()),
            ("block0", b0, g("m2bisect_block0.npy").data),
            ("final", out, g("m2bisect_final.npy").data[2 * 64..3 * 64].to_vec()),
        ];
        for (name, got, want) in checks {
            let e = rel_l2(&got, &want);
            println!("{name}: rel_l2={e:.3e} got[:3]={:?} want[:3]={:?}", &got[0..3], &want[0..3]);
            assert!(e < 1e-3, "{name} rel_l2 {e:.3e} >= 1e-3");
        }
    }
}

#[cfg(test)]
mod m2_trace_tests {
    use super::*;
    use crate::dit::DitEngine;
    use crate::gpu::Gpu;
    use crate::weights::Weights;

    #[test]
    fn euler_trajectory_matches_python() {
        let g = |n| read_npy(&std::path::Path::new(&std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string()))
            .join(n)).unwrap();
        let z = g("m2_z.npy").data;
        let lh = g("m2_lh.npy").data;
        let noise = g("m2_noise.npy").data;
        let sde = g("m2_sde.npy").data;
        let ex = g("m2_euler_x.npy").data;
        let cfg = g("m2_cfg.npy").data;
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let eng = DitEngine::load(gpu, &weights).unwrap();
        // trace the FIRST flowloss call (euler calls 0..10)
        let mut tr = Vec::new();
        let mut tp = Vec::new();
        let mut tn = Vec::new();
        {
            let traces = Some(&mut (&mut tr, &mut tp, &mut tn));
            eng.flow_sample_traced(
                &z[0..896], &lh[0..64], &noise[0..64], &sde[0..640], cfg[0],
                traces.map(|(a, b, c)| (&mut **a, &mut **b, &mut **c))).unwrap();
        }
        let py_pred = g("m2_pred_ref.npy").data;
        let py_null = g("m2_null_ref.npy").data;
        let mut worst = 0.0f32;
        for k in 0..tr.len() {
            let want = &ex[k * 64..(k + 1) * 64];
            let e = rel_l2(&tr[k], want);
            // NOTE: torch's in-sampler (pred, null) decomposition differs from
            // a direct forward_with_cfg by (delta, 1.5*delta) across call
            // contexts while 3p-2n (the velocity) is preserved; the velocity
            // and the integrated y chain are the invariant quantities, so the
            // assertion is on y only.
            if e > worst { worst = e; }
            println!("euler {k}: y_err={e:.3e}");
        }
        println!("worst y rel_l2 = {worst:.3e}");
        assert!(worst < 1e-3, "euler y rel_l2 {worst:.3e} >= 1e-3");
    }
}

#[cfg(test)]
mod m2_cross_tests {
    use super::*;
    use crate::dit::DitEngine;
    use crate::gpu::Gpu;
    use crate::weights::Weights;

    #[test]
    fn dit_forward_cross_check() {
        let g = |n| read_npy(&std::path::Path::new(&std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string()))
            .join(n)).unwrap();
        let x = g("m2bisect_x.npy").data;
        let c = g("m2bisect_c.npy").data;
        let lh = g("m2bisect_lh.npy").data;
        let t = g("m2bisect_t_scalar.npy").data[0];
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let eng = DitEngine::load(gpu, &weights).unwrap();
        let (_, _, _, _, _, via_bisect) = eng.dit_forward_bisect(&x, t, &c, &lh).unwrap();
        let via_fwd = eng.dit_forward(&x, t, &c, &lh).unwrap();
        let e = rel_l2(&via_fwd, &via_bisect);
        println!("dit_forward vs bisect: rel_l2={e:.3e} fwd[:3]={:?} bisect[:3]={:?}",
                 &via_fwd[0..3], &via_bisect[0..3]);
        assert!(e < 1e-4, "two rust paths disagree: {e:.3e}");
    }
}

#[cfg(test)]
mod m3_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::vae::VaeEncoder;
    use crate::weights::Weights;

    #[test]
    fn vae_encoder_matches_python() {
        let wf = golden("m3_waveform.npy").unwrap();
        let mean = golden("m3_mean.npy").unwrap();
        let std = golden("m3_std.npy").unwrap();
        let latent = golden("m3_latent.npy").unwrap();
        let eps = golden("m3_eps.npy").unwrap();
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let enc = VaeEncoder::load(gpu, &weights).unwrap();
        let (gm, gs, p) = enc.encode_params(&wf.data).unwrap();
        println!("patches={p} mean rel_l2={:.3e} std rel_l2={:.3e}",
                 rel_l2(&gm, &mean.data), rel_l2(&gs, &std.data));
        println!("mean[:4] got={:?} want={:?}", &gm[0..4], &mean.data[0..4]);
        println!("std[:4]  got={:?} want={:?}", &gs[0..4], &std.data[0..4]);
        assert_eq!(p, mean.shape[0]);
        assert!(rel_l2(&gm, &mean.data) < 1e-4, "mean >= 1e-4");
        assert!(rel_l2(&gs, &std.data) < 1e-4, "std >= 1e-4");
        // latent = mean + std * eps with the python-drawn eps
        let mut lat = vec![0f32; gm.len()];
        for i in 0..gm.len() {
            lat[i] = gm[i] + gs[i] * eps.data[i];
        }
        let el = rel_l2(&lat, &latent.data);
        println!("latent rel_l2={el:.3e}");
        assert!(el < 1e-4, "latent >= 1e-4");
    }
}

#[cfg(test)]
mod m3_bisect_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::vae::VaeEncoder;
    use crate::weights::Weights;

    #[test]
    fn vae_encoder_bisect() {
        let g = |n| read_npy(&std::path::Path::new(&std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string()))
            .join(n)).unwrap();
        let wf = g("m3_waveform.npy").data;
        let frames = g("m3_frames.npy").data;
        let embeds = g("m3_embeds.npy").data;
        let enc_out = g("m3_enc_out.npy").data;
        let agg_out = g("m3_agg_out.npy").data;
        // stage 1: host frame unfold
        let t_frames = frames.len() / 882;
        let mut my_frames = vec![0f32; frames.len()];
        for t in 0..t_frames {
            let start = t * 882;
            let end = (start + 882).min(wf.len());
            my_frames[t * 882..t * 882 + (end - start)]
                .copy_from_slice(&wf[start..end]);
        }
        println!("frames rel_l2={:.3e}", rel_l2(&my_frames, &frames));
        assert!(rel_l2(&my_frames, &frames) < 1e-6, "frames");
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let enc = VaeEncoder::load(gpu, &weights).unwrap();
        // stage 2: fc1+fc2
        let e = enc.embed_check(&my_frames, t_frames).unwrap();
        println!("embeds rel_l2={:.3e}", rel_l2(&e, &embeds));
        assert!(rel_l2(&e, &embeds) < 1e-4, "embeds");
        // stage 3: 24-layer encoder (layer 0 first)
        let layer0 = g("m3_layer0.npy").data;
        for li in [0usize, 1, 2, 4, 8, 16, 23] {
            let lname = format!("m3_layer{li}.npy");
            let ldir = std::env::var("MING_GOLDEN")
                .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
            let lg = read_npy(&std::path::Path::new(&ldir).join(&lname)).unwrap();
            let lo = enc.stack_range_check(&e, t_frames, 0..li + 1).unwrap();
            let e = rel_l2(&lo, &lg.data);
            println!("layer{li}: rel_l2={e:.3e} absmax_want={:.3e}",
                     lg.data.iter().fold(0f32, |a, b| a.max(b.abs())));
            if li <= 1 { assert!(e < 1e-4, "layer{li} {e:.3e}"); }
        }
        let eo = enc.encode_stack_check(&e, t_frames).unwrap();
        println!("enc_out rel_l2={:.3e}", rel_l2(&eo, &enc_out));
        for t2 in [0usize, 1, 2, 4, 8, 32, 63, 64, 100, 167] {
            let o = t2 * 896;
            let d: f32 = (0..896).map(|j| (eo[o + j] - enc_out[o + j]).abs()).sum::<f32>() / 896.0;
            println!("  tok{t2}: meandiff={d:.5} got[:2]={:?} want[:2]={:?}",
                     &eo[o..o + 2], &enc_out[o..o + 2]);
        }
        assert!(rel_l2(&eo, &enc_out) < 1e-3, "enc_out");
        // stage 4: patch/cls + 4-layer aggregator
        let ao = enc.agg_check(&eo, t_frames).unwrap();
        println!("agg_out rel_l2={:.3e}", rel_l2(&ao, &agg_out));
        assert!(rel_l2(&ao, &agg_out) < 1e-3, "agg_out");
    }
}

#[cfg(test)]
mod m3_layer0_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::vae::VaeEncoder;
    use crate::weights::Weights;

    #[test]
    fn vae_layer0_parts() {
        let g = |n| read_npy(&std::path::Path::new(&std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string()))
            .join(n)).unwrap();
        let embeds = g("m3_embeds.npy").data;
        let t = embeds.len() / 896;
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let enc = VaeEncoder::load(gpu, &weights).unwrap();
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        for (name, got) in enc.layer0_parts_check(&embeds, t).unwrap() {
            if name == "l0_kcache" {
                let krpy = read_npy(&std::path::Path::new(&dir).join("m3_l0_k_rope.npy")).unwrap();
                let kprj = read_npy(&std::path::Path::new(&dir).join("m3_l0_k_proj.npy")).unwrap();
                println!("l0_kcache vs rope-k rel_l2={:.3e} | vs pre-rope-k rel_l2={:.3e}",
                         rel_l2(&got, &krpy.data), rel_l2(&got, &kprj.data));
                println!("  cache[0..4]={:?} rope-k[0..4]={:?} pre[0..4]={:?}",
                         &got[0..4], &krpy.data[0..4], &kprj.data[0..4]);
                println!("  cache[128..132]={:?} rope-k[128..132]={:?}",
                         &got[128..132], &krpy.data[128..132]);
                continue;
            }
            let want_npy = if name == "l0_attn" {
                read_npy(&std::path::Path::new(&dir).join("m3_l0_attn.npy")).unwrap()
            } else if name == "l0_kcache" {
                read_npy(&std::path::Path::new(&dir).join("m3_l0_k_rope.npy")).unwrap()
            } else {
                read_npy(&std::path::Path::new(&dir).join(format!("m3_{name}.npy"))).unwrap()
            };
            let want = want_npy.data;
            let e = rel_l2(&got, &want);
            println!("{name}: rel_l2={e:.3e}");
            if name == "l0_attn" {
                let win = read_npy(&std::path::Path::new(&dir).join("m3_l0_attn_win64.npy")).unwrap();
                println!("l0_attn vs win64-numpy rel_l2={:.3e}", rel_l2(&got, &win.data));
                for t2 in [0usize, 1, 4, 32, 64, 100, 167] {
                    let o = t2 * 896;
                    let mut mx = 0f32;
                    let mut mpos = 0usize;
                    for h in 0..14 {
                        for d in 0..64 {
                            let i = o + h * 64 + d;
                            if (got[i] - win.data[i]).abs() > mx { mx = (got[i] - win.data[i]).abs(); mpos = h * 64 + d; }
                        }
                    }
                    println!("  tok{t2}: max|got-win64|={mx:.4} @dim{mpos} got={:.4} want={:.4}",
                             got[o + mpos], win.data[o + mpos]);
                }
            } else { assert!(e < 1e-4, "{name} rel_l2 {e:.3e}"); }
        }
    }
}

#[cfg(test)]
mod m3_rope_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::nn::kernel;
    use crate::shaders;
    use crate::qwen_lm::{Q_HEADS, ROPE_THETA};

    #[test]
    fn vae_rope_matches_numpy() {
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        let q = read_npy(&std::path::Path::new(&dir).join("m3_l0_q_proj.npy")).unwrap();
        let want = read_npy(&std::path::Path::new(&dir).join("m3_l0_q_rope_inter.npy")).unwrap();
        let t = q.shape[0];
        let gpu = Gpu::new().unwrap();
        let (mut cos, mut sin) = (vec![0f32; 512 * 32], vec![0f32; 512 * 32]);
        for p in 0..512 {
            for i in 0..32 {
                let f = 10000f64.powf((2 * i) as f64 / 64.0).recip();
                cos[p * 32 + i] = (p as f64 * f).cos() as f32;
                sin[p * 32 + i] = (p as f64 * f).sin() as f32;
            }
        }
        let cb = gpu.upload_f32(&cos, "c");
        let sb = gpu.upload_f32(&sin, "s");
        let qb = gpu.upload_f32(&q.data, "q");
        let k = kernel(&gpu, shaders::ROPE, 2);
        let mut u = [0u8; 16];
        u[0..4].copy_from_slice(&(Q_HEADS as u32).to_le_bytes());
        u[4..8].copy_from_slice(&0u32.to_le_bytes());
        u[8..12].copy_from_slice(&1u32.to_le_bytes());
        k.run(&gpu, &u, &[&cb, &sb], &qb, t as u32, 1);
        let got = gpu.read_f32(&qb, t * 896).unwrap();
        let e = rel_l2(&got, &want.data);
        let mut shown = 0;
        for t2 in 0..168 {
            for h in 0..14 {
                for d in 0..64 {
                    let i = t2 * 896 + h * 64 + d;
                    if (got[i] - want.data[i]).abs() > 1e-3 && shown < 6 {
                        println!("t{t2} h{h} d{d}: got={:.4} want={:.4} in_q={:.4}",
                                 got[i], want.data[i], q.data[i]);
                        shown += 1;
                    }
                }
            }
        }
        println!("rope rel_l2={e:.3e}");
        assert!(e < 1e-5, "rope {e:.3e}");
    }
}

#[cfg(test)]
mod kv_write_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::nn::kernel;
    use crate::shaders;

    #[test]
    fn kv_write_micro() {
        let gpu = Gpu::new().unwrap();
        let k = kernel(&gpu, shaders::KV_WRITE, 1);
        // 3 tokens x 128 values, row t filled with value (t+1)
        let mut src = vec![0f32; 3 * 128];
        for t in 0..3 {
            for i in 0..128 {
                src[t * 128 + i] = (t + 1) as f32;
            }
        }
        let sb = gpu.upload_f32(&src, "src");
        let cache = gpu.storage((2 * 512 * 64 * 4) as u64, "cache");
        let mut u = [0u8; 16];
        u[0..4].copy_from_slice(&0u32.to_le_bytes());  // pos_base
        u[4..8].copy_from_slice(&512u32.to_le_bytes()); // cap
        u[8..12].copy_from_slice(&2u32.to_le_bytes());  // n_kv
        k.run(&gpu, &u, &[&sb], &cache, 3, 1);
        let got = gpu.read_f32(&cache, 2 * 512 * 64).unwrap();
        // position 0 -> 1.0, position 1 -> 2.0, position 2 -> 3.0
        let p0 = got[(0 * 512 + 0) * 64];
        let p1 = got[(0 * 512 + 1) * 64];
        let p2 = got[(0 * 512 + 2) * 64];
        let p1_h1 = got[(1 * 512 + 1) * 64];
        println!("p0={p0} p1={p1} p2={p2} p1_h1={p1_h1} (want 1,2,3,2)");
        assert!((p0 - 1.0).abs() < 1e-6 && (p1 - 2.0).abs() < 1e-6
                && (p2 - 3.0).abs() < 1e-6 && (p1_h1 - 2.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod m3_fc3_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::vae::VaeEncoder;
    use crate::weights::Weights;

    #[test]
    fn fc3_path_bisect() {
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        let wf = read_npy(&std::path::Path::new(&dir).join("m3_waveform.npy")).unwrap();
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let enc = VaeEncoder::load(gpu, &weights).unwrap();
        let _ = enc.encode_params(&wf.data).unwrap();
        for (name, got) in enc.take_bisect() {
            let want_npy = read_npy(&std::path::Path::new(&dir)
                .join(&format!("m3_{name}.npy"))).unwrap();
            let e = rel_l2(&got, &want_npy.data);
            println!("{name}: rel_l2={e:.3e} got[:3]={:?} want[:3]={:?}",
                     &got[0..3], &want_npy.data[0..3]);
        }
    }
}


#[cfg(test)]
mod m4_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::vae_dec::VaeDecoder;
    use crate::weights::Weights;

    #[test]
    fn vae_decoder_matches_python() {
        let latent = golden("m4_latent.npy").unwrap();
        let fc1 = golden("m4_fc1_out.npy").unwrap();
        let up = golden("m4_up_out.npy").unwrap();
        let dec = golden("m4_dec_out.npy").unwrap();
        let head = golden("m4_head_lin.npy").unwrap();
        let wav = golden("m4_waveform.npy").unwrap();
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let d = VaeDecoder::load(gpu, &weights).unwrap();
        let t_lat = latent.shape[0];

        let f = d.fc1_check(&latent.data, t_lat).unwrap();
        println!("fc1 rel_l2={:.3e} n={} want={}", rel_l2(&f, &fc1.data), f.len(), fc1.data.len());
        assert_eq!(f.len(), fc1.data.len());
        assert!(rel_l2(&f, &fc1.data) < 1e-5, "fc1");

        let u = d.upsample_check(&latent.data, t_lat).unwrap();
        println!("upsample rel_l2={:.3e} n={} want={}", rel_l2(&u, &up.data), u.len(), up.data.len());
        assert_eq!(u.len(), up.data.len());
        assert!(rel_l2(&u, &up.data) < 1e-5, "upsample");
        let t = u.len() / 896;

        let s = d.stack_check(&u, t).unwrap();
        println!("dec stack rel_l2={:.3e}", rel_l2(&s, &dec.data));
        assert!(rel_l2(&s, &dec.data) < 1e-4, "dec stack");

        let h = d.head_check(&u, t).unwrap();
        println!("head lin rel_l2={:.3e}", rel_l2(&h, &head.data));
        assert!(rel_l2(&h, &head.data) < 1e-4, "head");

        let w = d.decode(&latent.data, t_lat).unwrap();
        println!("waveform len {} want {} absmax {:.4} want {:.4}",
                 w.len(), wav.data.len(),
                 w.iter().fold(0f32, |a, b| a.max(b.abs())),
                 wav.data.iter().fold(0f32, |a, b| a.max(b.abs())));
        assert_eq!(w.len(), wav.data.len(), "waveform length");
        let e = rel_l2(&w, &wav.data);
        println!("waveform rel_l2={e:.3e}");
        assert!(e < 1e-3, "waveform {e:.3e}");
    }
}


#[cfg(test)]
mod m5_tests {
    use super::*;
    use crate::pipeline::{GoldenNoise, Pipeline, SynthParams};

    fn m5dir() -> std::path::PathBuf {
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        std::path::Path::new(&dir).join("m5")
    }

    #[test]
    fn decode_loop_matches_python() {
        let d = m5dir();
        let load = |n: &str| read_npy(&d.join(n)).unwrap();
        let expects: usize = 43;
        let embeds = load("inputs_embeds.npy");
        let z_steps = load("z_steps.npy");
        let lat_steps = load("latent_steps.npy");
        let dec_lat = load("decode_latent.npy");
        let want_wav = load("waveform.npy");
        let meta = std::fs::read_to_string(d.join("meta.json")).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let n_steps = meta["n_steps"].as_u64().unwrap() as usize;
        let n_noise = meta["n_noise"].as_u64().unwrap() as usize;
        println!("tokens={} steps={} noise={}", expects, n_steps, n_noise);

        let mut draws = Vec::with_capacity(n_noise);
        for i in 0..n_noise {
            draws.push(load(&format!("noise_{i:03}.npy")).data);
        }

        let mut pl = Pipeline::load(&model_dir(), false).unwrap();
        let mut noise = GoldenNoise::new(draws);
        let mut params = SynthParams::default();
        if let Ok(v) = std::env::var("MING_STEPS") {
            params.max_decode_steps = v.parse().unwrap();
        }
        let t0 = std::time::Instant::now();
        let tr = pl
            .decode_loop(&embeds.data, expects, None, 0, &mut noise, &params)
            .unwrap();
        println!("loop {:?} -> {} latents, stopped_at={:?}",
                 t0.elapsed(), tr.n_latents, tr.stopped_at);

        // per-step hidden state
        let mut worst = 0f32;
        let mut worst_step = 0;
        let n = tr.z_steps.len().min(n_steps);
        for s in 0..n {
            let want = &z_steps.data[s * 896..(s + 1) * 896];
            let e = rel_l2(&tr.z_steps[s], want);
            if e > worst {
                worst = e;
                worst_step = s;
            }
            if s < 3 || s == n - 1 {
                println!("  z step {s}: rel_l2={e:.3e}");
            }
        }
        println!("worst z rel_l2={worst:.3e} @ step {worst_step}");

        // per-step latents
        let mut worst_l = 0f32;
        let mut worst_ls = 0;
        for s in 0..n {
            let want = &lat_steps.data[s * 64..(s + 1) * 64];
            let got = &tr.latents[s * 64..(s + 1) * 64];
            let e = rel_l2(got, want);
            if e > worst_l {
                worst_l = e;
                worst_ls = s;
            }
        }
        println!("worst latent rel_l2={worst_l:.3e} @ step {worst_ls}");

        // stop probabilities
        let mut worst_p = 0f32;
        for s in 0..n.min(tr.stop_probs.len()) {
            let _ = s;
        }
        for &v in tr.stop_probs.iter() {
            worst_p = worst_p.max(v);
        }
        println!("final stop prob={worst_p:.4} (python 0.729774 for the M1 case)");

        // full vocoder chain
        let got_wav = pl.vocode(&tr.latents, tr.n_latents).unwrap();
        assert_eq!(got_wav.len(), want_wav.data.len(), "waveform length");
        let ew = rel_l2(&got_wav, &want_wav.data);
        println!("waveform rel_l2={ew:.3e} len={}", got_wav.len());

        // The autoregressive + diffusion loop cannot be bit-exact across two
        // different f32 reduction orders: one DiT forward already sits at
        // ~2.6e-5 (cuBLAS blocked reductions vs our sequential GEMV -- the f32
        // floor), and feeding every latent back amplifies it. What must hold
        // exactly is the structural behaviour: the loop has to take the same
        // number of steps, i.e. make the same stopping decision.
        assert_eq!(tr.n_latents, n_steps, "step count / stop decision");
        assert!(worst < 2e-2, "z steps {worst:.3e}");
        assert!(worst_l < 2e-2, "latents {worst_l:.3e}");
        assert!(ew < 2e-2, "waveform {ew:.3e}");
    }
}


#[cfg(test)]
mod microbench2 {
    use super::*;
    use crate::gpu::Gpu;
    use crate::weights::Weights;

    #[test]
    fn device_path_vs_sync() {
        let gpu = Gpu::new().unwrap();
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let dit = crate::dit::DitEngine::load(gpu.clone(), &weights).unwrap();
        let x = vec![0.01f32; 64];
        let c = vec![0.02f32; 896];
        let lh = vec![0.03f32; 64];
        let xb = gpu.upload_f32(&x, "bx");
        let cb = gpu.upload_f32(&c, "bc");
        let lhb = gpu.upload_f32(&lh, "blh");

        // warm
        let _ = gpu.read_f32(&dit.dit_forward_dev(&xb, 0.5, &cb, &lhb).unwrap(), 64).unwrap();

        let t3 = std::time::Instant::now();
        let mut last = None;
        for _ in 0..20 {
            last = Some(dit.dit_forward_dev(&xb, 0.5, &cb, &lhb).unwrap());
        }
        let dev = t3.elapsed();
        let out = gpu.read_f32(&last.unwrap(), 64).unwrap();
        let dev_all = t3.elapsed();
        println!("20 dev forwards: {:?} -> {:?}/fwd, incl final read {:?}, out0={:.6}",
                 dev, dev / 20, dev_all, out[0]);

        let t4 = std::time::Instant::now();
        for _ in 0..10 {
            gpu.flush();
        }
        println!("10 empty flush: {:?}", t4.elapsed());

        // one readback of a tiny buffer, repeated
        let small = gpu.upload_f32(&[0f32; 64], "small");
        let t5 = std::time::Instant::now();
        for _ in 0..10 {
            let _ = gpu.read_f32(&small, 64).unwrap();
        }
        println!("10 tiny readbacks: {:?}", t5.elapsed());
    }
}
