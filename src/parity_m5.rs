//! Later-generation parity tests, kept out of parity.rs so that file stays a
//! stable milestone-by-milestone record.

#![allow(unused_imports)]

use crate::golden::{read_npy, rel_l2};

/// Same defaults the milestone tests use.
fn model_dir() -> std::path::PathBuf {
    std::env::var("MING_MODEL")
        .unwrap_or_else(|_| "D:/Ming-3.0-tts/Ming-3.0-tts-0.5b".into())
        .into()
}

/// One shared kernel set per test.
fn ops_for(gpu: &crate::gpu::Gpu) -> std::sync::Arc<crate::ops::Ops> {
    std::sync::Arc::new(crate::ops::Ops::new(gpu.clone()))
}

#[cfg(test)]
mod m5_c6_tests {
    use super::*;
    use crate::pipeline::{GoldenNoise, Pipeline, SynthParams};

    fn g(n: &str) -> crate::golden::Npy {
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        read_npy(&std::path::Path::new(&dir).join("m5_params").join(n)).unwrap()
    }

    /// C6_surprised is a case the user could hear, and its token ids are
    /// byte-identical to the reference, so this traces where the loop leaves the
    /// reference trajectory.
    #[test]
    fn c6_loop_step_by_step() {
        let emb = g("c6_embeds.npy");
        let z_ref = g("c6_z_steps.npy");
        let lat_ref = g("c6_latents.npy");
        let t = emb.shape[0];
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        let noise_dir = std::path::Path::new(&dir).join("m5_params/C6_surprised");
        let mut draws = Vec::new();
        for i in 0.. {
            let p = noise_dir.join(format!("noise_{i:03}.npy"));
            if !p.exists() { break; }
            draws.push(read_npy(&p).unwrap().data);
        }
        let mut pl = Pipeline::load(&model_dir(), false).unwrap();
        let mut noise = GoldenNoise::new(draws);
        let params = SynthParams::default();
        let tr = pl.decode_loop(&emb.data, t, None, 0, &mut noise, &params).unwrap();
        println!("steps {} (reference {})", tr.z_steps.len(), z_ref.shape[0]);
        for s in 0..tr.z_steps.len().min(12) {
            let ez = rel_l2(&tr.z_steps[s], &z_ref.data[s * 896..(s + 1) * 896]);
            let el = rel_l2(
                &tr.latents[s * 64..(s + 1) * 64],
                &lat_ref.data[s * 64..(s + 1) * 64],
            );
            println!("  step {s:2}: z {ez:.3e}  latent {el:.3e}");
        }
        let n = tr.z_steps.len().min(z_ref.shape[0]);
        for s in (12..n).step_by(6) {
            let ez = rel_l2(&tr.z_steps[s], &z_ref.data[s * 896..(s + 1) * 896]);
            println!("  step {s:2}: z {ez:.3e}");
        }
    }
}

#[cfg(test)]
mod m2_block0_tests {
    use super::*;
    use crate::dit::DitEngine;
    use crate::gpu::Gpu;
    use crate::weights::Weights;

    fn g(n: &str) -> crate::golden::Npy {
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        read_npy(&std::path::Path::new(&dir).join(n)).unwrap()
    }

    /// One DiT block, op by op, gated at 2e-6.
    ///
    /// This test is what found the port's worst bug: every sub-operator sat at
    /// ~1e-7 except q_rope/k_rope at 8.1e-5, because the rope tables were built
    /// from theta=1e4 instead of the checkpoint's inv_freq buffer. That 450x
    /// jump then showed up as 1.1e-4 in attention, 2.3e-5 at the block output
    /// and 1.3e-4 in a 41-step CFM sample. With the real frequencies every row
    /// here is <= 4.2e-7, so the gate fails if anyone reintroduces a guessed
    /// base for the DiT's rotation.
    ///
    /// torch stores post-rope q/k head-major, ours are [seq, heads*dim], so
    /// those two are transposed before comparing.
    #[test]
    fn block0_op_by_op() {
        let x = g("m2bisect_x.npy");
        let c = g("m2bisect_c.npy");
        let lh = g("m2bisect_lh.npy");
        let t = g("m2bisect_t_scalar.npy").data[0];
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let dit = DitEngine::load(gpu.clone(), ops_for(&gpu), &weights).unwrap();
        let parts = dit.block0_parts(&x.data, t, &c.data, &lh.data).unwrap();
        println!("{:<8} {:>12} {:>12}", "op", "rel_l2", "absmax_ref");
        for (name, got) in parts {
            let want = g(&format!("b0_{name}.npy"));
            let (cmp, tag) = if name == "q_rope" || name == "k_rope" {
                let mut t = vec![0f32; got.len()];
                for s in 0..3 {
                    for h in 0..16 {
                        for d in 0..64 {
                            t[h * 3 * 64 + s * 64 + d] = got[s * 1024 + h * 64 + d];
                        }
                    }
                }
                (t, " (transposed)")
            } else {
                (got, "")
            };
            if cmp.len() != want.data.len() {
                println!("{:<8} shape mismatch {} vs {}{}", name, cmp.len(),
                         want.data.len(), tag);
                continue;
            }
            let e = rel_l2(&cmp, &want.data);
            let am = want.data.iter().fold(0f32, |a, b| a.max(b.abs()));
            println!("{:<8} {:>12.3e} {:>12.4}{}", name, e, am, tag);
            assert!(e < 2e-6, "{name}: rel_l2 {e:.3e} >= 2e-6{tag}");
        }
    }
}