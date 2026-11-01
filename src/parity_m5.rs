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

#[cfg(test)]
mod long_prompt_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::qwen_lm::LlmEngine;
    use crate::weights::{bf16_widen, Weights};

    fn g(n: &str) -> crate::golden::Npy {
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        read_npy(&std::path::Path::new(&dir).join("m5_long").join(n)).unwrap()
    }

    /// The 36 params_test cases prefill 34-65 tokens. Long text prefills far
    /// more, so the first decode step has to be checked at a length the
    /// acceptance set never reaches: if the prefill drifts at 161 tokens, every
    /// latent of that utterance is conditioned on the wrong hidden state and no
    /// amount of downstream parity would show it.
    ///
    /// Goldens are the reference's `z_diff` captured on the first
    /// `FlowLoss.sample` call (tools/dump_z0.py), plus the exact ids.
    #[test]
    fn long_prompt_prefill_matches_python() {
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let table = weights.get("model.model.embed_tokens.weight").unwrap();
        let (vocab, dim) = (table.shape[0], table.shape[1]);
        for name in ["S3_zh60_one", "S6_zh200_cap", "S8_en260"] {
            let ids = g(&format!("{name}_ids.npy"));
            let want = g(&format!("{name}_z0.npy"));
            let mut eng = LlmEngine::load(gpu.clone(), ops_for(&gpu), &weights).unwrap();
            let t = ids.data.len();
            let mut embeds = vec![0f32; t * dim];
            for (i, &id) in ids.data.iter().enumerate() {
                let id = id as usize;
                assert!(id < vocab, "{name}: token id {id} >= vocab {vocab}");
                let off = id * dim * 2;
                for j in 0..dim {
                    let lo = table.data[off + 2 * j] as u32;
                    let hi = table.data[off + 2 * j + 1] as u32;
                    embeds[i * dim + j] = bf16_widen(lo | (hi << 8));
                }
            }
            let (z, prob) = eng.prefill_full(&embeds, t).unwrap();
            let e = rel_l2(&z, &want.data);
            println!("{name}: t={t} z rel_l2={e:.3e} stop_prob={:.4}", prob[1]);
            assert!(e < 1e-5, "{name}: prefill z rel_l2 {e:.3e} >= 1e-5");
        }
    }
}

#[cfg(test)]
mod long_audio_tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::vae::VaeEncoder;
    use crate::vae_dec::VaeDecoder;
    use crate::weights::Weights;

    fn g(n: &str) -> crate::golden::Npy {
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        read_npy(&std::path::Path::new(&dir).join("m5_long").join(n)).unwrap()
    }

    /// 160 latents = 640 decoder frames = 12.8 s of audio, i.e. past the 512
    /// positions the LLM decodes in.
    ///
    /// The VAE decoder used to borrow `KV_CAP` (the LLM's window) for its K/V
    /// buffers and rope table while writing the WHOLE utterance in one pass.
    /// Every waveform longer than 6.4 s therefore wrote past the cache and came
    /// out corrupted -- and no params_test case was long enough to notice.
    #[test]
    fn vae_decodes_past_the_llm_window() {
        let lat = g("long_dec_lat.npy");
        let want = g("long_dec_wav.npy");
        let t_lat = lat.shape[0];
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let dec = VaeDecoder::load(gpu.clone(), ops_for(&gpu), &weights).unwrap();
        let got = dec.decode(&lat.data, t_lat).unwrap();
        assert_eq!(got.len(), want.data.len(), "waveform length");
        let e = rel_l2(&got, &want.data);
        println!("decode {t_lat} latents ({:.1} s) rel_l2={e:.3e}", t_lat as f32 * 0.08);
        assert!(e < 1e-4, "long decode rel_l2 {e:.3e} >= 1e-4");
    }

    /// 12 s of prompt = 600 encoder frames (750 after the cls patches), past the
    /// same 512-position cache. Prompt latents for cloning are re-drawn with the
    /// reference's own RNG so the sampled latent can be compared directly.
    #[test]
    fn vae_encodes_past_the_llm_window() {
        let wav = g("long_enc_wav.npy");
        let want = g("long_enc_lat.npy");
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let enc = VaeEncoder::load(gpu.clone(), ops_for(&gpu), &weights).unwrap();
        let (mean, std, _n) = enc.encode_params(&wav.data).unwrap();
        let frames = mean.len() / 64;
        assert_eq!(frames * 64, want.data.len(), "latent frame count");
        let mut rng = crate::rng::Philox::new(1895);
        let raw = rng.draw(64 * frames);
        let mut lat = vec![0f32; mean.len()];
        for t in 0..frames {
            for c in 0..64 {
                let i = t * 64 + c;
                lat[i] = mean[i] + std[i] * raw[c * frames + t];
            }
        }
        let e = rel_l2(&lat, &want.data);
        println!("encode {} samples -> {frames} latents, rel_l2={e:.3e}", wav.data.len());
        assert!(e < 1e-4, "long encode rel_l2 {e:.3e} >= 1e-4");
    }
}
