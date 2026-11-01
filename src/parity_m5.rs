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

#[cfg(test)]
mod m2_pair_tests {
    use super::*;
    use crate::dit::{DitEngine, ZC};
    use crate::gpu::Gpu;
    use crate::weights::Weights;

    fn g(n: &str) -> crate::golden::Npy {
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        read_npy(&std::path::Path::new(&dir).join(n)).unwrap()
    }

    /// The CFG batching runs both branches in ONE 6-row pass. Every row must see
    /// exactly the arithmetic it saw when the branches were two separate
    /// forwards -- otherwise the "same audio as the reference" property drifts
    /// silently.
    #[test]
    fn cfg_pair_matches_two_single_forwards() {
        let x = g("m2bisect_x.npy");
        let c = g("m2bisect_c.npy");
        let lh = g("m2bisect_lh.npy");
        let t = g("m2bisect_t_scalar.npy").data[0];
        let weights = Weights::load(&model_dir().join("model.safetensors")).unwrap();
        let gpu = Gpu::new().unwrap();
        let dit = DitEngine::load(gpu.clone(), ops_for(&gpu), &weights).unwrap();
        let xb = gpu.upload_f32(&x.data, "x");
        let cb = gpu.upload_f32(&c.data, "c");
        let lhb = gpu.upload_f32(&lh.data, "lh");
        let zero = gpu.upload_f32(&vec![0f32; 896], "zero_c");

        let cond = dit.dit_forward_dev(&xb, t, &cb, &lhb).unwrap();
        let null = dit.dit_forward_dev(&xb, t, &zero, &lhb).unwrap();
        let got_cond = gpu.read_f32(&cond, 3 * ZC).unwrap();
        let got_null = gpu.read_f32(&null, 3 * ZC).unwrap();

        let pair = dit.dit_forward_pair_dev(&xb, t, &cb, &lhb).unwrap();
        let all = gpu.read_f32(&pair, 6 * ZC).unwrap();
        let (pair_cond, pair_null) = all.split_at(3 * ZC);

        let ec = rel_l2(pair_cond, &got_cond);
        let en = rel_l2(pair_null, &got_null);
        let maxc = pair_cond.iter().zip(&got_cond).fold(0f32, |a, (p, q)| a.max((p - q).abs()));
        let maxn = pair_null.iter().zip(&got_null).fold(0f32, |a, (p, q)| a.max((p - q).abs()));
        println!("cond: rel={ec:.3e} maxabs={maxc:.3e}   null: rel={en:.3e} maxabs={maxn:.3e}");
        assert!(ec < 1e-6 && en < 1e-6, "batched branch drifted: rel {ec:.3e} / {en:.3e}");
    }
}

#[cfg(test)]
mod gemm_bench {
    use super::*;
    use crate::gpu::Gpu;
    use crate::nn::kernel;
    use crate::ops::{Lin, Ops};
    use crate::shaders;

    fn lin_of(gpu: &Gpu, rows: usize, inf: usize) -> Lin {
        // deterministic pseudo-random bf16 weights, packed two per u32
        let words = rows * inf / 2;
        let mut w = Vec::with_capacity(words);
        let mut state: u32 = 0x1234_5678;
        for _ in 0..words {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // keep both bf16 halves finite: exp 0x7e, random mantissa byte
            let lo = (state & 0xFF) | 0x3F00;
            let hi = ((state >> 16) & 0xFF) | 0x3F00;
            w.push(lo | (hi << 16));
        }
        let b = vec![0f32; rows];
        Lin {
            w: gpu.upload_u32(&w, "bench.w"),
            b: gpu.upload_f32(&b, "bench.b"),
            rows,
            in_features: inf,
            id: 0,
        }
    }

    /// The vec4 single-token kernel must reproduce the multi-token kernel's first
    /// row exactly (same weights, same x row).
    #[test]
    fn vec4_single_matches_multi() {
        let gpu = Gpu::new().unwrap();
        let ops = Ops::new(gpu.clone());
        for (rows, inf) in [(896usize, 896usize), (4864, 896), (1024, 1024), (4096, 1024)] {
            let lin = lin_of(&gpu, rows, inf);
            let mut xv = vec![0f32; 2 * inf];
            for j in 0..inf {
                xv[j] = ((j % 17) as f32 - 8.0) * 0.031;
                xv[inf + j] = xv[j];
            }
            let xb = gpu.upload_f32(&xv, "iso.x");
            let y1 = ops.linear(&lin, &xb, 1).unwrap();
            let y2 = ops.linear(&lin, &xb, 2).unwrap();
            let a = gpu.read_f32(&y1, rows).unwrap();
            let b = gpu.read_f32(&y2, 2 * rows).unwrap();
            let e = rel_l2(&a, &b[..rows]);
            println!("  {rows}x{inf}: vec4 vs t2 rel_l2={e:.3e}");
            assert!(e < 1e-5, "{rows}x{inf}: {e:.3e}");
        }
    }

    /// The vec4 six-token kernel must reproduce the 8-token kernel's rows.
    #[test]
    fn vec4_six_matches_t8() {
        let gpu = Gpu::new().unwrap();
        let ops = Ops::new(gpu.clone());
        for (rows, inf) in [(1024usize, 1024usize), (4096, 1024), (1024, 4096)] {
            let lin = lin_of(&gpu, rows, inf);
            let mut xv = vec![0f32; 8 * inf];
            for j in 0..8 * inf {
                xv[j] = ((j % 23) as f32 - 11.0) * 0.017;
            }
            let xb = gpu.upload_f32(&xv, "iso6.x");
            let y6 = ops.linear(&lin, &xb, 6).unwrap();
            let y8 = ops.linear(&lin, &xb, 8).unwrap();   // 8 tokens -> T8 kernel
            let a = gpu.read_f32(&y6, 6 * rows).unwrap();
            let b = gpu.read_f32(&y8, 8 * rows).unwrap();
            let e = rel_l2(&a, &b[..6 * rows]);
            println!("  {rows}x{inf}: v4t6 vs t8 rel_l2={e:.3e}");
            assert!(e < 1e-5, "{rows}x{inf}: {e:.3e}");
        }
    }

    /// Tiled prefill must match the generic kernel and should be faster.
    #[test]
    fn prefill_tile_matches_generic() {
        let gpu = Gpu::new().unwrap();
        let old_env = std::env::var("MING_PREFILL").ok();
        std::env::set_var("MING_PREFILL", "base");
        let generic = Ops::new(gpu.clone());
        match old_env {
            Some(v) => std::env::set_var("MING_PREFILL", v),
            None => std::env::remove_var("MING_PREFILL"),
        }
        let tiled = Ops::new(gpu.clone());
        let t = 43;
        for (rows, inf) in [(896usize, 896usize), (4864, 896), (896, 4864), (1024, 1024)] {
            let lin = lin_of(&gpu, rows, inf);
            let xv: Vec<f32> = (0..t * inf)
                .map(|j| ((j % 31) as f32 - 15.0) * 0.023)
                .collect();
            let x = gpu.upload_f32(&xv, "prefill.x");
            let ya = generic.linear(&lin, &x, t).unwrap();
            let yb = tiled.linear(&lin, &x, t).unwrap();
            let a = gpu.read_f32(&ya, t * rows).unwrap();
            let b = gpu.read_f32(&yb, t * rows).unwrap();
            let e = rel_l2(&a, &b);
            println!("  prefill {rows}x{inf} t={t}: rel_l2={e:.3e}");
            assert!(e < 1e-5, "{rows}x{inf}: {e:.3e}");

            for _ in 0..2 {
                std::hint::black_box(generic.linear(&lin, &x, t).unwrap());
                std::hint::black_box(tiled.linear(&lin, &x, t).unwrap());
            }
            gpu.flush();
            let iters = 4;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(generic.linear(&lin, &x, t).unwrap());
            }
            gpu.flush();
            let a_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(tiled.linear(&lin, &x, t).unwrap());
            }
            gpu.flush();
            let b_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("    generic {a_us:8.1} us | tiled {b_us:8.1} us | {:.2}x", a_us / b_us);
        }
    }

    /// What the linear kernel actually sustains, against the ~268 GB/s that the
    /// qwen3-asr port measured on this same card for the same kind of GEMV.
    #[test]
    fn linear_bandwidth() {
        let gpu = Gpu::new().unwrap();
        let ops = Ops::new(gpu.clone());
        for (rows, inf, tok) in [
            (4096usize, 1024usize, 6usize),
            (1024, 4096, 6),
            (1024, 1024, 6),
            (896, 896, 1),
            (4864, 896, 1),
            (896, 4864, 1),
        ] {
            let lin = lin_of(&gpu, rows, inf);
            let x = gpu.upload_f32(&vec![0.5f32; tok * inf], "bench.x");
            for _ in 0..3 {
                let y = ops.linear(&lin, &x, tok);
                std::hint::black_box(&y);
            }
            gpu.flush();
            let iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let y = ops.linear(&lin, &x, tok);
                std::hint::black_box(&y);
            }
            gpu.flush();
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            let mb = (rows * inf * 2) as f64 / 1e6;
            println!(
                "  {rows:5}x{inf:<5} t={tok} : {:6.1} us, {:.0} GB/s  ({mb:.1} MB)",
                dt * 1e6,
                mb / 1e3 / dt
            );
        }
        let _ = kernel;
        let _ = shaders::PACK3;
    }
}