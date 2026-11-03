//! The --device cpu entry point: the same fragment loop as the wgpu path.

use anyhow::{bail, Result};

use super::{CpuPipeline, ZC};
use crate::qwen_lm::HIDDEN;

/// --device cpu: same CLI surface, host pipeline. Prompt/clone conditioning is
/// not wired yet (it needs the VAE encoder), and says so instead of silently
/// producing the wrong voice.
pub fn run(a: &crate::synth::Args, model_dir: &std::path::Path) -> Result<()> {
    use crate::tasks;

    let tok = crate::tok::Tok::load(model_dir)?;
    let controls = {
        let c = tasks::Controls {
            speed: a.speed.clone(),
            pitch: a.pitch.clone(),
            volume: a.volume.clone(),
            emotion: a.emotion.clone(),
            dialect: a.dialect.clone(),
            style: a.style.clone(),
            ip: a.ip.clone(),
        };
        if c.speed.is_none()
            && c.pitch.is_none()
            && c.volume.is_none()
            && c.emotion.is_none()
            && c.dialect.is_none()
            && c.style.is_none()
            && c.ip.is_none()
        {
            None
        } else {
            Some(c)
        }
    };
    let req = tasks::build_task_generation_kwargs(
        &a.task,
        &a.text,
        a.language_id.as_deref(),
        a.instruction.as_deref(),
        controls.as_ref(),
        a.instruction_schema.as_deref(),
        a.prompt_wav_path.first().map(|s| s.as_str()),
        a.prompt_text.as_deref(),
        a.spk_emb_path.first().map(|s| s.as_str()),
        a.use_spk_emb,
        a.use_zero_spk_emb,
        a.cfg,
        a.sigma,
        a.temperature,
    )?;
    let fragments = crate::synth::prepare_text(&tok, &req, a)?;
    // STYLE re-conditions every later fragment on the first generated chunk,
    // exactly like ming_tts.synthesize (see synth.rs).
    let reference_first_fragment = req.taskname == "STYLE" && fragments.len() > 1;
    let mut rng: Box<dyn crate::pipeline::Noise> = Box::new(crate::pipeline::Torch::new(a.seed));

    println!("device: cpu (hand-written Rust backend, rayon)");
    println!("fragments: {}", fragments.len());
    let t0 = std::time::Instant::now();
    let w = crate::weights::Weights::load(&model_dir.join("model.safetensors"))?;
    let mut pl = CpuPipeline::load(&w)?;
    println!("  model loaded in {:?}", t0.elapsed());

    // ---- voice conditioning (same rules as synth.rs) ----------------------
    let mut spk_rows: Vec<Vec<f32>> = Vec::new();
    let mut prompt_latent: Option<(Vec<f32>, usize)> = None;
    let mut prompt_text = req.prompt_text.clone();
    if a.use_zero_spk_emb {
        spk_rows.push(vec![0.0f32; HIDDEN]);
    }
    if a.use_spk_emb && !a.use_zero_spk_emb && !a.spk_emb_path.is_empty() {
        for path in &a.spk_emb_path {
            let v = crate::pt::load_tensor(std::path::Path::new(path))?;
            if v.len() != 192 {
                bail!("{path} holds a {}-dim embedding, expected 192", v.len());
            }
            spk_rows.push(pl.project_speaker(&v));
        }
    }
    if !a.prompt_wav_path.is_empty() {
        let mut at_44k: Vec<f32> = Vec::new();
        let mut for_spk: Vec<f32> = Vec::new();
        for path in &a.prompt_wav_path {
            let (x, r) = crate::audio::read_wav(path)?;
            if a.use_spk_emb {
                for_spk.extend_from_slice(&if r != 16000 {
                    crate::audio::resample(&x, r, 16000)
                } else {
                    x.clone()
                });
            }
            at_44k.extend_from_slice(&if r != 44100 {
                crate::audio::resample(&x, r, 44100)
            } else {
                x
            });
        }
        if a.use_spk_emb && spk_rows.is_empty() {
            let onnx = model_dir.join("campplus.onnx");
            let emb = crate::spk::embed_16k_with(&onnx, &for_spk)?;
            spk_rows.push(pl.project_speaker(&emb));
        }
        if req.prompt_text.is_some() {
            prompt_latent = Some(pl.encode_prompt_latent(&mut *rng, &at_44k)?);
        }
    }
    if prompt_latent.is_none() {
        prompt_text = None;
    }
    let mut latent_rows = prompt_latent
        .as_ref()
        .map(|(lat, n)| pl.project_latent_rows(lat, *n));
    let mut latent_steps = prompt_latent.as_ref().map_or(0, |(_, n)| *n);

    let mut out: Vec<f32> = Vec::new();
    for (index, fragment) in fragments.iter().enumerate() {
        let layout = crate::tok::build_input_ids(
            &tok,
            crate::tok::DENSE,
            &crate::tok::InputSeqArgs {
                prompt: &req.prompt,
                text: fragment,
                num_speakers: spk_rows.len(),
                instruction: req.instruction.as_deref(),
                prompt_text: if latent_rows.is_some() { prompt_text.as_deref() } else { None },
                prompt_latent_steps: latent_steps,
            },
        )?;
        let mut embeds = pl.embed_ids(&layout.input_ids)?;
        for (slot, se) in spk_rows.iter().enumerate() {
            let pos = layout.speaker_positions[slot];
            embeds[pos * HIDDEN..(pos + 1) * HIDDEN].copy_from_slice(&se[..]);
        }
        if let Some(rows) = &latent_rows {
            let start = layout.latent_position as usize;
            embeds[start * HIDDEN..(start + latent_steps) * HIDDEN].copy_from_slice(rows);
        }
        if let Ok(path) = std::env::var("MING_DUMP_EMBEDS") {
            let path = format!("{path}.{index}.f32");
            let mut bytes = Vec::with_capacity(embeds.len() * 4);
            for v in &embeds {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&path, &bytes)?;
            println!("embeds -> {path} ({} x 896)", layout.input_ids.len());
        }
        let t = layout.input_ids.len();
        let t1 = std::time::Instant::now();
        let latents = pl.decode_loop(
            &embeds,
            t,
            &mut *rng,
            a.cfg as f32,
            a.sigma as f32,
            a.temperature as f32,
            a.max_decode_steps,
        )?;
        let n = latents.len() / ZC;
        let d = t1.elapsed();
        let t2 = std::time::Instant::now();
        let mut wav = pl.vocode(&latents);
        let v = t2.elapsed();
        if !a.no_trim_silence {
            wav = crate::pipeline::Pipeline::trim_trailing_silence(&wav, 44100);
        }
        if reference_first_fragment && index == 0 {
            // Same as ming_tts._prepare_generated_prompt: the trimmed chunk
            // becomes the acoustic prompt (and the voice) for every later one.
            let spk16 = crate::audio::resample(&wav, 44100, 16000);
            let onnx = model_dir.join("campplus.onnx");
            let se = crate::spk::embed_16k_with(&onnx, &spk16)?;
            spk_rows = vec![pl.project_speaker(&se)];
            let (lat, np) = pl.encode_prompt_latent(&mut *rng, &wav)?;
            latent_rows = Some(pl.project_latent_rows(&lat, np));
            latent_steps = np;
            prompt_text = Some(fragment.clone());
            println!(
                "  re-conditioning on fragment 0: {np} prompt patches, {} speaker samples",
                spk16.len()
            );
        }
        println!(
            "  fragment {index}: {n} latents in {:.2}s, vae {:.2}s",
            d.as_secs_f64(),
            v.as_secs_f64()
        );
        out.extend_from_slice(&wav);
    }
    crate::audio::write_wav(&a.output, &out, 44100)?;
    println!(
        "wrote {} ({:.2}s @ 44100)",
        a.output,
        out.len() as f64 / 44100.0
    );
    Ok(())
}


// ---------------------------------------------------------------------------
// VAE encoder (hand-written twin of vae.rs): waveform -> posterior params
// ---------------------------------------------------------------------------
