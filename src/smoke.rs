//! Smoke tests for a finished port: just enough to prove the front end and the
//! audio I/O still work, plus one end-to-end synthesis when a model is around.
//!
//! The numeric verification lives outside this repository: it compares against
//! goldens dumped from the Python reference (see README), which is not something
//! a source checkout can carry.

use crate::audio;

/// Resampling + WAV writing + reading back must round-trip the samples.
#[test]
fn wav_roundtrip() {
    let rate = 44100u32;
    let samples: Vec<f32> = (0..rate as usize)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();
    let path = std::env::temp_dir().join("ming_tts_smoke_roundtrip.wav");
    let path = path.to_string_lossy().to_string();

    audio::write_wav(&path, &samples, rate).expect("write wav");
    let (got, got_rate) = audio::read_wav(&path).expect("read wav");
    assert_eq!(got_rate, rate);
    assert_eq!(got.len(), samples.len());
    for (a, b) in samples.iter().zip(&got) {
        assert!((a - b).abs() < 1e-6, "sample differs: {a} vs {b}");
    }
}

/// The text front end the CLI uses: split, then run the reference's
/// normalisation chain over each fragment.
#[test]
fn text_front_end() {
    let fragments = crate::text::cut_text_by_semantic_length(
        &crate::text::add_period(&crate::text::preprosess_text(
            "今天天气真不错，我们一起去公园散步吧。",
        )),
        24,
    );
    assert!(!fragments.is_empty());

    let tn = crate::tn::TalkerTn::load(&crate::tn::default_dir()).expect("load the FST data");
    let normalized = tn
        .normalize_fragment("价格是 350 元，2026 年 11 月 1 日，增长 12.5%。")
        .expect("normalise");
    let kit = crate::text::normalize_text_kit(&normalized);
    // Digits, money and dates must come out as spoken Chinese, not symbols.
    assert!(!kit.contains('3'), "digit left in {kit:?}");
    assert!(!kit.contains('%'), "percent sign left in {kit:?}");
}

/// One short utterance through the whole pipeline.
///
/// Needs the 0.5b checkpoint, which is not shipped: point `MING_MODEL` at the
/// downloaded directory (default `../Ming-3.0-tts-0.5b`).
#[test]
fn synthesize_short_utterance() {
    let model = std::env::var("MING_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../Ming-3.0-tts-0.5b")
        });
    if !model.join("model.safetensors").is_file() {
        eprintln!("[skip] no model at {} (set MING_MODEL)", model.display());
        return;
    }

    let out = std::env::temp_dir().join("ming_tts_smoke.wav");
    let args = crate::synth::Args {
        model_path: model.to_string_lossy().to_string(),
        task: "TTS".into(),
        text: "今天天气真不错。".into(),
        output: out.to_string_lossy().to_string(),
        device: std::env::var("MING_DEVICE").unwrap_or_else(|_| "cuda:0".into()),
        seed: 1895,
        language_id: None,
        instruction: None,
        instruction_schema: None,
        speed: None,
        pitch: None,
        volume: None,
        emotion: None,
        dialect: None,
        style: None,
        ip: None,
        cfg: 2.0,
        sigma: 0.25,
        temperature: 0.0,
        max_length: 24,
        max_decode_steps: 1000,
        prompt_wav_path: vec![],
        prompt_text: None,
        spk_emb_path: vec![],
        use_spk_emb: false,
        use_zero_spk_emb: false,
        no_text_normalization: false,
        raw_text: false,
        no_trim_silence: false,
    };
    crate::synth::run(args).expect("synthesize");
    let (samples, rate) = audio::read_wav(&out.to_string_lossy()).expect("read the output wav");
    assert_eq!(rate, 44100);
    assert!(samples.len() > 44100 / 2, "too short: {} samples", samples.len());
}

/// The acoustic-prompt layout must stay the one the reference builds.
///
/// Numbers measured from `infer.py` @ transformers 4.46.3 with
/// `--no_text_normalization`, text "这是测试。" and the shipped preset
/// (`data/voice_presets/voice_audio/lingguang.wav` + its transcript, 42 prompt
/// patches): 35 token ids without the prompt, 92 with it, and the 42
/// `<audioPatch>` rows last. A regression here silently re-conditions the model
/// on a different acoustic prompt, which is what the reported "prompt path
/// swallows words" looked like (the duration then changes, the ids do not
/// necessarily fail to tokenize).
#[test]
fn prompt_layout_matches_reference() {
    let model = std::env::var("MING_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../Ming-3.0-tts-0.5b")
        });
    if !model.join("tokenizer.json").is_file() {
        eprintln!("[skip] no tokenizer at {} (set MING_MODEL)", model.display());
        return;
    }
    let tok = crate::tok::Tok::load(&model).expect("load tokenizer");
    let text = "这是测试。";
    let prompt_text = "诶，你今天有空吗？我们能不能聊聊这件事啊？";

    let plain = crate::tok::build_input_ids(
        &tok,
        crate::tok::DENSE,
        &crate::tok::InputSeqArgs {
            prompt: "",
            text,
            num_speakers: 0,
            instruction: None,
            prompt_text: None,
            prompt_latent_steps: 0,
        },
    )
    .expect("layout without a prompt");
    assert_eq!(plain.input_ids.len(), 35, "reference builds 35 ids");
    assert_eq!(plain.latent_position, -1, "no latent rows without a prompt");

    let prompted = crate::tok::build_input_ids(
        &tok,
        crate::tok::DENSE,
        &crate::tok::InputSeqArgs {
            prompt: "",
            text,
            num_speakers: 0,
            instruction: None,
            prompt_text: Some(prompt_text),
            prompt_latent_steps: 42,
        },
    )
    .expect("layout with a prompt");
    assert_eq!(prompted.input_ids.len(), 92, "reference builds 92 ids");
    assert_eq!(prompted.latent_position, 50, "latent rows start at 50");
    let patch = tok.token_id(crate::tok::AUDIO_PATCH).expect("audioPatch id");
    assert!(
        prompted.input_ids[50..].iter().all(|&id| id == patch),
        "the 42 latent rows must be audioPatch ids"
    );
}
