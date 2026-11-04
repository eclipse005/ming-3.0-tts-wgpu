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
