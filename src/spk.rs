//! CAMPlus speaker embedding (campplus.onnx). Not part of the wgpu graph.
//!
//! spkemb_extractor.py computes the embedding as
//!
//! ```text
//! feat = kaldi.fbank(wav16, num_mel_bins=80, dither=0, sample_frequency=16000)
//! feat = feat - feat.mean(dim=0, keepdim=True)
//! emb  = campplus.run([feat[None]])   # (1, 192)
//! ```
//!
//! so this module contains both halves: a bit-faithful port of
//! torchaudio.compliance.kaldi.fbank (16 kHz, 80 mel bins, povey window,
//! DC removal, 0.97 pre-emphasis, 512-point FFT, log mel energies) and the
//! ONNX Runtime call. The intermediate stages are validated against goldens
//! dumped from Python, so a break points at the stage that moved instead of
//! only at the 192-dim output.
//!
//! The ONNX Runtime shared library is not vendored: `ort` is built with
//! `load-dynamic` and the DLL is located at run time (see
//! `resolve_onnxruntime_dylib`). Set `ORT_DYLIB_PATH` to override.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub const SPEAKER_EMBEDDING_DIM: usize = 192;

// --- kaldi fbank parameters (torchaudio.compliance.kaldi defaults) --------
const SAMPLE_FREQUENCY: f64 = 16000.0;
const FRAME_LENGTH_MS: f64 = 25.0;
const FRAME_SHIFT_MS: f64 = 10.0;
const NUM_MEL_BINS: usize = 80;
const LOW_FREQ: f64 = 20.0;
const HIGH_FREQ: f64 = 0.0; // <= 0 means "nyquist + this"
const PREEMPHASIS: f32 = 0.97;
const MILLISECONDS_TO_SECONDS: f64 = 0.001;

#[inline]
fn window_shift() -> usize {
    (SAMPLE_FREQUENCY * FRAME_SHIFT_MS * MILLISECONDS_TO_SECONDS) as usize
}

#[inline]
fn window_size() -> usize {
    (SAMPLE_FREQUENCY * FRAME_LENGTH_MS * MILLISECONDS_TO_SECONDS) as usize
}

#[inline]
fn padded_window_size() -> usize {
    next_power_of_two(window_size())
}

fn next_power_of_two(x: usize) -> usize {
    if x == 0 {
        1
    } else {
        1usize << (usize::BITS - (x - 1).leading_zeros())
    }
}

fn mel_scale_scalar(freq: f64) -> f64 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// torchaudio.compliance.kaldi._feature_window_function for `window_type="povey"`:
/// a symmetric hann window raised to 0.85.
pub fn povey_window(size: usize) -> Vec<f32> {
    // torch builds the window in the tensor's own dtype: step is a float32
    // 2*pi/(N-1) and the cosine is cosf, so this is f32 arithmetic.
    let step = 2.0 * std::f32::consts::PI / (size as f32 - 1.0);
    (0..size)
        .map(|n| {
            let hann = 0.5f32 - 0.5f32 * (step * n as f32).cos();
            hann.powf(0.85)
        })
        .collect()
}

/// kaldi.get_mel_banks(...) for the checkpoint's settings, with fbank's zero
/// column appended, i.e. a `NUM_MEL_BINS x (padded/2 + 1)` row-major matrix.
pub fn mel_filterbank() -> Vec<f32> {
    let padded = padded_window_size();
    let num_fft_bins = padded / 2;
    let nyquist = 0.5 * SAMPLE_FREQUENCY;
    let high_freq = if HIGH_FREQ <= 0.0 {
        HIGH_FREQ + nyquist
    } else {
        HIGH_FREQ
    };
    let fft_bin_width = SAMPLE_FREQUENCY / padded as f64;
    let mel_low = mel_scale_scalar(LOW_FREQ);
    let mel_high = mel_scale_scalar(high_freq);
    let mel_freq_delta = (mel_high - mel_low) / (NUM_MEL_BINS as f64 + 1.0);

    // get_mel_banks builds its arrays from an integer arange scaled by python
    // floats, which torch promotes to float32.
    let mel_low32 = mel_low as f32;
    let delta32 = mel_freq_delta as f32;
    let freq32 = fft_bin_width as f32;
    let mut bins = vec![0.0f32; NUM_MEL_BINS * (num_fft_bins + 1)];
    for bin in 0..NUM_MEL_BINS {
        let b = bin as f32;
        let left = mel_low32 + b * delta32;
        let center = mel_low32 + (b + 1.0) * delta32;
        let right = mel_low32 + (b + 2.0) * delta32;
        for i in 0..num_fft_bins {
            let mel = 1127.0f32 * (1.0 + (freq32 * i as f32) / 700.0).ln();
            let up = (mel - left) / (center - left);
            let down = (right - mel) / (right - center);
            // max(0, min(up_slope, down_slope))
            let value = up.min(down).max(0.0);
            bins[bin * (num_fft_bins + 1) + i] = value;
        }
        // The last (nyquist) column stays zero.
    }
    bins
}

/// The strided, DC-removed, pre-emphasised, windowed frames, zero padded to
/// `padded_window_size()`: exactly the tensor `torch.fft.rfft` receives.
pub fn fft_input(wav: &[f32]) -> (Vec<f32>, usize) {
    let size = window_size();
    let shift = window_shift();
    let padded = padded_window_size();
    let window = povey_window(size);
    let frames = num_frames(wav.len());
    let mut out = vec![0.0f32; frames * padded];
    let mut row = vec![0.0f32; size];
    for frame in 0..frames {
        let start = frame * shift;
        let slice = &wav[start..start + size];
        // remove_dc_offset: subtract the frame mean.
        let mut sum = 0.0f32;
        for &value in slice {
            sum += value;
        }
        let mean = sum / size as f32;
        for (dst, &value) in row.iter_mut().zip(slice) {
            *dst = value - mean;
        }
        // preemphasis_coefficient: x[j] -= 0.97 * x[max(0, j-1)], where the
        // padding replicates the first sample (f.pad(..., "replicate")).
        let mut previous = row[0];
        row[0] -= PREEMPHASIS * previous;
        for j in 1..size {
            let current = row[j];
            row[j] = current - PREEMPHASIS * previous;
            previous = current;
        }
        let dst = &mut out[frame * padded..frame * padded + size];
        for (d, (r, w)) in dst.iter_mut().zip(row.iter().zip(&window)) {
            *d = r * w;
        }
    }
    (out, frames)
}

/// `1 + (num_samples - window_size) / window_shift`, zero when too short.
pub fn num_frames(num_samples: usize) -> usize {
    let size = window_size();
    if num_samples < size {
        0
    } else {
        1 + (num_samples - size) / window_shift()
    }
}

/// `torch.fft.rfft(frames).abs().pow(2)`, one row of
/// `padded_window_size() / 2 + 1` bins per frame.
pub fn power_spectrum(frames: &[f32], n_frames: usize) -> Vec<f32> {
    let padded = padded_window_size();
    let bins = padded / 2 + 1;
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(padded);
    let mut buffer: Vec<rustfft::num_complex::Complex<f32>> =
        vec![rustfft::num_complex::Complex::new(0.0, 0.0); padded];
    let mut out = vec![0.0f32; n_frames * bins];
    for frame in 0..n_frames {
        for (slot, &value) in buffer.iter_mut().zip(&frames[frame * padded..]) {
            *slot = rustfft::num_complex::Complex::new(value, 0.0);
        }
        fft.process(&mut buffer);
        for bin in 0..bins {
            let z = buffer[bin];
            // torch: complex.abs() is std::abs (hypot), then .pow(2.0).
            let magnitude = z.re.hypot(z.im);
            out[frame * bins + bin] = magnitude * magnitude;
        }
    }
    out
}

/// `log(max(spectrum @ mel.T, epsilon))$: the raw fbank features before the
/// per-bin mean is removed.
pub fn log_mel_energies(spectrum: &[f32], n_frames: usize) -> Vec<f32> {
    let bins = padded_window_size() / 2 + 1;
    let mel = mel_filterbank();
    let mut out = vec![0.0f32; n_frames * NUM_MEL_BINS];
    for frame in 0..n_frames {
        let row = &spectrum[frame * bins..frame * bins + bins];
        for bin in 0..NUM_MEL_BINS {
            let weights = &mel[bin * bins..bin * bins + bins];
            let mut acc = 0.0f32;
            for (&s, &w) in row.iter().zip(weights) {
                acc += s * w;
            }
            out[frame * NUM_MEL_BINS + bin] = acc.max(f32::EPSILON).ln();
        }
    }
    out
}

/// Subtract the per-bin mean over frames (the reference's
/// `feat - feat.mean(dim=0, keepdim=True)`).
pub fn subtract_mean(features: &mut [f32], n_frames: usize) {
    if n_frames == 0 {
        return;
    }
    let count = n_frames as f32;
    for bin in 0..NUM_MEL_BINS {
        let mut sum = 0.0f32;
        for frame in 0..n_frames {
            sum += features[frame * NUM_MEL_BINS + bin];
        }
        let mean = sum / count;
        for frame in 0..n_frames {
            features[frame * NUM_MEL_BINS + bin] -= mean;
        }
    }
}

/// The complete CAMPlus feature front end: fbank minus the per-bin mean,
/// shape `(frames, 80)`.
pub fn campplus_features(wav_16k: &[f32]) -> Vec<f32> {
    let (frames, n_frames) = fft_input(wav_16k);
    let spectrum = power_spectrum(&frames, n_frames);
    let mut features = log_mel_energies(&spectrum, n_frames);
    subtract_mean(&mut features, n_frames);
    features
}

// --- ONNX Runtime -------------------------------------------------------
//
// `ort` is built with `load-dynamic`, so nothing is downloaded or linked at
// build time and the runtime is loaded from a shared library at run time.

static ENVIRONMENT: OnceLock<Result<PathBuf, String>> = OnceLock::new();
#[allow(clippy::type_complexity)]
static SESSIONS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<ort::session::Session>>>>> =
    OnceLock::new();

/// Candidate locations of `onnxruntime.dll`, most explicit first.
pub fn resolve_onnxruntime_dylib() -> Result<PathBuf> {
    let mut tried: Vec<String> = Vec::new();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        candidates.push(PathBuf::from(path));
    }
    if let Ok(dir) = std::env::var("MING_ONNXRUNTIME_DIR") {
        candidates.push(PathBuf::from(dir).join("onnxruntime.dll"));
    }
    // A DLL sitting next to the binary, then the usual conda/pip layouts on
    // this development box.
    candidates.push(PathBuf::from("onnxruntime.dll"));
    for python in [
        "C:/Users/ADMIN/miniconda3/envs/videolingo/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        "C:/Users/ADMIN/miniconda3/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    ] {
        candidates.push(PathBuf::from(python));
    }
    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
        tried.push(candidate.display().to_string());
    }
    bail!(
        "onnxruntime shared library not found (set ORT_DYLIB_PATH or \
         MING_ONNXRUNTIME_DIR); tried: {}",
        tried.join(", ")
    )
}

fn environment() -> Result<&'static PathBuf> {
    let result = ENVIRONMENT.get_or_init(|| {
        let dylib = resolve_onnxruntime_dylib().map_err(|e| format!("{e}"))?;
        let committed = ort::init_from(&dylib)
            .map_err(|e| format!("load onnxruntime from {}: {e:?}", dylib.display()))?
            .commit();
        if !committed {
            return Err("ONNX Runtime environment was already initialised".to_string());
        }
        Ok(dylib)
    });
    match result {
        Ok(path) => Ok(path),
        Err(message) => bail!("{message}"),
    }
}

/// Default `campplus.onnx`: `MING_CAMPPLUS_ONNX`, else the sibling checkpoint.
pub fn resolve_campplus() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("MING_CAMPPLUS_ONNX") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        bail!("MING_CAMPPLUS_ONNX={} does not exist", path.display());
    }
    let candidate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../Ming-3.0-tts-0.5b/campplus.onnx");
    if candidate.is_file() {
        return Ok(candidate);
    }
    bail!(
        "campplus.onnx not found at {} (set MING_CAMPPLUS_ONNX)",
        candidate.display()
    )
}

fn session_for(model: &Path) -> Result<Arc<Mutex<ort::session::Session>>> {
    environment()?;
    let cache = SESSIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("session cache poisoned");
    if let Some(existing) = guard.get(model) {
        return Ok(existing.clone());
    }
    let session = ort::session::Session::builder()
        .with_context(|| "create ONNX Runtime session builder")?
        .commit_from_file(model)
        .with_context(|| format!("load ONNX model {}", model.display()))?;
    let session = Arc::new(Mutex::new(session));
    guard.insert(model.to_path_buf(), session.clone());
    Ok(session)
}

/// Run CAMPlus on pre-computed `(frames, 80)` features, returning 192 floats.
pub fn embed_features(model: &Path, features: &[f32], frames: usize) -> Result<Vec<f32>> {
    if frames == 0 {
        bail!("campplus: no frames (audio shorter than 25 ms)");
    }
    if features.len() != frames * NUM_MEL_BINS {
        bail!(
            "campplus: expected {} features, got {}",
            frames * NUM_MEL_BINS,
            features.len()
        );
    }
    let session = session_for(model)?;
    let tensor = ort::value::Tensor::from_array((
        [1i64, frames as i64, NUM_MEL_BINS as i64],
        features.to_vec(),
    ))
    .with_context(|| "build campplus input tensor")?;
    let mut session = session.lock().expect("session mutex poisoned");
    let output = session
        .run(ort::inputs!["input" => tensor])
        .with_context(|| "run campplus.onnx")?;
    let (shape, values) = output[0]
        .try_extract_tensor::<f32>()
        .with_context(|| "extract campplus output")?;
    let values = values.to_vec();
    if values.len() != SPEAKER_EMBEDDING_DIM {
        bail!(
            "campplus: expected {SPEAKER_EMBEDDING_DIM}-dim embedding, got {:?}",
            shape
        );
    }
    Ok(values)
}

/// 192-dim speaker embedding from 16 kHz mono audio, using the checkpoint's
/// `campplus.onnx`.
pub fn embed_16k(wav_16k: &[f32]) -> Result<Vec<f32>> {
    let model = resolve_campplus()?;
    embed_16k_with(&model, wav_16k)
}

/// Convenience wrapper mirroring ming_tts._load_prompt: resample to 16 kHz
/// first when needed, then embed.
pub fn embed(wav: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
    if sample_rate == 16000 {
        embed_16k(wav)
    } else {
        embed_16k(&crate::audio::resample(wav, sample_rate, 16000))
    }
}

/// Same, with an explicit model path (tests, or a checkpoint elsewhere).
pub fn embed_16k_with(model: &Path, wav_16k: &[f32]) -> Result<Vec<f32>> {
    if wav_16k.is_empty() {
        bail!("campplus: empty waveform");
    }
    let features = campplus_features(wav_16k);
    let frames = num_frames(wav_16k.len());
    embed_features(model, &features, frames)
}

// ---------------------------------------------------------------- tests
#[cfg(test)]
mod tests {
    use super::*;

    fn goldens_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens").join("m5_audio")
    }

    fn repo_file(rel: &str) -> String {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(rel)
            .to_string_lossy()
            .to_string()
    }

    fn read_f32(name: &str) -> Vec<f32> {
        let path = goldens_dir().join(name);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len(), "length {} vs {}", got.len(), want.len());
        got.iter()
            .zip(want)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
    }

    fn mean_abs_diff(got: &[f32], want: &[f32]) -> f64 {
        let mut sum = 0.0f64;
        for (a, b) in got.iter().zip(want) {
            sum += ((a - b) as f64).abs();
        }
        sum / got.len() as f64
    }

    /// The 16 kHz prompt the Python reference feeds to campplus.
    fn lingguang_16k() -> Vec<f32> {
        let wav = read_f32("lingguang_mono.f32");
        crate::audio::resample(&wav, 48000, 16000)
    }

    #[test]
    fn povey_window_matches_torchaudio() {
        let want = read_f32("spk_window_povey.f32");
        let got = povey_window(400);
        let diff = max_abs_diff(&got, &want);
        println!(
            "povey window (400): max abs diff = {diff:e}, w[0]={:.8} w[200]={:.8} w[399]={:.8}",
            got[0], got[200], got[399]
        );
        assert!(diff < 1e-6, "max abs diff {diff:e}");
    }

    #[test]
    fn fft_input_matches_torchaudio() {
        let wav = lingguang_16k();
        assert_eq!(wav.len(), 52691, "16 kHz prompt length");
        let (got, frames) = fft_input(&wav);
        let want = read_f32("spk_fft_input.f32");
        assert_eq!(frames, 327, "frame count");
        assert_eq!(got.len(), want.len(), "fft input length");
        let diff = max_abs_diff(&got, &want);
        println!(
            "fft input ({frames} x 512): max abs diff = {diff:e}, mean abs diff = {:.3e}",
            mean_abs_diff(&got, &want)
        );
        assert!(diff < 1e-6, "max abs diff {diff:e}");
    }

    #[test]
    fn power_spectrum_matches_torchaudio() {
        let wav = lingguang_16k();
        let (frames, n_frames) = fft_input(&wav);
        let got = power_spectrum(&frames, n_frames);
        let want = read_f32("spk_power_spectrum.f32");
        assert_eq!(got.len(), want.len(), "power spectrum length");
        let diff = max_abs_diff(&got, &want);
        let peak = want.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let relative = diff / peak;
        println!(
            "power spectrum ({n_frames} x 257): max abs diff = {diff:e}, max rel = {relative:e}, peak = {peak:e}"
        );
        assert!(relative < 1e-5, "max rel diff {relative:e}");
    }

    #[test]
    fn mel_filterbank_matches_torchaudio() {
        let got = mel_filterbank();
        let want = read_f32("spk_mel_padded.f32");
        assert_eq!(got.len(), want.len(), "mel matrix length");
        let diff = max_abs_diff(&got, &want);
        println!(
            "mel filterbank (80 x 257): max abs diff = {diff:e}, row sums first = {:.6} last = {:.6}",
            got[..257].iter().sum::<f32>(),
            got[79 * 257..].iter().sum::<f32>()
        );
        // torch's own float32 evaluation order differs from any closed form at
        // the 1e-5 level here (cancellation in center-left), so 1e-4 is the
        // tightest meaningful bound.
        assert!(diff < 1e-4, "max abs diff {diff:e}");
    }

    #[test]
    fn log_mel_energies_match_torchaudio() {
        let wav = lingguang_16k();
        let (frames, n_frames) = fft_input(&wav);
        let spectrum = power_spectrum(&frames, n_frames);
        let got = log_mel_energies(&spectrum, n_frames);
        let want = read_f32("spk_fbank_raw.f32");
        assert_eq!(got.len(), want.len());
        let diff = max_abs_diff(&got, &want);
        println!(
            "log mel energies ({n_frames} x 80): max abs diff = {diff:e}, mean abs diff = {:.3e}",
            mean_abs_diff(&got, &want)
        );
        assert!(diff < 1e-3, "max abs diff {diff:e}");
    }

    #[test]
    fn campplus_features_match_torchaudio() {
        let wav = lingguang_16k();
        let got = campplus_features(&wav);
        let want = read_f32("spk_lingguang_fbank.f32");
        assert_eq!(got.len(), want.len());
        let diff = max_abs_diff(&got, &want);
        println!(
            "campplus features ({:} x 80, mean removed): max abs diff = {diff:e}, mean abs diff = {:.3e}",
            got.len() / 80,
            mean_abs_diff(&got, &want)
        );
        assert!(diff < 1e-3, "max abs diff {diff:e}");
    }

    #[test]
    fn speaker_embedding_matches_python() {
        let path = goldens_dir().join("spk_lingguang.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let golden: serde_json::Value = serde_json::from_str(&text).expect("parse golden json");
        let want: Vec<f32> = golden["embedding"]
            .as_array()
            .expect("embedding array")
            .iter()
            .map(|v| v.as_f64().expect("float") as f32)
            .collect();
        assert_eq!(want.len(), SPEAKER_EMBEDDING_DIM);

        // The full reference path: 48 kHz preset -> 16 kHz -> fbank -> CAMPlus.
        let wav48 = read_f32("lingguang_mono.f32");
        let got = embed(&wav48, 48000).expect("campplus embedding");
        assert_eq!(got.len(), SPEAKER_EMBEDDING_DIM);
        let diff = max_abs_diff(&got, &want);
        let peak = want.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        println!(
            "campplus embedding (192): max abs diff = {diff:e} (peak {peak:.6}), mean abs diff = {:.3e}",
            mean_abs_diff(&got, &want)
        );
        println!(
            "  first 4 rust = {:?}",
            &got[..4]
        );
        println!(
            "  first 4 python = {:?}",
            &want[..4]
        );
        assert!(
            diff < 1e-3,
            "max abs diff {diff:e} >= 1e-3: the ONNX path or the fbank front end is off"
        );
    }
}
