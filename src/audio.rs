//! Audio I/O: WAV read/write and torchaudio-compatible resampling.
//!
//! Two reference behaviours are reproduced bit-for-bit here:
//!
//! * `read_wav` is `torchaudio.load(path, backend="soundfile")` followed by
//!   `waveform.mean(dim=0)` (ming_tts._load_prompt), i.e. soundfile's
//!   integer -> float scaling and a per-frame channel mean.
//! * `write_wav` is `torchaudio.save` with the soundfile backend, which means
//!   libsndfile's 32-bit float WAV layout: a `fmt ` chunk with format tag 3
//!   (IEEE float), a `fact` chunk holding the frame count, a `PEAK` chunk
//!   (version, wall-clock timestamp, then peak/position per channel) and the
//!   `data` chunk. Matching that header is what makes the output files
//!   byte-comparable with the Python pipeline; only the PEAK timestamp, which
//!   is `time(NULL)`, can differ.
//!
//! `resample` ports torchaudio.transforms.Resample with its defaults
//! (lowpass_filter_width=6, rolloff=0.99, sinc_interp_hann). The kernel is
//! built in f64 and cast to f32, exactly like the transform does when it caches
//! a `dtype=None` kernel, then applied as the same stride-`orig_freq`
//! convolution.

use anyhow::{bail, Context, Result};
use std::io::Write;

/// `torchaudio.load(...).mean(dim=0)`: mono f32 samples and the sample rate.
pub fn read_wav(path: &str) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open wav {path}"))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        bail!("{path}: zero channels");
    }
    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<f32>, _>>()
            .with_context(|| format!("read f32 samples from {path}"))?,
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<std::result::Result<Vec<f32>, _>>()
            .with_context(|| format!("read i16 samples from {path}"))?,
        (hound::SampleFormat::Int, bits @ (24 | 32)) => {
            let scale = if bits == 24 { 8388608.0f32 } else { 2147483648.0 };
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<std::result::Result<Vec<f32>, _>>()
                .with_context(|| format!("read i{bits} samples from {path}"))?
        }
        // 8-bit WAV is unsigned PCM, which hound does not expose; the
        // reference pipeline never reads it (all presets are 16-bit and the
        // model writes 32-bit float).
        (format, bits) => bail!("{path}: unsupported wav format {format:?} {bits} bit"),
    };
    if interleaved.len() % channels != 0 {
        bail!(
            "{path}: {} samples is not a multiple of {channels} channels",
            interleaved.len()
        );
    }
    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    if channels == 1 {
        mono.extend_from_slice(&interleaved);
    } else {
        // waveform.mean(dim=0): sum the channels in order, then divide by the
        // channel count, all in f32.
        let count = channels as f32;
        for frame in interleaved.chunks_exact(channels) {
            let mut sum = 0.0f32;
            for &value in frame {
                sum += value;
            }
            mono.push(sum / count);
        }
    }
    Ok((mono, spec.sample_rate))
}

/// `torchaudio.save` for a mono waveform (soundfile / libsndfile, float32).
pub fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    write_wav_channels(path, std::slice::from_ref(&samples), sample_rate)
}

/// `torchaudio.save` for an interleaved-agnostic multichannel waveform.
///
/// `channels[i]` is channel `i`; every channel must have the same length.
pub fn write_wav_channels(path: &str, channels: &[&[f32]], sample_rate: u32) -> Result<()> {
    if channels.is_empty() {
        bail!("write_wav: no channels");
    }
    let frames = channels[0].len();
    for (index, channel) in channels.iter().enumerate() {
        if channel.len() != frames {
            bail!(
                "write_wav: channel {index} has {} samples, expected {frames}",
                channel.len()
            );
        }
    }
    let n_channels = channels.len();
    let block_align = (n_channels * 4) as u16;
    let data_len = (frames * n_channels * 4) as u32;
    let peak_len = 8 + 8 * n_channels;
    // RIFF(12) + fmt(8+16) + fact(8+4) + PEAK(8+peak_len) + data(8+data_len)
    let riff_size = 12 + 24 + 12 + 8 + peak_len + 8 + data_len as usize - 8;

    let mut out: Vec<u8> = Vec::with_capacity(riff_size + 8);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // WAVE_FORMAT_IEEE_FLOAT
    out.extend_from_slice(&(n_channels as u16).to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * block_align as u32).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"fact");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&(frames as u32).to_le_bytes());
    out.extend_from_slice(b"PEAK");
    out.extend_from_slice(&(peak_len as u32).to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes()); // PEAK version
    out.extend_from_slice(&unix_now().to_le_bytes());
    for channel in channels {
        // libsndfile keeps the first index whose |value| strictly exceeds the
        // running peak, so ties resolve to the earliest sample.
        let mut peak = 0.0f32;
        let mut position = 0u32;
        for (index, &value) in channel.iter().enumerate() {
            let magnitude = value.abs();
            if magnitude > peak {
                peak = magnitude;
                position = index as u32;
            }
        }
        out.extend_from_slice(&peak.to_le_bytes());
        out.extend_from_slice(&position.to_le_bytes());
    }
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for frame in 0..frames {
        for channel in channels {
            out.extend_from_slice(&channel[frame].to_le_bytes());
        }
    }

    let mut file =
        std::fs::File::create(path).with_context(|| format!("create wav {path}"))?;
    file.write_all(&out)
        .with_context(|| format!("write wav {path}"))?;
    Ok(())
}

fn unix_now() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

/// `torchaudio.transforms.Resample(from_rate, to_rate)` with default options.
pub fn resample(x: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == 0 || to_rate == 0 || x.is_empty() {
        return Vec::new();
    }
    if from_rate == to_rate {
        return x.to_vec();
    }
    let (kernel, width, orig, new) = sinc_kernel(from_rate, to_rate);
    apply_kernel(x, &kernel, width, orig, new)
}

/// torchaudio.functional._get_sinc_resample_kernel, f64 kernel cast to f32.
///
/// Returns `(kernel, width, orig_freq, new_freq)` where the kernel is laid out
/// as `new_freq` phases of `2 * width + orig_freq` taps.
fn sinc_kernel(from_rate: u32, to_rate: u32) -> (Vec<f32>, usize, usize, usize) {
    let gcd = gcd_u32(from_rate, to_rate) as usize;
    let orig = from_rate as usize / gcd;
    let new = to_rate as usize / gcd;
    let lowpass_filter_width = 6usize;
    let rolloff = 0.99f64;

    let base_freq = (orig.min(new) as f64) * rolloff;
    // width = ceil(lowpass_filter_width * orig_freq / base_freq)
    let width = ((lowpass_filter_width as f64 * orig as f64) / base_freq).ceil() as usize;
    let taps = 2 * width + orig;

    let scale = base_freq / orig as f64;
    let mut kernel = vec![0.0f32; new * taps];
    for phase in 0..new {
        for tap in 0..taps {
            let idx = (tap as f64 - width as f64) / orig as f64;
            let mut t = (-(phase as f64) / new as f64 + idx) * base_freq;
            t = t.clamp(-(lowpass_filter_width as f64), lowpass_filter_width as f64);
            let window = (t * std::f64::consts::PI / lowpass_filter_width as f64 / 2.0)
                .cos()
                .powi(2);
            let t_pi = t * std::f64::consts::PI;
            let sinc = if t_pi == 0.0 { 1.0 } else { t_pi.sin() / t_pi };
            kernel[phase * taps + tap] = (sinc * window * scale) as f32;
        }
    }
    (kernel, width, orig, new)
}

/// torchaudio.functional._apply_sinc_resample_kernel.
fn apply_kernel(x: &[f32], kernel: &[f32], width: usize, orig: usize, new: usize) -> Vec<f32> {
    let taps = 2 * width + orig;
    let length = x.len();
    let blocks = length / orig + 1;
    let raw_len = blocks * new;
    // target_length = ceil(as_tensor(new_freq * length / orig_freq)) in f32,
    // then truncated to an integer, matching torch's float32 intermediate.
    let target_length = (((new * length) as f64 / orig as f64) as f32).ceil() as usize;
    let out_len = raw_len.min(target_length);

    // xpad[i] = x[i - width], zero outside the signal.
    let mut out = Vec::with_capacity(out_len);
    let mut frame = vec![0.0f32; taps];
    for block in 0..blocks {
        let base = block * orig;
        for (tap, slot) in frame.iter_mut().enumerate() {
            let pos = base as isize + tap as isize - width as isize;
            *slot = if pos >= 0 && (pos as usize) < length {
                x[pos as usize]
            } else {
                0.0
            };
        }
        for phase in 0..new {
            let offset = phase * taps;
            let mut acc = 0.0f32;
            for tap in 0..taps {
                acc += kernel[offset + tap] * frame[tap];
            }
            if out.len() == out_len {
                return out;
            }
            out.push(acc);
        }
    }
    out
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

// ---------------------------------------------------------------- tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

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
        assert_eq!(bytes.len() % 4, 0, "{} is not f32 payload", path.display());
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        let mut worst = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs();
            if d > worst {
                worst = d;
            }
        }
        worst
    }

    fn max_abs(a: &[f32]) -> f32 {
        a.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
    }

    #[test]
    fn read_wav_matches_torchaudio_load_lingguang() {
        let (samples, rate) = read_wav(&repo_file("data/voice_presets/voice_audio/lingguang.wav"))
            .expect("read lingguang.wav");
        let want = read_f32("lingguang_mono.f32");
        assert_eq!(rate, 48000, "sample rate");
        assert_eq!(samples.len(), want.len(), "sample count");
        let diff = max_abs_diff(&samples, &want);
        let exact = samples.iter().zip(&want).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        println!(
            "lingguang.wav: {} samples @ {rate} Hz, max abs diff vs torchaudio.load = {diff:e}, bit-identical {exact}/{}",
            samples.len(),
            want.len()
        );
        assert_eq!(diff, 0.0, "PCM16 -> f32 scaling must be exact");
        assert_eq!(exact, want.len());
    }

    #[test]
    fn read_wav_matches_torchaudio_load_db30() {
        let (samples, rate) = read_wav(&repo_file("data/voice_presets/voice_audio/db30.wav"))
            .expect("read db30.wav");
        let want = read_f32("db30_mono.f32");
        assert_eq!(rate, 16000);
        assert_eq!(samples.len(), want.len());
        let diff = max_abs_diff(&samples, &want);
        println!(
            "db30.wav: {} samples @ {rate} Hz, max abs diff = {diff:e}, peak {:.6}",
            samples.len(),
            max_abs(&samples)
        );
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn read_wav_averages_channels_like_mean_dim0() {
        let path = goldens_dir().join("stereo_input.wav");
        let (samples, rate) = read_wav(&path.to_string_lossy()).expect("read stereo_input.wav");
        let want = read_f32("stereo_mean.f32");
        assert_eq!(rate, 48000);
        assert_eq!(samples.len(), want.len());
        let diff = max_abs_diff(&samples, &want);
        println!(
            "stereo_input.wav: {} frames, 2 channels averaged, max abs diff vs mean(dim=0) = {diff:e}",
            samples.len()
        );
        assert_eq!(diff, 0.0, "channel mean must be exact");
    }

    #[test]
    fn resample_lingguang_to_16000() {
        let input = read_f32("lingguang_mono.f32");
        let want = read_f32("lingguang_48000_to_16000.f32");
        let got = resample(&input, 48000, 16000);
        assert_eq!(got.len(), want.len(), "output length");
        let diff = max_abs_diff(&got, &want);
        println!(
            "resample 48000 -> 16000: {} -> {} samples, max abs err = {diff:e} (peak {:.6})",
            input.len(),
            got.len(),
            max_abs(&want)
        );
        assert!(diff < 1e-4, "max abs err {diff:e} >= 1e-4");
    }

    #[test]
    fn resample_lingguang_to_44100() {
        let input = read_f32("lingguang_mono.f32");
        let want = read_f32("lingguang_48000_to_44100.f32");
        let got = resample(&input, 48000, 44100);
        assert_eq!(got.len(), want.len(), "output length");
        let diff = max_abs_diff(&got, &want);
        println!(
            "resample 48000 -> 44100: {} -> {} samples, max abs err = {diff:e} (peak {:.6})",
            input.len(),
            got.len(),
            max_abs(&want)
        );
        assert!(diff < 1e-4, "max abs err {diff:e} >= 1e-4");
    }

    #[test]
    fn resample_db30_to_44100() {
        let input = read_f32("db30_mono.f32");
        let want = read_f32("db30_16000_to_44100.f32");
        let got = resample(&input, 16000, 44100);
        assert_eq!(got.len(), want.len(), "output length");
        let diff = max_abs_diff(&got, &want);
        println!(
            "resample 16000 -> 44100: {} -> {} samples, max abs err = {diff:e} (peak {:.6})",
            input.len(),
            got.len(),
            max_abs(&want)
        );
        assert!(diff < 1e-4, "max abs err {diff:e} >= 1e-4");
    }

    #[test]
    fn resample_chirp_down() {
        let input = read_f32("chirp_44100.f32");
        let want = read_f32("chirp_44100_to_16000.f32");
        let got = resample(&input, 44100, 16000);
        assert_eq!(got.len(), want.len(), "output length");
        let diff = max_abs_diff(&got, &want);
        println!(
            "resample chirp 44100 -> 16000: {} -> {} samples, max abs err = {diff:e} (peak {:.6})",
            input.len(),
            got.len(),
            max_abs(&want)
        );
        assert!(diff < 1e-4, "max abs err {diff:e} >= 1e-4");
    }

    #[test]
    fn resample_sine_up() {
        let input = read_f32("sine_22050.f32");
        let want = read_f32("sine_22050_to_44100.f32");
        let got = resample(&input, 22050, 44100);
        assert_eq!(got.len(), want.len(), "output length");
        let diff = max_abs_diff(&got, &want);
        println!(
            "resample sine 22050 -> 44100: {} -> {} samples, max abs err = {diff:e} (peak {:.6})",
            input.len(),
            got.len(),
            max_abs(&want)
        );
        assert!(diff < 1e-4, "max abs err {diff:e} >= 1e-4");
    }

    #[test]
    fn resample_identity_and_empty() {
        let input = read_f32("sine_22050.f32");
        let want = read_f32("sine_22050_to_22050.f32");
        let got = resample(&input, 22050, 22050);
        assert_eq!(got, want, "same-rate must return the input unchanged");
        assert!(resample(&[], 48000, 16000).is_empty());
        println!("resample x -> x: {} samples returned unchanged", got.len());
    }

    /// The WAV header is byte-identical to `torchaudio.save`'s; only the PEAK
    /// timestamp (wall clock, `time(NULL)` in libsndfile) is masked.
    const PEAK_TIMESTAMP: std::ops::Range<usize> = 60..64;

    fn compare_wav_bytes(name: &str) -> (usize, usize) {
        let reference = goldens_dir().join(format!("{name}.wav"));
        let want = std::fs::read(&reference)
            .unwrap_or_else(|e| panic!("read {}: {e}", reference.display()));
        let out = std::env::temp_dir().join(format!("m5_audio_{name}.wav"));
        // Re-read the payload the golden was written from so the writer is
        // exercised through the public API only.
        let input = read_f32(&format!("{name}_in.f32"));
        let declared = read_meta_shape(&format!("{name}_in.f32"));
        let rate = read_meta_rate(&format!("{name}_in.f32"));
        if declared.len() == 2 {
            let channels = declared[0];
            let frames = declared[1];
            assert_eq!(input.len(), channels * frames);
            let views: Vec<&[f32]> = (0..channels)
                .map(|c| &input[c * frames..(c + 1) * frames])
                .collect();
            write_wav_channels(&out.to_string_lossy(), &views, rate).expect("write wav");
        } else {
            write_wav(&out.to_string_lossy(), &input, rate).expect("write wav");
        }
        let got = std::fs::read(&out).unwrap_or_else(|e| panic!("read back: {e}"));
        assert_eq!(
            got.len(),
            want.len(),
            "{name}: file length (header layout differs)"
        );
        let mut differing = 0usize;
        let header_differing = {
            let mut n = 0;
            for i in 0..got.len() {
                let masked = PEAK_TIMESTAMP.contains(&i);
                if got[i] != want[i] {
                    if masked {
                        continue;
                    }
                    n += 1;
                    if n <= 4 {
                        println!("  {name}: first diff at byte {i}: {:#04x} != {:#04x}", got[i], want[i]);
                    }
                }
            }
            n
        };
        differing += header_differing;
        println!(
            "write_wav {name}: {} bytes written, {} bytes != torchaudio.save (PEAK timestamp masked)",
            got.len(),
            differing
        );
        assert_eq!(differing, 0, "{name}: {} bytes differ", differing);
        (got.len(), differing)
    }

    fn meta_entry(name: &str) -> serde_json::Value {
        let path = goldens_dir().join("meta.json");
        let text = std::fs::read_to_string(&path).expect("read meta.json");
        let meta: serde_json::Value = serde_json::from_str(&text).expect("parse meta.json");
        meta["f32"]
            .get(name)
            .unwrap_or_else(|| panic!("meta.json has no entry for {name}"))
            .clone()
    }

    fn read_meta_shape(name: &str) -> Vec<usize> {
        meta_entry(name)["shape"]
            .as_array()
            .expect("shape array")
            .iter()
            .map(|v| v.as_u64().expect("shape entry") as usize)
            .collect()
    }

    fn read_meta_rate(name: &str) -> u32 {
        meta_entry(name)["sample_rate"]
            .as_u64()
            .unwrap_or_else(|| panic!("meta.json has no sample_rate for {name}")) as u32
    }

    #[test]
    fn write_wav_matches_torchaudio_save_bytes() {
        let cases = [
            "wav_ref_1ch_44100",
            "wav_ref_2ch_16000",
            "wav_ref_zeros_44100",
            "wav_ref_clip_44100",
            "wav_ref_tie_44100",
            "wav_ref_one_44100",
            "wav_ref_odd_16000",
        ];
        let mut total = 0usize;
        for name in cases {
            let (bytes, differing) = compare_wav_bytes(name);
            assert_eq!(differing, 0);
            total += bytes;
        }
        println!("write_wav: {}/{} files byte-identical", cases.len(), cases.len());
        println!("  total {total} bytes compared");
    }

    /// A real model output from the Python pipeline (32-bit float, 44.1 kHz):
    /// read it, write it back, and the bytes match except the PEAK timestamp.
    #[test]
    fn roundtrip_pipeline_output_wav() {
        let path = repo_file("output/params_test/A1_bare.wav");
        if !std::path::Path::new(&path).is_file() {
            println!("skipping: {path} not present");
            return;
        }
        let want = std::fs::read(&path).expect("read A1_bare.wav");
        let (samples, rate) = read_wav(&path).expect("read A1_bare.wav");
        println!(
            "A1_bare.wav: {} samples @ {rate} Hz, peak {:.6}",
            samples.len(),
            max_abs(&samples)
        );
        assert_eq!(rate, 44100);
        // The fact chunk says 141120 frames; the file is 80 + 141120 * 4 bytes.
        assert_eq!(samples.len(), 141120);
        assert_eq!(want.len(), 80 + samples.len() * 4);
        let out = std::env::temp_dir().join("m5_audio_A1_bare_roundtrip.wav");
        write_wav(&out.to_string_lossy(), &samples, rate).expect("write wav");
        let got = std::fs::read(&out).expect("read back");
        assert_eq!(got.len(), want.len());
        let mut differing = 0usize;
        for i in 0..got.len() {
            if got[i] != want[i] && !PEAK_TIMESTAMP.contains(&i) {
                differing += 1;
            }
        }
        println!(
            "A1_bare roundtrip: {} bytes, {} bytes differ from the Python output (PEAK timestamp masked)",
            got.len(),
            differing
        );
        assert_eq!(differing, 0);
    }

    #[test]
    fn roundtrip_lingguang_wav_is_byte_identical() {
        let reference = goldens_dir().join("wav_ref_lingguang_48000.wav");
        let want = std::fs::read(&reference).expect("read reference wav");
        let path = repo_file("data/voice_presets/voice_audio/lingguang.wav");
        let (samples, rate) = read_wav(&path).expect("read lingguang.wav");
        assert_eq!(rate, 48000);
        // read -> write must reproduce the Python-written float32 file.
        let out = std::env::temp_dir().join("m5_audio_lingguang_roundtrip.wav");
        write_wav(&out.to_string_lossy(), &samples, rate).expect("write wav");
        let got = std::fs::read(&out).expect("read back");
        assert_eq!(got.len(), want.len(), "file length");
        let mut differing = 0usize;
        for i in 0..got.len() {
            if got[i] != want[i] && !PEAK_TIMESTAMP.contains(&i) {
                differing += 1;
            }
        }
        println!(
            "roundtrip lingguang: {} samples, {} bytes, {} bytes differ from torchaudio.save (PEAK timestamp masked)",
            samples.len(),
            got.len(),
            differing
        );
        assert_eq!(differing, 0);

        // ... and reading the float32 file back yields the same samples.
        let (again, rate2) = read_wav(&out.to_string_lossy()).expect("re-read");
        assert_eq!(rate2, rate);
        assert_eq!(again.len(), samples.len());
        let diff = max_abs_diff(&again, &samples);
        println!("roundtrip: max abs diff after re-read = {diff:e}");
        assert_eq!(diff, 0.0);
    }
}
