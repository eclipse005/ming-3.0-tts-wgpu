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
    if !interleaved.len().is_multiple_of(channels) {
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

/// Trim trailing silence exactly like `ming_tts.trim_trailing_silence`.
pub fn trim_trailing_silence(speech: &[f32], sample_rate: u32) -> Vec<f32> {
    let sample_rate = sample_rate as usize;
    let frame = sample_rate / 10;
    let tail = sample_rate * 3 / 10;
    if frame == 0 || speech.len() < frame {
        return speech[..speech.len().min(tail)].to_vec();
    }
    // The reference first truncates to a whole number of 0.1 s frames and only
    // then slices the tail off that truncated signal; slicing the original
    // length instead leaves the partial frame in place.
    let num_frames = (speech.len() - frame) / frame + 1;
    let body = &speech[..num_frames * frame];
    let mut scores = vec![0f32; num_frames];
    for (i, s) in scores.iter_mut().enumerate() {
        let seg = &body[i * frame..(i + 1) * frame];
        *s = seg.iter().map(|v| v.abs()).sum::<f32>() / frame as f32;
    }
    let mut index = num_frames as isize - 1;
    while index >= 0 && scores[index as usize] <= 1e-3 {
        index -= 1;
    }
    if index < 0 {
        return body[..body.len().min(tail)].to_vec();
    }
    let end = (index as usize * frame + frame + tail).min(body.len());
    body[..end].to_vec()
}

// ---------------------------------------------------------------- tests
