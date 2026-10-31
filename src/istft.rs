//! ISTFT head (CPU side): irfft(n_fft) -> Hann window -> overlap-add ->
//! envelope normalisation with padding="same" -- a faithful port of
//! audio_tokenizer/istft.py (ISTFT.forward, non-streaming branch).
//!
//! Only ~12.5 frames per second of audio, so a CPU rustfft is far cheaper than
//! shipping an FFT kernel to Vulkan.

use rustfft::{num_complex::Complex32, FftPlanner};

pub struct Istft {
    n_fft: usize,
    hop: usize,
    win: Vec<f32>,
    pad: usize,
    plan: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Istft {
    pub fn new(n_fft: usize, hop: usize, win: Vec<f32>) -> Self {
        assert_eq!(win.len(), n_fft, "window length must equal n_fft");
        let mut planner = FftPlanner::new();
        Self {
            n_fft,
            hop,
            win,
            pad: (n_fft - hop) / 2,
            plan: planner.plan_fft_inverse(n_fft),
        }
    }

    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// spec is row-major [t_frames][n_bins] as Complex32 (re, im): the layout
    /// torch gets after mag * (cos + j sin) and permute(0, 2, 1).
    /// Returns the trimmed waveform, length (T-1)*hop + win - 2*pad.
    pub fn forward(&self, spec: &[Complex32], t_frames: usize) -> Vec<f32> {
        let n = self.n_fft;
        let bins = self.n_bins();
        assert_eq!(spec.len(), t_frames * bins, "spec size mismatch");

        let out_len = (t_frames - 1) * self.hop + n;
        let mut y = vec![0f32; out_len];
        let mut env = vec![0f32; out_len];
        for t in 0..t_frames {
            let base = t * self.hop;
            for i in 0..n {
                let w = self.win[i];
                env[base + i] += w * w;
            }
        }

        let mut buf = vec![Complex32::new(0.0, 0.0); n];
        let mut frame = vec![0f32; n];
        for t in 0..t_frames {
            let row = &spec[t * bins..(t + 1) * bins];
            // Hermitian extension; torch.fft.irfft reads only .real of bin 0
            // and of the Nyquist bin.
            buf[0] = Complex32::new(row[0].re, 0.0);
            buf[n / 2] = Complex32::new(row[n / 2].re, 0.0);
            for k in 1..n / 2 {
                let v = row[k];
                buf[k] = v;
                buf[n - k] = v.conj();
            }
            self.plan.process(&mut buf);
            // norm="backward" => the inverse transform carries a 1/n factor
            let inv = 1.0 / n as f32;
            for i in 0..n {
                frame[i] = buf[i].re * inv * self.win[i];
            }
            let base = t * self.hop;
            for i in 0..n {
                y[base + i] += frame[i];
            }
        }

        let start = self.pad;
        let end = out_len - self.pad;
        let mut out = Vec::with_capacity(end - start);
        for i in start..end {
            out.push(y[i] / env[i]);
        }
        out
    }
}
