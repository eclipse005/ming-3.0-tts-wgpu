//! Audio I/O: WAV read/write and torchaudio-compatible resampling.

pub fn read_wav(path: &str) -> anyhow::Result<(Vec<f32>, u32)> {
    let _ = path;
    todo!()
}

pub fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    let _ = (path, samples, sample_rate);
    todo!()
}

pub fn resample(x: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let _ = (x, from_rate, to_rate);
    todo!()
}
