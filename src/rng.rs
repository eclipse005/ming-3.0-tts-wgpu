//! torch-compatible normal RNG.
//!
//! The reference pipeline draws its noise with `torch.randn(..., device='cuda')`.
//! Reproducing the same utterances sample for sample therefore needs the same
//! numbers, and "same seed" only means anything if the generator matches.
//!
//! Reverse-engineered from the reference's own draws (see PROGRESS.md):
//! CUDA torch.randn is curand's Philox4x32-10 with
//! * key: `(seed & 0xffffffff, seed >> 32)`
//! * counter.x: the draw index (one per torch.randn call, +1 each)
//! * counter.z: the element index inside that call
//! * value: `box_muller(uniform(r0), uniform(r1)).sin`
//!
//! with `curand_uniform(x) = x * 2^-32 + 2^-33`.
//! Validated against 3,680 recorded values: max abs error 9.5e-7 (torch uses
//! the fast __sincosf intrinsic, we use the exact sin).

const M0: u64 = 0xD2511F53;
const M1: u64 = 0xCD9E8D57;
const W0: u32 = 0x9E3779B9;
const W1: u32 = 0xBB67AE85;
const TWO_POW_M32: f32 = 2.3283064e-10;
const TWO_POW_M33: f32 = 1.1641532e-10;

#[inline]
fn philox4x32_10(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let p0 = M0.wrapping_mul(c[0] as u64);
        let p1 = M1.wrapping_mul(c[2] as u64);
        c = [
            ((p1 >> 32) as u32) ^ c[1] ^ k[0],
            p1 as u32,
            ((p0 >> 32) as u32) ^ c[3] ^ k[1],
            p0 as u32,
        ];
        k[0] = k[0].wrapping_add(W0);
        k[1] = k[1].wrapping_add(W1);
    }
    c
}

#[inline]
fn uniform(x: u32) -> f32 {
    x as f32 * TWO_POW_M32 + TWO_POW_M33
}

#[inline]
fn normal_from(u: [u32; 4]) -> f32 {
    let r = (-2.0f32 * uniform(u[0]).ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * uniform(u[1]);
    r * theta.sin()
}

/// Drop-in replacement for the reference's `torch.randn` stream.
pub struct Philox {
    key: [u32; 2],
    draw: u32,
}

impl Philox {
    pub fn new(seed: i64) -> Self {
        let s = seed as u64;
        Self { key: [s as u32, (s >> 32) as u32], draw: 0 }
    }

    /// One `torch.randn(n)` call: `n` normals, and the shared draw counter advances.
    pub fn draw(&mut self, n: usize) -> Vec<f32> {
        let d = self.draw;
        self.draw = self.draw.wrapping_add(1);
        (0..n)
            .map(|k| normal_from(philox4x32_10([d, 0, k as u32, 0], self.key)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::read_npy;

    /// The recorded torch.randn stream from a real reference run.
    #[test]
    fn matches_torch_randn() {
        crate::need_goldens!();
        let dir = std::env::var("MING_GOLDEN")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string());
        let dir = std::path::Path::new(&dir).join("m5");
        let mut rng = Philox::new(1895);
        let mut worst = 0f32;
        let mut total = 0usize;
        for i in 0..80 {
            let path = dir.join(format!("noise_{i:03}.npy"));
            if !path.exists() {
                break;
            }
            let want = read_npy(&path).unwrap();
            let got = rng.draw(want.data.len());
            for (a, b) in got.iter().zip(want.data.iter()) {
                worst = worst.max((a - b).abs());
            }
            total += want.data.len();
        }
        println!("torch randn parity: {total} values, max abs error {worst:.3e}");
        assert!(total > 3000, "expected the full recorded stream");
        assert!(worst < 2e-6, "max abs error {worst:.3e}");
    }
}