//! Golden-file support: a minimal .npy reader (numpy.save v1.0, C order) and
//! the rel_l2 metric used by every parity assertion.
//!
//! This lives outside the tests because the CLI needs it too: MING_NOISE_GOLDEN
//! replays the reference's torch.randn draws, which are stored as .npy.

// Parts of this module are exercised only by the parity tests (they read
// goldens) or kept as public API; the release binary does not call them.

use anyhow::{bail, Context, Result};

pub struct Npy {
    /// Read by the parity tests; the runtime only needs `data`.
    #[allow(dead_code)]
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn read_npy(path: &std::path::Path) -> Result<Npy> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if &bytes[0..6] != b"\x93NUMPY" {
        bail!("not an npy file: {}", path.display());
    }
    let major = bytes[6];
    let (hdr_len, off) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[off..off + hdr_len])?;
    let descr = extract(header, "'descr': '")?;
    let fortran = extract(header, "'fortran_order': ")?;
    if fortran.trim() != "False" {
        bail!("fortran-order npy not supported");
    }
    let shape = parse_shape(&extract(header, "'shape': (")?);
    let dtype = match descr.as_str() {
        "<f4" => D::F32,
        "<f8" => D::F64,
        "<i8" => D::I64,
        "|b1" => D::Bool,
        other => bail!("unsupported npy dtype {other}"),
    };
    let payload = &bytes[off + hdr_len..];
    let data = match dtype {
        D::F32 => payload
            .as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        D::F64 => payload
            .as_chunks::<8>().0.iter()
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        D::I64 => payload
            .as_chunks::<8>().0.iter()
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        D::Bool => payload.iter().map(|&b| b as f32).collect(),
    };
    Ok(Npy { shape, data })
}

enum D {
    F32,
    F64,
    I64,
    Bool,
}

fn extract<'a>(header: &'a str, key: &'a str) -> Result<String> {
    let start = header
        .find(key)
        .with_context(|| format!("header missing {key}"))?
        + key.len();
    let rest = &header[start..];
    let end = rest
        .find([',', '}'])
        .with_context(|| format!("unterminated {key}"))?;
    Ok(rest[..end]
        .trim()
        .trim_matches('\'')
        .trim_matches('"')
        .to_string())
}

fn parse_shape(s: &str) -> Vec<usize> {
    s.trim_end_matches(',')
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect()
}

#[cfg(test)]
pub fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    (num.sqrt() / den.sqrt().max(1e-30)) as f32
}

/// The verification goldens (~67 MB of `.npy`/`.pt` dumped from the Python
/// reference) are not part of the source repository; `tools/dump_*.py`
/// regenerate them against the reference checkout. Tests that compare against
/// them skip when the directory is absent instead of failing on a missing file.
#[cfg(test)]
pub fn goldens_present() -> bool {
    static PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PRESENT.get_or_init(|| {
        let present = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens").is_dir();
        if !present {
            // Straight to the OS handle: `cargo test` captures the print macros
            // for passing tests, and a silent 3-second "80 passed" would look
            // like everything was verified.
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "\n[skip] goldens/ is absent -- the numeric and audio tests are \
                 skipped, only the text/logic ones run (see README, Tests)\n"
            );
        }
        present
    })
}

/// First line of any test that reads the numeric/audio goldens.
#[cfg(test)]
#[macro_export]
macro_rules! need_goldens {
    () => {
        if !$crate::golden::goldens_present() {
            eprintln!("[skip] goldens/ absent (see README, Verification)");
            return;
        }
    };
}
