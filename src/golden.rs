//! A minimal .npy reader (numpy.save v1.0, C order).
//!
//! `MING_NOISE_GOLDEN` replays the reference's recorded torch.randn draws,
//! which is what makes a sample-for-sample comparison possible.

use anyhow::{bail, Context, Result};

/// The payload of an `.npy` file, flattened to f32.
pub struct Npy {
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
    Ok(Npy { data })
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
