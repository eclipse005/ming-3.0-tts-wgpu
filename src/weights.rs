//! safetensors loader keeping weights as raw bf16 (packed 2/u32 for shaders).
//!
//! The checkpoint stores bf16. We never convert to f16 (ASR project style):
//! bf16 -> f32 is a lossless 16-bit shift, done inside shaders, so the math
//! matches the Python fp32 path exactly while halving bus traffic.

// Parts of this module are exercised only by the parity tests (they read
// goldens) or kept as public API; the release binary does not call them.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Bf16,
    I64,
    Other,
}

impl DType {
    fn parse(descr: safetensors::tensor::Dtype) -> Self {
        use safetensors::tensor::Dtype as S;
        match descr {
            S::F32 => DType::F32,
            S::F16 => DType::F16,
            S::BF16 => DType::Bf16,
            S::I64 => DType::I64,
            _ => DType::Other,
        }
    }
}

pub struct TensorEntry {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

impl TensorEntry {
    /// bf16 -> f32 lossless widen; f32 passthrough.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        match self.dtype {
            DType::F32 => Ok(self
                .data
                .as_chunks::<4>().0.iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            DType::Bf16 => Ok(self.bf16_iter().collect()),
            DType::F16 => bail!("f16 weights are not used by this port (bf16 checkpoint)"),
            _ => bail!("unsupported dtype {:?} for f32 conversion", self.dtype),
        }
    }

    pub fn bf16_iter(&self) -> impl Iterator<Item = f32> + '_ {
        self.data.as_chunks::<2>().0.iter().map(|c| {
            let bits = (c[0] as u32) | ((c[1] as u32) << 8);
            f32::from_bits(bits << 16)
        })
    }

    /// Pack bf16 pairs into u32: low half = even column, high half = odd column.
    /// Odd element count is zero-padded (callers pad the activation to match).
    pub fn to_bf16_packed_u32(&self) -> Result<Vec<u32>> {
        if self.dtype != DType::Bf16 {
            bail!("packed u32 accessor requires bf16 weights");
        }
        let (words, rem) = self.data.as_chunks::<4>();
        let mut out = Vec::with_capacity(words.len() + usize::from(!rem.is_empty()));
        for c in words {
            out.push(
                (c[0] as u32)
                    | ((c[1] as u32) << 8)
                    | ((c[2] as u32) << 16)
                    | ((c[3] as u32) << 24),
            );
        }
        match rem {
            [] => {}
            [lo, hi] => out.push((*lo as u32) | ((*hi as u32) << 8)),
            _ => bail!("bf16 data length {} is not a multiple of 2", self.data.len()),
        }
        Ok(out)
    }
}

pub struct Weights {
    map: HashMap<String, TensorEntry>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap).context("parse safetensors")?;
        let mut map = HashMap::new();
        for (name, view) in st.iter() {
            map.insert(
                name.to_string(),
                TensorEntry {
                    shape: view.shape().to_vec(),
                    dtype: DType::parse(view.dtype()),
                    data: view.data().to_vec(),
                },
            );
        }
        Ok(Self { map })
    }

    pub fn get(&self, name: &str) -> Result<&TensorEntry> {
        self.map
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    #[cfg(test)]
    pub fn names_matching(&self, pat: &str) -> Vec<&String> {
        let mut v: Vec<&String> = self
            .map
            .keys()
            .filter(|k| k.contains(pat))
            .collect();
        v.sort();
        v
    }
}

/// bf16 bit widen used everywhere (single source of truth for the semantics).
#[inline]
pub fn bf16_widen(u: u32) -> f32 {
    f32::from_bits(u << 16)
}