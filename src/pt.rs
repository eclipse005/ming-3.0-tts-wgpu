//! Minimal reader for torch `.pt` files holding a speaker embedding.
//!
//! The reference's `--spk_emb_path` accepts files written by `torch.save`: a
//! ZIP archive whose `data.pkl` is a pickle of
//! `torch._utils._rebuild_tensor_v2(...)` and whose `data/<key>` entry holds the
//! raw little-endian storage. This module parses exactly that subset (the one
//! torch emits for `torch.save(tensor)` and `torch.save({"spk_emb": tensor})`),
//! so the port keeps its "no torch, no Python" property.
//!
//! Supported storages: FloatStorage, DoubleStorage, HalfStorage. Anything else
//! (compressed entries, legacy pre-1.6 pickles, non-tensor payloads) fails with
//! a message that says what the file actually contains.


use anyhow::{bail, Context, Result};
use std::path::Path;

/// A tensor reference recovered from the pickle.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub storage_key: String,
    pub dtype: DType,
    pub numel: u64,
    pub offset: u64,
    pub sizes: Vec<u64>,
    pub strides: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    F16,
}

/// A storage reference as resolved from a persistent id.
#[derive(Debug, Clone)]
pub struct StorageRef {
    pub key: String,
    pub dtype: DType,
    pub numel: u64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // the pickle value model; not every variant is read
enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global(String),
    Storage(Box<StorageRef>),
    Tensor(Box<TensorRef>),
}

impl Val {
    fn as_dict_get(&self, key: &str) -> Option<&Val> {
        if let Val::Dict(entries) = self {
            for (k, v) in entries {
                if let Val::Str(s) = k {
                    if s == key {
                        return Some(v);
                    }
                }
            }
        }
        None
    }
}

/// Stack machine for the pickle subset torch writes.
struct Unpickler<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Val>,
    marks: Vec<usize>,
    memo: Vec<Option<Val>>,
}

impl<'a> Unpickler<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: Vec::new(),
        }
    }

    fn le_u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn le_u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn le_i32(&mut self) -> Result<i32> {
        Ok(self.le_u32()? as i32)
    }

    fn le_f64(&mut self) -> Result<f64> {
        let b = self.take(8)?;
        Ok(f64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn u8(&mut self) -> Result<u8> {
        let b = *self.data.get(self.pos).context("pickle ended")?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("pickle length")?;
        if end > self.data.len() {
            bail!("pickle wants {n} bytes at {} of {}", self.pos, self.data.len());
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn line(&mut self) -> Result<&'a [u8]> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            bail!("unterminated pickle line");
        }
        let s = &self.data[start..self.pos];
        self.pos += 1;
        Ok(s)
    }

    fn pop(&mut self) -> Result<Val> {
        self.stack.pop().context("pickle stack underflow")
    }

    fn pop_mark(&mut self) -> Result<Vec<Val>> {
        let m = self.marks.pop().context("pickle mark underflow")?;
        if m > self.stack.len() {
            bail!("pickle mark out of range");
        }
        let items = self.stack.split_off(m);
        Ok(items)
    }

    /// ("storage", FloatStorage, "0", "cpu", numel) -> a storage reference.
    fn parse_storage(v: &Val) -> Result<StorageRef> {
        let tuple = match v {
            Val::Tuple(t) => t,
            Val::Storage(s) => return Ok((**s).clone()),
            other => bail!("expected a storage tuple, got {other:?}"),
        };
        if tuple.len() < 5 {
            bail!("storage tuple has {} entries", tuple.len());
        }
        let key = match &tuple[2] {
            Val::Str(s) => s.clone(),
            other => bail!("storage key is {other:?}"),
        };
        let dtype = match &tuple[1] {
            Val::Global(g) if g.ends_with("FloatStorage") => DType::F32,
            Val::Global(g) if g.ends_with("DoubleStorage") => DType::F64,
            Val::Global(g) if g.ends_with("HalfStorage") => DType::F16,
            other => bail!("unsupported storage type {other:?}"),
        };
        let numel = match &tuple[4] {
            Val::Int(n) => *n as u64,
            _ => 0,
        };
        Ok(StorageRef { key, dtype, numel })
    }

    /// Read the storage tuple and tensor metadata out of a `_rebuild_tensor_v2`
    /// REDUCE.
    fn rebuild_tensor(args: &[Val]) -> Result<Val> {
        let storage = match args.first() {
            Some(v) => Self::parse_storage(v)?,
            None => bail!("tensor rebuild without a storage"),
        };
        let key = storage.key.clone();
        let dtype = storage.dtype;
        let numel = storage.numel;
        let offset = match args.get(1) {
            Some(Val::Int(n)) => *n as u64,
            _ => 0,
        };
        let sizes = match args.get(2) {
            Some(Val::Tuple(t)) => t
                .iter()
                .map(|v| match v {
                    Val::Int(n) => Ok(*n as u64),
                    other => bail!("tensor size is {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?,
            _ => Vec::new(),
        };
        let strides = match args.get(3) {
            Some(Val::Tuple(t)) => t
                .iter()
                .map(|v| match v {
                    Val::Int(n) => Ok(*n as u64),
                    other => bail!("tensor stride is {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?,
            _ => Vec::new(),
        };
        Ok(Val::Tensor(Box::new(TensorRef {
            storage_key: key,
            dtype,
            numel,
            offset,
            sizes,
            strides,
        })))
    }

    fn run(&mut self) -> Result<Val> {
        loop {
            let op = self.u8()?;
            match op {
                b'\x80' => {
                    let _proto = self.u8()?;
                }
                b'c' => {
                    let module = self.line()?.to_vec();
                    let name = self.line()?.to_vec();
                    let full = format!(
                        "{}.{}",
                        String::from_utf8_lossy(&module),
                        String::from_utf8_lossy(&name)
                    );
                    self.stack.push(Val::Global(full));
                }
                b'X' => {
                    let n = self.le_u32()? as usize;
                    let s = String::from_utf8(self.take(n)?.to_vec()).context("pickle string")?;
                    self.stack.push(Val::Str(s));
                }
                b'K' => {
                    let v = self.u8()? as i64;
                    self.stack.push(Val::Int(v));
                }
                b'M' => {
                    let v = self.le_u16()?;
                    self.stack.push(Val::Int(v as i64));
                }
                b'J' => {
                    let v = self.le_i32()?;
                    self.stack.push(Val::Int(v as i64));
                }
                b'\x8a' => {
                    let n = self.u8()? as usize;
                    let bytes = self.take(n)?;
                    let mut v: i64 = if bytes.last().map(|b| b & 0x80 != 0).unwrap_or(false) {
                        -1
                    } else {
                        0
                    };
                    for (i, b) in bytes.iter().enumerate() {
                        v |= (*b as i64) << (8 * i);
                    }
                    self.stack.push(Val::Int(v));
                }
                b'G' => {
                    let v = self.le_f64()?;
                    self.stack.push(Val::Float(v));
                }
                b'(' => self.marks.push(self.stack.len()),
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Val::Tuple(items));
                }
                b'\x85' => {
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a]));
                }
                b'\x86' => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a, b]));
                }
                b'\x87' => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Val::Tuple(vec![a, b, c]));
                }
                b')' => self.stack.push(Val::Tuple(Vec::new())),
                b']' => self.stack.push(Val::List(Vec::new())),
                b'}' => self.stack.push(Val::Dict(Vec::new())),
                b'\x89' | b'\x88' => self.stack.push(Val::Bool(true)),
                b'N' => self.stack.push(Val::None),
                b'q' => {
                    let idx = self.u8()? as usize;
                    let v = self.stack.last().cloned().context("BINPUT on empty stack")?;
                    self.memo_put_at(idx, v);
                }
                b'r' => {
                    let idx = self.le_u32()? as usize;
                    let v = self.stack.last().cloned().context("LONG_BINPUT on empty stack")?;
                    self.memo_put_at(idx, v);
                }
                b'h' => {
                    let idx = self.u8()? as usize;
                    let v = self.memo.get(idx).cloned().flatten().context("memo miss")?;
                    self.stack.push(v);
                }
                b'j' => {
                    let idx = self.le_u32()? as usize;
                    let v = self.memo.get(idx).cloned().flatten().context("memo miss")?;
                    self.stack.push(v);
                }
                b'\x94' => {
                    // MEMOIZE: store the top of stack under the next index.
                    let idx = self.memo.len();
                    let v = self.stack.last().cloned().context("MEMOIZE on empty stack")?;
                    self.memo_put_at(idx, v);
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    if let Some(Val::Dict(entries)) = self.stack.last_mut() {
                        entries.push((k, v));
                    } else {
                        bail!("SETITEM on a non-dict");
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        bail!("SETITEMS with an odd number of values");
                    }
                    if let Some(Val::Dict(entries)) = self.stack.last_mut() {
                        for pair in items.chunks(2) {
                            entries.push((pair[0].clone(), pair[1].clone()));
                        }
                    } else {
                        bail!("SETITEMS on a non-dict");
                    }
                }
                b'Q' => {
                    let v = self.pop()?;
                    self.stack.push(Val::Storage(Box::new(Self::parse_storage(&v)?)));
                }
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let g = match &func {
                        Val::Global(g) => g.clone(),
                        other => bail!("REDUCE on {other:?}"),
                    };
                    if g.ends_with("_rebuild_tensor_v2") || g.ends_with("_rebuild_tensor") {
                        let args = match args {
                            Val::Tuple(t) => t,
                            other => bail!("rebuild args are {other:?}"),
                        };
                        let t = Self::rebuild_tensor(&args)?;
                        self.stack.push(t);
                    } else {
                        // collections.OrderedDict(), torch.Size(...) and similar
                        // wrappers keep the value the caller can use.
                        self.stack.push(Val::Tuple(Vec::new()));
                    }
                }
                b'b' => {
                    // BUILD: state applies to the object below; nothing to do
                    // for tensors (the state is the backward-hooks dict).
                    self.pop()?;
                }
                b'.' => break,
                other => bail!("unsupported pickle opcode 0x{other:02x} at {}", self.pos - 1),
            }
        }
        self.stack.pop().context("empty pickle")
    }

    fn memo_put_at(&mut self, idx: usize, v: Val) {
        if self.memo.len() <= idx {
            self.memo.resize(idx + 1, None);
        }
        self.memo[idx] = Some(v);
    }
}

// ---------------------------------------------------------------------------
// ZIP (stored entries only, which is what torch writes)
// ---------------------------------------------------------------------------

struct ZipEntry {
    name: String,
    offset: usize,
    size: usize,
    method: u16,
}

fn rd_u16(data: &[u8], off: usize) -> Result<u16> {
    let s = data.get(off..off + 2).context("truncated ZIP field")?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn rd_u32(data: &[u8], off: usize) -> Result<u32> {
    let s = data.get(off..off + 4).context("truncated ZIP field")?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_zip(data: &[u8]) -> Result<Vec<ZipEntry>> {
    if data.len() < 22 {
        bail!("not a ZIP file ({} bytes)", data.len());
    }
    // End of central directory: scan backwards for the signature.
    let mut eocd = None;
    for i in (0..=data.len() - 22).rev() {
        if data[i..i + 4] == [0x50, 0x4b, 0x05, 0x06] {
            eocd = Some(i);
            break;
        }
    }
    let eocd = eocd.context("not a ZIP file (no end-of-central-directory)")?;
    let count = rd_u16(data, eocd + 10)? as usize;
    let mut cd = rd_u32(data, eocd + 16)? as usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if data.get(cd..cd + 4) != Some(&[0x50, 0x4b, 0x01, 0x02]) {
            bail!("bad ZIP central directory");
        }
        let method = rd_u16(data, cd + 10)?;
        let size = rd_u32(data, cd + 20)? as usize;
        let name_len = rd_u16(data, cd + 28)? as usize;
        let extra_len = rd_u16(data, cd + 30)? as usize;
        let comment_len = rd_u16(data, cd + 32)? as usize;
        let local = rd_u32(data, cd + 42)? as usize;
        let name = String::from_utf8_lossy(
            data.get(cd + 46..cd + 46 + name_len)
                .context("ZIP entry name is out of bounds")?,
        )
        .into_owned();
        // Local header: fixed part 30 bytes + name + extra, then the payload.
        let l_name = rd_u16(data, local + 26)? as usize;
        let l_extra = rd_u16(data, local + 28)? as usize;
        let payload = local
            .checked_add(30 + l_name + l_extra)
            .context("ZIP local header overflow")?;
        entries.push(ZipEntry {
            name,
            offset: payload,
            size,
            method,
        });
        cd += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

fn entry_bytes<'a>(data: &'a [u8], e: &ZipEntry) -> Result<&'a [u8]> {
    if e.method != 0 {
        bail!(
            "{} is compressed (method {}); torch.save writes stored entries, so this \
             file came from another tool",
            e.name,
            e.method
        );
    }
    let end = e.offset.checked_add(e.size).context("ZIP entry size")?;
    data.get(e.offset..end)
        .with_context(|| format!("ZIP entry {} is out of bounds", e.name))
}

/// Load the first speaker-embedding tensor from a `.pt` file, as f32.
pub fn load_tensor(path: &Path) -> Result<Vec<f32>> {
    let data = std::fs::read(path).with_context(|| format!("open {}", path.display()))?;
    if data.starts_with(b"PK\x03\x04") {
        load_zip_tensor(path, &data)
    } else {
        bail!(
            "{} is a legacy (pre-1.6) torch pickle; re-save it with torch.save on \
             torch >= 1.6 to get the ZIP format",
            path.display()
        );
    }
}

fn load_zip_tensor(path: &Path, data: &[u8]) -> Result<Vec<f32>> {
    let entries = read_zip(data)?;
    let pkl = entries
        .iter()
        .find(|e| e.name.ends_with("data.pkl"))
        .with_context(|| format!("{} has no data.pkl", path.display()))?;
    let prefix = pkl
        .name
        .strip_suffix("data.pkl")
        .unwrap_or("")
        .to_string();
    let pickle = entry_bytes(data, pkl)?;
    let value = Unpickler::new(pickle).run().with_context(|| {
        format!(
            "{}: parse the tensor pickle (only torch.save of a tensor or a dict \
             of tensors is supported)",
            path.display()
        )
    })?;
    let tensor = match &value {
        Val::Tensor(t) => (**t).clone(),
        Val::Dict(_) => {
            let mut found = None;
            for key in ["spk_emb", "speaker_embedding", "embedding", "emb"] {
                if let Some(Val::Tensor(t)) = value.as_dict_get(key) {
                    found = Some((**t).clone());
                    break;
                }
            }
            found.with_context(|| {
                format!(
                    "{} is a dict without a speaker tensor (looked for spk_emb / \
                     speaker_embedding / embedding / emb)",
                    path.display()
                )
            })?
        }
        other => bail!("{} holds {other:?}, not a tensor", path.display()),
    };
    let storage = entries
        .iter()
        .find(|e| e.name == format!("{prefix}data/{}", tensor.storage_key))
        .with_context(|| {
            format!(
                "{} has no storage entry data/{}",
                path.display(),
                tensor.storage_key
            )
        })?;
    let bytes = entry_bytes(data, storage)?;
    let elem = match tensor.dtype {
        DType::F32 => 4,
        DType::F64 => 8,
        DType::F16 => 2,
    };
    // We read the storage linearly, so the tensor must be C-contiguous. A
    // transposed view saved with torch.save keeps the strides; silently
    // ignoring them would produce a wrong embedding, so reject it.
    if !tensor.sizes.is_empty() && !tensor.strides.is_empty() {
        let mut expected = vec![1u64; tensor.sizes.len()];
        for i in (0..tensor.sizes.len().saturating_sub(1)).rev() {
            expected[i] = expected[i + 1] * tensor.sizes[i + 1];
        }
        if tensor.strides != expected {
            bail!(
                "{} holds a non-contiguous tensor (strides {:?}, expected {:?}); \
                 call .contiguous() before torch.save",
                path.display(),
                tensor.strides,
                expected
            );
        }
    }
    let numel = if tensor.numel > 0 {
        tensor.numel as usize
    } else {
        tensor.sizes.iter().product::<u64>() as usize
    };
    let start = tensor.offset as usize * elem;
    let need = numel * elem;
    let raw = bytes
        .get(start..start + need)
        .with_context(|| format!("{}: storage is too short", path.display()))?;
    let mut out = Vec::with_capacity(numel);
    match tensor.dtype {
        DType::F32 => {
            for c in raw.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        DType::F64 => {
            for c in raw.as_chunks::<8>().0 {
                out.push(f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32);
            }
        }
        DType::F16 => {
            for c in raw.as_chunks::<2>().0 {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                out.push(half_to_f32(bits));
            }
        }
    }
    Ok(out)
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_files() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens");
        let path = base.join("spk_tensor.pt");
        if !path.exists() {
            return;
        }
        let data = std::fs::read(&path).unwrap();
        for cut in [3usize, 21, 40, data.len() / 2, data.len() - 10] {
            let truncated = base.join("_truncated.pt");
            std::fs::write(&truncated, &data[..cut]).unwrap();
            let err = load_tensor(&truncated).unwrap_err().to_string();
            assert!(!err.is_empty(), "cut {cut} should fail");
            let _ = std::fs::remove_file(&truncated);
        }
        // A file that is neither ZIP nor pickle.
        let junk = base.join("_junk.pt");
        std::fs::write(&junk, b"not a torch file").unwrap();
        assert!(load_tensor(&junk).is_err());
        let _ = std::fs::remove_file(&junk);
    }

    #[test]
    fn reads_tensor_and_dict_files() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens");
        for name in ["spk_tensor.pt", "spk_dict.pt"] {
            let path = base.join(name);
            if !path.exists() {
                continue;
            }
            let v = load_tensor(&path).unwrap();
            assert_eq!(v.len(), 192, "{name}");
            for (i, x) in v.iter().enumerate() {
                assert!((x - i as f32 / 192.0).abs() < 1e-7, "{name}[{i}] = {x}");
            }
        }
    }
}
