//! GGUF v3 parser and zero-copy metadata reader (@C:MMAP).
//!
//! Parses GGUF v3 files according to the specification (§6.1) without heap allocations
//! for tensor metadata where possible, enforcing boundary validation and version checks.

use memmap2::{Mmap, MmapOptions};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;
use thiserror::Error;

/// GGUF file format magic number: 0x46554747 ("GGUF" in little-endian ASCII).
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Supported GGUF specification version.
pub const GGUF_VERSION_V3: u32 = 3;

/// Errors returned during GGUF parsing and validation.
#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error reading GGUF file: {0}")]
    Io(#[from] std::io::Error),
    #[error(
        "file truncated: needed {needed} bytes at offset {offset}, but file has {total} bytes"
    )]
    UnexpectedEof {
        offset: usize,
        needed: usize,
        total: usize,
    },
    #[error("invalid GGUF magic: expected 0x{expected:08x}, got 0x{actual:08x}")]
    InvalidMagic { expected: u32, actual: u32 },
    #[error("unsupported GGUF version {0} (only v3 is supported in v0.1)")]
    UnsupportedVersion(u32),
    #[error("invalid UTF-8 string at offset {offset}: {source}")]
    InvalidUtf8 {
        offset: usize,
        #[source]
        source: std::str::Utf8Error,
    },
    #[error("unknown GGUF value type ID {0}")]
    UnknownValueType(u32),
    #[error("unsupported GGML tensor type ID {type_id} for tensor '{name}' (v0.1 supports F32, F16, Q4_0, Q8_0)")]
    UnsupportedTensorType { name: String, type_id: u32 },
    #[error("tensor '{name}' offset {offset} + size {size} exceeds file length {total}")]
    TensorOutOfBounds {
        name: String,
        offset: usize,
        size: usize,
        total: usize,
    },
    #[error("duplicate tensor name '{0}'")]
    DuplicateTensor(String),
}

/// GGUF metadata value types (§6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn from_u32(val: u32) -> Result<Self, GgufError> {
        match val {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(GgufError::UnknownValueType(other)),
        }
    }
}

/// A parsed GGUF metadata value borrowing strings directly from the mmap.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(&'a str),
    Array(Vec<GgufValue<'a>>),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// GGML quant/tensor types supported in v0.1.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    Q6_K = 14,
    Bf16 = 30,
}

impl GgmlType {
    pub fn from_u32(val: u32, tensor_name: &str) -> Result<Self, GgufError> {
        match val {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            14 => Err(GgufError::UnsupportedTensorType {
                name: tensor_name.to_string(),
                type_id: val,
            }),
            30 => Ok(Self::Bf16),
            other => Err(GgufError::UnsupportedTensorType {
                name: tensor_name.to_string(),
                type_id: other,
            }),
        }
    }

    /// Block size: number of values per quant block (1 for unquantized types, 32 for Q4_0 and Q8_0).
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q6_K => 256,
        }
    }

    /// Byte size per block.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18, // 2-byte f16 scale + 16 bytes (32 x 4-bit)
            Self::Q8_0 => 34, // 2-byte f16 scale + 32 bytes (32 x 8-bit)
            Self::Q6_K => 210,
        }
    }

    /// Calculates raw byte size for a tensor with the given number of elements.
    pub fn calculate_size_bytes(&self, num_elements: usize) -> usize {
        let bs = self.block_size();
        let ts = self.type_size();
        (num_elements / bs) * ts
    }
}

/// Description of a tensor stored in the GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
    pub raw_size: usize,
}

impl<'a> TensorInfo<'a> {
    /// Total number of scalar weights in this tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

/// A zero-copy view of a parsed GGUF v3 file.
#[derive(Debug)]
pub struct GgufFile<'a> {
    pub version: u32,
    pub metadata: BTreeMap<&'a str, GgufValue<'a>>,
    pub tensors: Vec<TensorInfo<'a>>,
    pub tensor_index: HashMap<&'a str, usize>,
    pub alignment: usize,
    pub data_start: usize,
    raw_data: &'a [u8],
}

impl<'a> GgufFile<'a> {
    /// Parses a GGUF v3 file from a byte slice.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, GgufError> {
        let mut cur = 0;

        let magic = read_u32(bytes, &mut cur)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic {
                expected: GGUF_MAGIC,
                actual: magic,
            });
        }

        let version = read_u32(bytes, &mut cur)?;
        if version != GGUF_VERSION_V3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = read_u64(bytes, &mut cur)? as usize;
        let metadata_kv_count = read_u64(bytes, &mut cur)? as usize;

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(bytes, &mut cur)?;
            let val_type_id = read_u32(bytes, &mut cur)?;
            let val_type = GgufValueType::from_u32(val_type_id)?;
            let value = read_value(bytes, &mut cur, val_type)?;
            metadata.insert(key, value);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) => *a as usize,
            Some(GgufValue::U64(a)) => *a as usize,
            _ => 32, // Default alignment per GGUF specification
        };

        let mut tensors = Vec::with_capacity(tensor_count);
        let mut tensor_index = HashMap::with_capacity(tensor_count);

        for _ in 0..tensor_count {
            let name = read_string(bytes, &mut cur)?;
            let n_dims = read_u32(bytes, &mut cur)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(bytes, &mut cur)?);
            }
            let type_id = read_u32(bytes, &mut cur)?;
            let ggml_type = GgmlType::from_u32(type_id, name)?;
            let offset = read_u64(bytes, &mut cur)?;

            let num_elements: usize = dims.iter().product::<u64>() as usize;
            let raw_size = ggml_type.calculate_size_bytes(num_elements);

            if tensor_index.contains_key(name) {
                return Err(GgufError::DuplicateTensor(name.to_string()));
            }

            let idx = tensors.len();
            tensor_index.insert(name, idx);
            tensors.push(TensorInfo {
                name,
                n_dims,
                dims,
                ggml_type,
                offset,
                raw_size,
            });
        }

        let data_start = align_up(cur, alignment);

        // Validate all tensor boundaries against the file length
        for t in &tensors {
            let abs_start = data_start + t.offset as usize;
            let abs_end = abs_start.saturating_add(t.raw_size);
            if abs_end > bytes.len() {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.to_string(),
                    offset: abs_start,
                    size: t.raw_size,
                    total: bytes.len(),
                });
            }
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            tensor_index,
            alignment,
            data_start,
            raw_data: bytes,
        })
    }

    /// Gets a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&GgufValue<'a>> {
        self.metadata.get(key)
    }

    /// Gets a string metadata value.
    pub fn get_str(&self, key: &str) -> Option<&'a str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(s)) => Some(*s),
            _ => None,
        }
    }

    /// Gets a u32 metadata value (also handles u64 if it fits).
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufValue::U32(v)) => Some(*v),
            Some(GgufValue::U64(v)) => u32::try_from(*v).ok(),
            Some(GgufValue::I32(v)) if *v >= 0 => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Gets a u64 metadata value.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(GgufValue::U64(v)) => Some(*v),
            Some(GgufValue::U32(v)) => Some(*v as u64),
            _ => None,
        }
    }

    /// Gets a f32 metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufValue::F32(v)) => Some(*v),
            Some(GgufValue::F64(v)) => Some(*v as f32),
            _ => None,
        }
    }

    /// Gets a bool metadata value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key) {
            Some(GgufValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Gets an array of strings metadata value.
    pub fn get_arr_str(&self, key: &str) -> Option<Vec<&'a str>> {
        match self.metadata.get(key) {
            Some(GgufValue::Array(items)) => {
                let mut res = Vec::with_capacity(items.len());
                for item in items {
                    if let GgufValue::String(s) = item {
                        res.push(*s);
                    } else {
                        return None;
                    }
                }
                Some(res)
            }
            _ => None,
        }
    }

    /// Looks up a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo<'a>> {
        self.tensor_index.get(name).map(|&idx| &self.tensors[idx])
    }

    /// Returns a zero-copy slice of the raw tensor bytes from the file mmap.
    pub fn tensor_data(&self, tensor: &TensorInfo) -> Result<&'a [u8], GgufError> {
        let abs_start = self.data_start + tensor.offset as usize;
        let abs_end = abs_start + tensor.raw_size;
        if abs_end <= self.raw_data.len() {
            Ok(&self.raw_data[abs_start..abs_end])
        } else {
            Err(GgufError::TensorOutOfBounds {
                name: tensor.name.to_string(),
                offset: abs_start,
                size: tensor.raw_size,
                total: self.raw_data.len(),
            })
        }
    }
}

/// Owned GGUF mmap container.
pub struct GgufMmap {
    _mmap: Mmap,
    // Pointer to the mmap bytes with lifetime managed by `_mmap`
    file: GgufFile<'static>,
}

impl GgufMmap {
    /// Opens and memory-maps a GGUF file with `MADV_SEQUENTIAL` and `MADV_WILLNEED`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let mmap = unsafe {
            // SAFETY: File is opened read-only and memory-mapped private read-only.
            MmapOptions::new().map(&file)?
        };

        // Advise kernel of sequential read access for initial metadata loading
        #[cfg(target_os = "linux")]
        unsafe {
            // SAFETY: Mmap pointer and length are valid.
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_SEQUENTIAL | libc::MADV_WILLNEED,
            );
        }

        // Transmute slice lifetime to 'static because Mmap will be kept alongside GgufFile
        let slice: &'static [u8] = unsafe {
            // SAFETY: The slice is valid for the duration of the GgufMmap struct.
            std::slice::from_raw_parts(mmap.as_ptr(), mmap.len())
        };

        let parsed = GgufFile::parse(slice)?;

        Ok(Self {
            _mmap: mmap,
            file: parsed,
        })
    }

    pub fn file(&self) -> &GgufFile<'_> {
        &self.file
    }
}

pub fn align_up(val: usize, align: usize) -> usize {
    assert!(
        align > 0 && (align & (align - 1)) == 0,
        "align must be power of 2"
    );
    (val + align - 1) & !(align - 1)
}

fn read_exact<'a>(bytes: &'a [u8], cur: &mut usize, len: usize) -> Result<&'a [u8], GgufError> {
    let end = cur.saturating_add(len);
    if end > bytes.len() {
        return Err(GgufError::UnexpectedEof {
            offset: *cur,
            needed: len,
            total: bytes.len(),
        });
    }
    let slice = &bytes[*cur..end];
    *cur = end;
    Ok(slice)
}

fn read_u8(bytes: &[u8], cur: &mut usize) -> Result<u8, GgufError> {
    let s = read_exact(bytes, cur, 1)?;
    Ok(s[0])
}

fn read_i8(bytes: &[u8], cur: &mut usize) -> Result<i8, GgufError> {
    let s = read_exact(bytes, cur, 1)?;
    Ok(s[0] as i8)
}

fn read_u16(bytes: &[u8], cur: &mut usize) -> Result<u16, GgufError> {
    let s = read_exact(bytes, cur, 2)?;
    Ok(u16::from_le_bytes(s.try_into().unwrap()))
}

fn read_i16(bytes: &[u8], cur: &mut usize) -> Result<i16, GgufError> {
    let s = read_exact(bytes, cur, 2)?;
    Ok(i16::from_le_bytes(s.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], cur: &mut usize) -> Result<u32, GgufError> {
    let s = read_exact(bytes, cur, 4)?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}

fn read_i32(bytes: &[u8], cur: &mut usize) -> Result<i32, GgufError> {
    let s = read_exact(bytes, cur, 4)?;
    Ok(i32::from_le_bytes(s.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], cur: &mut usize) -> Result<u64, GgufError> {
    let s = read_exact(bytes, cur, 8)?;
    Ok(u64::from_le_bytes(s.try_into().unwrap()))
}

fn read_i64(bytes: &[u8], cur: &mut usize) -> Result<i64, GgufError> {
    let s = read_exact(bytes, cur, 8)?;
    Ok(i64::from_le_bytes(s.try_into().unwrap()))
}

fn read_f32(bytes: &[u8], cur: &mut usize) -> Result<f32, GgufError> {
    let s = read_exact(bytes, cur, 4)?;
    Ok(f32::from_le_bytes(s.try_into().unwrap()))
}

fn read_f64(bytes: &[u8], cur: &mut usize) -> Result<f64, GgufError> {
    let s = read_exact(bytes, cur, 8)?;
    Ok(f64::from_le_bytes(s.try_into().unwrap()))
}

fn read_string<'a>(bytes: &'a [u8], cur: &mut usize) -> Result<&'a str, GgufError> {
    let offset = *cur;
    let len = read_u64(bytes, cur)? as usize;
    let slice = read_exact(bytes, cur, len)?;
    std::str::from_utf8(slice).map_err(|e| GgufError::InvalidUtf8 { offset, source: e })
}

fn read_value<'a>(
    bytes: &'a [u8],
    cur: &mut usize,
    val_type: GgufValueType,
) -> Result<GgufValue<'a>, GgufError> {
    match val_type {
        GgufValueType::Uint8 => Ok(GgufValue::U8(read_u8(bytes, cur)?)),
        GgufValueType::Int8 => Ok(GgufValue::I8(read_i8(bytes, cur)?)),
        GgufValueType::Uint16 => Ok(GgufValue::U16(read_u16(bytes, cur)?)),
        GgufValueType::Int16 => Ok(GgufValue::I16(read_i16(bytes, cur)?)),
        GgufValueType::Uint32 => Ok(GgufValue::U32(read_u32(bytes, cur)?)),
        GgufValueType::Int32 => Ok(GgufValue::I32(read_i32(bytes, cur)?)),
        GgufValueType::Float32 => Ok(GgufValue::F32(read_f32(bytes, cur)?)),
        GgufValueType::Bool => Ok(GgufValue::Bool(read_u8(bytes, cur)? != 0)),
        GgufValueType::String => Ok(GgufValue::String(read_string(bytes, cur)?)),
        GgufValueType::Array => {
            let elem_type_id = read_u32(bytes, cur)?;
            let elem_type = GgufValueType::from_u32(elem_type_id)?;
            let len = read_u64(bytes, cur)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value(bytes, cur, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        GgufValueType::Uint64 => Ok(GgufValue::U64(read_u64(bytes, cur)?)),
        GgufValueType::Int64 => Ok(GgufValue::I64(read_i64(bytes, cur)?)),
        GgufValueType::Float64 => Ok(GgufValue::F64(read_f64(bytes, cur)?)),
    }
}

#[cfg(test)]
pub mod test_utils {
    use super::*;

    pub type TestTensorSpec<'a> = (&'a str, &'a [u64], u32, u64, &'a [u8]);

    /// Helper to construct a valid in-memory GGUF buffer for testing.
    pub fn create_test_gguf(
        magic: u32,
        version: u32,
        kvs: &[(&str, GgufValueType, &[u8])],
        tensors: &[TestTensorSpec],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&magic.to_le_bytes());
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());

        for (k, vtype, val_bytes) in kvs {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&(*vtype as u32).to_le_bytes());
            buf.extend_from_slice(val_bytes);
        }

        for (name, dims, type_id, offset, _) in tensors {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&type_id.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }

        // Align data start to 32 bytes
        let data_start = align_up(buf.len(), 32);
        buf.resize(data_start, 0);

        for (_, _, _, offset, data) in tensors {
            let start = data_start + *offset as usize;
            let end = start + data.len();
            if buf.len() < end {
                buf.resize(end, 0);
            }
            buf[start..end].copy_from_slice(data);
        }

        buf
    }
}

#[cfg(test)]
pub use test_utils::create_test_gguf;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_gguf() {
        let mut kv_arch = Vec::new();
        kv_arch.extend_from_slice(&(5u64).to_le_bytes());
        kv_arch.extend_from_slice(b"llama");

        let raw_tensor_data = vec![0xAB; 32 * 4]; // 32 floats = 128 bytes
        let test_data = create_test_gguf(
            GGUF_MAGIC,
            GGUF_VERSION_V3,
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[(
                "token_embd.weight",
                &[32],
                0, /* F32 */
                0,
                &raw_tensor_data,
            )],
        );

        let gguf = GgufFile::parse(&test_data).expect("valid gguf must parse");
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.get_str("general.architecture"), Some("llama"));
        assert_eq!(gguf.tensors.len(), 1);
        let t = gguf.tensor("token_embd.weight").expect("tensor must exist");
        assert_eq!(t.num_elements(), 32);
        assert_eq!(t.raw_size, 128);

        let data = gguf.tensor_data(t).expect("tensor data slice");
        assert_eq!(data.len(), 128);
        assert_eq!(data[0], 0xAB);
    }

    #[test]
    fn test_bad_magic() {
        let test_data = create_test_gguf(0x12345678, GGUF_VERSION_V3, &[], &[]);
        let err = GgufFile::parse(&test_data).unwrap_err();
        assert!(matches!(err, GgufError::InvalidMagic { .. }));
    }

    #[test]
    fn test_bad_version() {
        let test_data = create_test_gguf(GGUF_MAGIC, 2, &[], &[]);
        let err = GgufFile::parse(&test_data).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion(2)));
    }

    #[test]
    fn test_truncated_header() {
        let test_data = vec![0x47, 0x47, 0x55, 0x46]; // Only 4 bytes
        let err = GgufFile::parse(&test_data).unwrap_err();
        assert!(matches!(err, GgufError::UnexpectedEof { .. }));
    }

    #[test]
    fn test_tensor_oob() {
        let mut buf = create_test_gguf(
            GGUF_MAGIC,
            GGUF_VERSION_V3,
            &[],
            &[("bad_tensor", &[32], 0 /* F32 */, 1000, &[0u8; 128])],
        );
        // Truncate the buffer so offset 1000 is out of bounds
        buf.truncate(buf.len() - 64);
        let err = GgufFile::parse(&buf).unwrap_err();
        assert!(matches!(err, GgufError::TensorOutOfBounds { .. }));
    }

    #[test]
    fn test_unsupported_tensor_type() {
        let test_data = create_test_gguf(
            GGUF_MAGIC,
            GGUF_VERSION_V3,
            &[],
            &[("unsupported", &[32], 99 /* invalid */, 0, &[0u8; 32])],
        );
        let err = GgufFile::parse(&test_data).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedTensorType { .. }));
    }
}
