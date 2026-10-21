//! Streaming zero-copy tensor loader into the static arena (@C:MMAP, A4, A5, A6).
//!
//! Streams tensors from read-only memory mappings into the page-aligned static arena,
//! performing on-the-fly Q4_0 AoS -> SoA repacking, enforcing 4096-B alignment and 64-B
//! padding for zero-copy Intel OpenCL compatibility, and advising the Linux kernel with
//! `posix_fadvise(POSIX_FADV_DONTNEED)` to drop page cache pages during load.

use crate::arena::{Arena, ArenaError, ArenaRegion, ArenaSlice};
use crate::budget::{BudgetError, MemoryBudget, MemoryBudgetPlan};
use crate::gguf::{GgmlType, GgufError, GgufMmap};
use crate::model_config::{ConfigError, ModelConfig};
use crate::repack::{q4_0_soa_sizes, repack_q4_0_aos_to_soa, RepackError};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use thiserror::Error;

/// Errors returned during model streaming and loading.
#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("GGUF error: {0}")]
    Gguf(#[from] GgufError),
    #[error("configuration error: {0}")]
    Config(#[from] ConfigError),
    #[error("budget error: {0}")]
    Budget(#[from] BudgetError),
    #[error("arena memory error: {0}")]
    Arena(#[from] ArenaError),
    #[error("repack error for tensor '{0}': {1}")]
    Repack(String, RepackError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A tensor loaded into page-aligned arena memory.
pub enum LoadedTensorData {
    /// Q4_0 SoA layout: separated contiguous scales and quants arrays.
    Q4_0 {
        scales: ArenaSlice,
        quants: ArenaSlice,
        num_blocks: usize,
    },
    /// Raw un-repacked tensor (F32, F16, Q8_0).
    Raw(ArenaSlice),
}

/// Metadata and arena storage handle for an active model tensor.
pub struct LoadedTensor {
    pub name: String,
    pub ggml_type: GgmlType,
    pub dims: Vec<u64>,
    pub data: LoadedTensorData,
}

impl LoadedTensor {
    /// Verifies that all pointers and padded sizes satisfy Intel zero-copy invariants ($ZCOPY).
    pub fn assert_zcopy_invariants(&self) {
        match &self.data {
            LoadedTensorData::Q4_0 { scales, quants, .. } => {
                assert_eq!(
                    (scales.as_ptr() as usize) % 4096,
                    0,
                    "tensor '{}' scales pointer must be 4096-aligned",
                    self.name
                );
                assert_eq!(
                    scales.padded_size() % 64,
                    0,
                    "tensor '{}' scales padded size must be multiple of 64",
                    self.name
                );
                assert_eq!(
                    (quants.as_ptr() as usize) % 4096,
                    0,
                    "tensor '{}' quants pointer must be 4096-aligned",
                    self.name
                );
                assert_eq!(
                    quants.padded_size() % 64,
                    0,
                    "tensor '{}' quants padded size must be multiple of 64",
                    self.name
                );
            }
            LoadedTensorData::Raw(slice) => {
                assert_eq!(
                    (slice.as_ptr() as usize) % 4096,
                    0,
                    "tensor '{}' pointer must be 4096-aligned",
                    self.name
                );
                assert_eq!(
                    slice.padded_size() % 64,
                    0,
                    "tensor '{}' padded size must be multiple of 64",
                    self.name
                );
            }
        }
    }
}

/// An fully loaded and initialized model ready for inference.
pub struct LoadedModel {
    pub config: ModelConfig,
    pub plan: MemoryBudgetPlan,
    pub tensors: HashMap<String, LoadedTensor>,
    pub arena: Arena,
}

impl LoadedModel {
    /// Loads a model from a GGUF file path into a static arena.
    pub fn load<P: AsRef<Path>>(
        path: P,
        requested_ctx: Option<usize>,
    ) -> Result<Self, LoaderError> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let file_fd = file.as_raw_fd();

        let gguf_mmap = GgufMmap::open(path_ref)?;
        let gguf = gguf_mmap.file();

        let config = ModelConfig::from_gguf(gguf)?;
        config.validate_tensor_shapes(gguf)?;

        let plan = MemoryBudget::plan(gguf, &config, requested_ctx)?;

        // Allocate static arena with the planned total capacity
        let mut arena = Arena::new(plan.total_planned_bytes)?;

        let mut tensors = HashMap::with_capacity(gguf.tensors.len());

        for t in &gguf.tensors {
            let tensor_raw_bytes = gguf.tensor_data(t)?;

            let loaded_data = match t.ggml_type {
                GgmlType::Q4_0 => {
                    let num_elements = t.num_elements();
                    let (scales_len, quants_len) = q4_0_soa_sizes(num_elements);

                    let mut scales_slice = arena.alloc(scales_len, ArenaRegion::Weights)?;
                    let mut quants_slice = arena.alloc(quants_len, ArenaRegion::Weights)?;

                    let num_blocks = repack_q4_0_aos_to_soa(
                        tensor_raw_bytes,
                        scales_slice.as_mut_slice(),
                        quants_slice.as_mut_slice(),
                    )
                    .map_err(|e| LoaderError::Repack(t.name.to_string(), e))?;

                    LoadedTensorData::Q4_0 {
                        scales: scales_slice,
                        quants: quants_slice,
                        num_blocks,
                    }
                }
                _ => {
                    let mut slice = arena.alloc(t.raw_size, ArenaRegion::Weights)?;
                    slice.as_mut_slice().copy_from_slice(tensor_raw_bytes);
                    LoadedTensorData::Raw(slice)
                }
            };

            let loaded_tensor = LoadedTensor {
                name: t.name.to_string(),
                ggml_type: t.ggml_type,
                dims: t.dims.clone(),
                data: loaded_data,
            };

            // Assert $ZCOPY zero-copy alignment constraints
            loaded_tensor.assert_zcopy_invariants();

            tensors.insert(t.name.to_string(), loaded_tensor);

            // Evict file page cache chunk after copying into the arena (A5)
            #[cfg(target_os = "linux")]
            unsafe {
                let abs_offset = (gguf.data_start + t.offset as usize) as libc::off_t;
                let chunk_len = t.raw_size as libc::off_t;
                libc::posix_fadvise(file_fd, abs_offset, chunk_len, libc::POSIX_FADV_DONTNEED);
            }
        }

        Ok(Self {
            config,
            plan,
            tensors,
            arena,
        })
    }
}

/// Reads current and peak RSS (VmHWM) in bytes from `/proc/self/status`.
pub fn get_process_memory_stats() -> (u64, u64) {
    let mut vm_rss = 0;
    let mut vm_hwm = 0;

    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(val) = parse_status_kb(line) {
                    vm_rss = val * 1024;
                }
            } else if line.starts_with("VmHWM:") {
                if let Some(val) = parse_status_kb(line) {
                    vm_hwm = val * 1024;
                }
            }
        }
    }

    (vm_rss, vm_hwm)
}

fn parse_status_kb(line: &str) -> Option<u64> {
    let mut parts = line.split_whitespace();
    parts.next(); // Skip label
    parts.next().and_then(|s| s.parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{create_test_gguf, GgufValueType, GGUF_MAGIC, GGUF_VERSION_V3};
    use std::io::Write;

    #[test]
    fn test_load_synthetic_llama_model() {
        let mut kvs: Vec<(&str, GgufValueType, &[u8])> = Vec::new();
        let arch_bytes = [5u8, 0, 0, 0, 0, 0, 0, 0, b'l', b'l', b'a', b'm', b'a'];
        let ctx_bytes = (512u32).to_le_bytes();
        let embd_bytes = (128u32).to_le_bytes();
        let blk_bytes = (2u32).to_le_bytes();
        let ffn_bytes = (512u32).to_le_bytes();
        let heads_bytes = (4u32).to_le_bytes();
        let heads_kv_bytes = (2u32).to_le_bytes();

        kvs.push(("general.architecture", GgufValueType::String, &arch_bytes));
        kvs.push(("llama.context_length", GgufValueType::Uint32, &ctx_bytes));
        kvs.push(("llama.embedding_length", GgufValueType::Uint32, &embd_bytes));
        kvs.push(("llama.block_count", GgufValueType::Uint32, &blk_bytes));
        kvs.push((
            "llama.feed_forward_length",
            GgufValueType::Uint32,
            &ffn_bytes,
        ));
        kvs.push((
            "llama.attention.head_count",
            GgufValueType::Uint32,
            &heads_bytes,
        ));
        kvs.push((
            "llama.attention.head_count_kv",
            GgufValueType::Uint32,
            &heads_kv_bytes,
        ));

        // Create 2 synthetic Q4_0 tensors (each 128 elements = 4 blocks = 72 bytes)
        let q4_data = [0x33; 72];
        let raw_norm = [0xAA; 512];
        let tensors = [
            (
                "token_embd.weight",
                &[128, 1][..],
                2u32, /* Q4_0 */
                0u64,
                &q4_data[..],
            ),
            (
                "output_norm.weight",
                &[128][..],
                0u32, /* F32 */
                128u64,
                &raw_norm[..],
            ),
        ];

        let gguf_bytes = create_test_gguf(GGUF_MAGIC, GGUF_VERSION_V3, &kvs, &tensors);

        let temp_path =
            std::env::temp_dir().join(format!("sulimo_test_model_{}.gguf", std::process::id()));
        let mut file = File::create(&temp_path).unwrap();
        file.write_all(&gguf_bytes).unwrap();
        file.flush().unwrap();
        drop(file);

        let model = LoadedModel::load(&temp_path, Some(512)).expect("load model");
        assert_eq!(model.tensors.len(), 2);

        for t in model.tensors.values() {
            t.assert_zcopy_invariants();
        }

        let (rss, hwm) = get_process_memory_stats();
        assert!(rss > 0);
        assert!(hwm > 0);

        let _ = std::fs::remove_file(&temp_path);
    }
}
