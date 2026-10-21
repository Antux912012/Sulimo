//! Súlimo Core library (@C:MMAP).
//!
//! Provides GGUF v3 parsing, static page-aligned arena allocation ($ZCOPY),
//! 5.2 GB memory budget planning ($MEM<8G), Q4_0 SoA block repacking,
//! and streaming zero-copy model loading.

pub mod arena;
pub mod budget;
pub mod gguf;
pub mod loader;
pub mod model_config;
pub mod repack;
pub mod sysmem;
pub mod tier;

pub use arena::{Arena, ArenaError, ArenaRegion, ArenaSlice};
pub use budget::{BudgetError, MemoryBudget, MemoryBudgetPlan};
pub use gguf::{GgmlType, GgufError, GgufFile, GgufMmap, GgufValue, TensorInfo};
pub use loader::{
    get_process_memory_stats, LoadedModel, LoadedTensor, LoadedTensorData, LoaderError,
};
pub use model_config::{ConfigError, ModelArch, ModelConfig};
pub use repack::{
    dequantize_q4_0_aos, dequantize_q4_0_soa, q4_0_soa_sizes, repack_q4_0_aos_to_soa, RepackError,
    Q4_0_AOS_BLOCK_BYTES, Q4_0_BLOCK_SIZE,
};
pub use sysmem::{SysMem, SysMemError, ThpMode, MAX_RAM_BUDGET_BYTES};
pub use tier::ExecutionTier;
