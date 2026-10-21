//! Súlimo Core library (@C:MMAP).
//!
//! Provides memory tracking, budget calculation, GGUF structures, and platform types.

pub mod sysmem;
pub mod tier;

pub use sysmem::{SysMem, SysMemError, ThpMode, MAX_RAM_BUDGET_BYTES};
pub use tier::ExecutionTier;
