//! Súlimo CPU backend (@C:CPU).
//!
//! Provides topology detection, physical-core pinning, and AVX2/FMA execution primitives.

pub mod topology;

pub use topology::{parse_cpu_list, CpuFeatures, CpuTopology, CpuTopologyError, PhysicalCore};
