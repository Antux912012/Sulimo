//! Execution tier classification for Súlimo.
//!
//! Súlimo supports asymmetric compute tiers depending on host capabilities:
//! - `hybrid-gen9`: iGPU OpenCL FP16 SIMD8 prefill + CPU physical core AVX2+FMA decode
//! - `cpu-avx2`: CPU physical core AVX2+FMA decode + AVX2 prefill (OpenCL disabled or unavailable)
//! - `cpu-scalar`: Scalar reference fallback (CPUs lacking AVX2/FMA, e.g. Celeron/Pentium)

use serde::{Deserialize, Serialize};

/// Target execution tier selected during hardware capability probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionTier {
    /// Asymmetric iGPU OpenCL FP16 SIMD8 + CPU AVX2/FMA.
    HybridGen9,
    /// Pure CPU AVX2+FMA physical-core inference.
    CpuAvx2,
    /// Scalar CPU fallback for older/restricted x86_64 CPUs.
    CpuScalar,
}

impl ExecutionTier {
    /// Canonical string identifier for the execution tier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HybridGen9 => "hybrid-gen9",
            Self::CpuAvx2 => "cpu-avx2",
            Self::CpuScalar => "cpu-scalar",
        }
    }
}

impl std::fmt::Display for ExecutionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
