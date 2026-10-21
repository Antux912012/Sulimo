//! Engine capability synthesis and execution tier selection (@C:ENG).
//!
//! Synthesizes CPU, GPU, and system memory probe results into a cohesive
//! capability report and selects the execution tier.

use serde::{Deserialize, Serialize};
use sulimo_core::{ExecutionTier, SysMem, SysMemError};
use sulimo_cpu::{CpuTopology, CpuTopologyError};
use sulimo_ocl::{probe as probe_ocl, OclProbeResult};
use thiserror::Error;

/// Errors that may occur during engine capability probing.
#[derive(Debug, Error)]
pub enum CapabilityError {
    #[error("failed to probe system memory: {0}")]
    SysMem(#[from] SysMemError),
    #[error("failed to probe CPU topology: {0}")]
    Cpu(#[from] CpuTopologyError),
}

/// Consolidated hardware capability report for Súlimo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityReport {
    /// Host system memory details.
    pub sysmem: SysMem,
    /// Host CPU topology and SIMD features.
    pub cpu: CpuTopology,
    /// OpenCL device probe result.
    pub ocl: OclProbeResult,
    /// The selected execution tier.
    pub selected_tier: ExecutionTier,
    /// Human-readable rationale for the tier selection.
    pub tier_reason: String,
}

impl CapabilityReport {
    /// Conducts a complete hardware capability probe and resolves the execution tier.
    pub fn probe() -> Result<Self, CapabilityError> {
        let sysmem = SysMem::read()?;
        let cpu = CpuTopology::detect()?;
        let ocl = probe_ocl();

        let (selected_tier, tier_reason) = Self::select_tier(&cpu, &ocl);

        Ok(Self {
            sysmem,
            cpu,
            ocl,
            selected_tier,
            tier_reason,
        })
    }

    /// Determines the execution tier from CPU features and OpenCL capabilities.
    pub fn select_tier(cpu: &CpuTopology, ocl: &OclProbeResult) -> (ExecutionTier, String) {
        if !cpu.features.has_avx2_fma() {
            return (
                ExecutionTier::CpuScalar,
                "CPU lacks AVX2/FMA vector extensions; running scalar fallback".to_string(),
            );
        }

        match ocl {
            OclProbeResult::Available(dev) if dev.is_hybrid_gen9_capable() => (
                ExecutionTier::HybridGen9,
                format!(
                    "Intel iGPU with FP16 ({}) + CPU AVX2/FMA physical-core decode",
                    dev.device_name
                ),
            ),
            OclProbeResult::Available(dev) => (
                ExecutionTier::CpuAvx2,
                format!(
                    "GPU '{}' lacks required Gen8-11 FP16/subgroup extensions; using AVX2+FMA physical-core decode",
                    dev.device_name
                ),
            ),
            OclProbeResult::Unavailable { reason } => (
                ExecutionTier::CpuAvx2,
                format!(
                    "OpenCL unavailable ({reason}); using AVX2+FMA physical-core decode"
                ),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_succeeds() {
        let report = CapabilityReport::probe().expect("capability probe must succeed");
        assert!(report.sysmem.total_bytes > 0);
        assert!(!report.cpu.physical_cores.is_empty());
        assert!(!report.tier_reason.is_empty());
    }
}
