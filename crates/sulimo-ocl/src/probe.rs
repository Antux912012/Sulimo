//! OpenCL device capability discovery and extension verification (@C:OCL-GEN9).
//!
//! Probes OpenCL platforms and devices at runtime without panicking, verifying
//! Intel Gen8–11 FP16, sub-group, and unified memory support (A16).

use serde::{Deserialize, Serialize};

/// Detailed device attributes discovered during the OpenCL capability probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OclDeviceInfo {
    /// OpenCL platform name (e.g. "Intel(R) OpenCL HD Graphics").
    pub platform_name: String,
    /// OpenCL platform version string.
    pub platform_version: String,
    /// OpenCL device name (e.g. "Intel(R) HD Graphics 620").
    pub device_name: String,
    /// Hardware vendor string.
    pub device_vendor: String,
    /// Driver version string.
    pub driver_version: String,
    /// Execution units / compute units (`CL_DEVICE_MAX_COMPUTE_UNITS`).
    pub compute_units: u32,
    /// Memory base address alignment in bits (`CL_DEVICE_MEM_BASE_ADDR_ALIGN`).
    pub base_addr_align_bits: u32,
    /// Whether host and device share unified physical RAM (`CL_DEVICE_HOST_UNIFIED_MEMORY`).
    pub host_unified_memory: bool,
    /// Whether `cl_khr_fp16` extension is supported for half-precision math.
    pub has_fp16: bool,
    /// Whether `cl_intel_subgroups` extension is supported for SIMD8 primitives.
    pub has_intel_subgroups: bool,
    /// Whether `cl_intel_required_subgroup_size` is supported.
    pub has_intel_reqd_subgroup_size: bool,
    /// Full list of reported device extensions.
    pub extensions: Vec<String>,
}

impl OclDeviceInfo {
    /// Returns true if this device meets all requirements for the `hybrid-gen9` tier.
    pub fn is_hybrid_gen9_capable(&self) -> bool {
        self.has_fp16
            && (self.has_intel_subgroups
                || self.has_intel_reqd_subgroup_size
                || self.host_unified_memory)
    }
}

/// The result of an OpenCL runtime probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "details")]
pub enum OclProbeResult {
    /// A suitable OpenCL device is present and initialized.
    Available(Box<OclDeviceInfo>),
    /// OpenCL runtime or compatible device is unavailable.
    Unavailable { reason: String },
}

impl OclProbeResult {
    /// Returns true if an OpenCL device is available.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }
}

/// Probes the host system for OpenCL devices.
///
/// Under no circumstances does this function panic. If `libOpenCL.so.1` is missing,
/// no ICDs are registered, or device queries fail, it returns [`OclProbeResult::Unavailable`].
pub fn probe() -> OclProbeResult {
    #[cfg(feature = "opencl")]
    {
        probe_opencl_internal()
    }

    #[cfg(not(feature = "opencl"))]
    {
        OclProbeResult::Unavailable {
            reason: "OpenCL support disabled at compile time (--no-default-features)".to_string(),
        }
    }
}

#[cfg(feature = "opencl")]
pub fn ensure_icd_env() {
    if std::env::var("OCL_ICD_VENDORS").is_err() {
        if let Ok(home) = std::env::var("HOME") {
            let user_icd = std::path::Path::new(&home).join(".local/share/sulimo/drivers/icd");
            if user_icd.exists() {
                // SAFETY: setting environment variable before OpenCL platform lookup.
                unsafe { std::env::set_var("OCL_ICD_VENDORS", user_icd) };
            }
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        let user_lib = std::path::Path::new(&home).join(".local/share/sulimo/drivers/lib");
        if user_lib.exists() {
            let cur = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            if !cur.contains(user_lib.to_str().unwrap_or_default()) {
                let new_ld = if cur.is_empty() {
                    user_lib.to_string_lossy().to_string()
                } else {
                    format!("{}:{cur}", user_lib.display())
                };
                unsafe { std::env::set_var("LD_LIBRARY_PATH", new_ld) };
            }
        }
    }
}

#[cfg(feature = "opencl")]
fn probe_opencl_internal() -> OclProbeResult {
    use opencl3::device::{Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU};
    use opencl3::platform::get_platforms;

    ensure_icd_env();

    let platforms = match get_platforms() {
        Ok(p) => p,
        Err(e) => {
            return OclProbeResult::Unavailable {
                reason: format!("No OpenCL platforms found or driver missing ({e})"),
            };
        }
    };

    if platforms.is_empty() {
        return OclProbeResult::Unavailable {
            reason: "Zero OpenCL platforms returned by ICD loader".to_string(),
        };
    }

    let mut candidate_devices = Vec::new();

    for platform in &platforms {
        let platform_name = platform
            .name()
            .unwrap_or_else(|_| "Unknown Platform".into());
        let platform_version = platform
            .version()
            .unwrap_or_else(|_| "Unknown Version".into());

        // First look for GPUs
        let mut dev_ids = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default();
        if dev_ids.is_empty() {
            // Fallback to all devices if no GPU is reported
            dev_ids = platform.get_devices(CL_DEVICE_TYPE_ALL).unwrap_or_default();
        }

        for dev_id in dev_ids {
            let device = Device::new(dev_id);
            let device_name = device.name().unwrap_or_else(|_| "Unknown Device".into());
            let device_vendor = device.vendor().unwrap_or_else(|_| "Unknown Vendor".into());
            let driver_version = device
                .driver_version()
                .unwrap_or_else(|_| "Unknown Driver".into());

            let compute_units = device.max_compute_units().unwrap_or_default();
            let base_addr_align_bits = device.mem_base_addr_align().unwrap_or_default();
            let host_unified_memory = device.host_unified_memory().unwrap_or_default();

            let extensions_raw = device.extensions().unwrap_or_default();
            let extensions: Vec<String> = extensions_raw
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            let has_fp16 = extensions.iter().any(|ext| ext == "cl_khr_fp16");
            let has_intel_subgroups = extensions.iter().any(|ext| ext == "cl_intel_subgroups");
            let has_intel_reqd_subgroup_size = extensions
                .iter()
                .any(|ext| ext == "cl_intel_required_subgroup_size");

            candidate_devices.push(OclDeviceInfo {
                platform_name: platform_name.clone(),
                platform_version: platform_version.clone(),
                device_name,
                device_vendor,
                driver_version,
                compute_units,
                base_addr_align_bits,
                host_unified_memory,
                has_fp16,
                has_intel_subgroups,
                has_intel_reqd_subgroup_size,
                extensions,
            });
        }
    }

    if candidate_devices.is_empty() {
        return OclProbeResult::Unavailable {
            reason: "No OpenCL devices found across installed platforms".to_string(),
        };
    }

    // Prefer Intel GPUs or devices with FP16
    if let Some(pos) = candidate_devices.iter().position(|d| {
        d.device_vendor.to_lowercase().contains("intel")
            || d.device_name.to_lowercase().contains("intel")
            || d.has_fp16
    }) {
        let chosen = candidate_devices.remove(pos);
        OclProbeResult::Available(Box::new(chosen))
    } else {
        let first = candidate_devices.remove(0);
        OclProbeResult::Available(Box::new(first))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_does_not_panic() {
        // Must succeed without panicking even on systems with no OpenCL drivers
        let result = probe();
        match result {
            OclProbeResult::Available(info) => {
                assert!(!info.device_name.is_empty());
            }
            OclProbeResult::Unavailable { reason } => {
                assert!(!reason.is_empty());
            }
        }
    }
}
