//! CPU topology, physical core mapping, and SIMD feature detection (@C:CPU).
//!
//! Provides hardware discovery to enforce physical-core pinning and asymmetric
//! thermal scheduling without SMT/HyperThreading contention (A10).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Errors that may occur during CPU topology and feature probing.
#[derive(Debug, Error)]
pub enum CpuTopologyError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse CPU list '{0}'")]
    InvalidCpuList(String),
}

/// SIMD and architectural feature flags relevant to Súlimo inference kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CpuFeatures {
    /// Intel SSE4.2 support.
    pub sse4_2: bool,
    /// AVX 256-bit vector extensions.
    pub avx: bool,
    /// AVX2 integer & vector extensions (required for primary CPU decode path).
    pub avx2: bool,
    /// Fused Multiply-Add (FMA3) extensions (required for dot-product accumulation).
    pub fma: bool,
    /// 16-bit floating-point conversion instructions.
    pub f16c: bool,
    /// AVX-512 Foundation instructions (Ice Lake+).
    pub avx512f: bool,
}

impl CpuFeatures {
    /// Detects SIMD features using runtime CPUID probing.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse4_2: is_x86_feature_detected!("sse4.2"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                fma: is_x86_feature_detected!("fma"),
                f16c: is_x86_feature_detected!("f16c"),
                avx512f: is_x86_feature_detected!("avx512f"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::default()
        }
    }

    /// Whether this CPU meets the AVX2+FMA requirement for fast decode.
    pub fn has_avx2_fma(&self) -> bool {
        self.avx2 && self.fma
    }
}

/// A physical CPU core with its primary thread and all SMT/HyperThreading siblings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalCore {
    /// Hardware physical core ID assigned by the CPU topology.
    pub core_id: u32,
    /// Hardware socket / package ID.
    pub package_id: u32,
    /// The primary logical CPU ID selected for worker pinning.
    pub primary_cpu: usize,
    /// All logical CPU IDs sharing this physical core (SMT siblings).
    pub smt_siblings: Vec<usize>,
}

/// Detailed CPU topology information for Súlimo thread scheduling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuTopology {
    /// Vendor identification (e.g. "GenuineIntel", "AuthenticAMD").
    pub vendor_id: String,
    /// Human-readable brand/model string.
    pub model_name: String,
    /// Detected SIMD and instruction set features.
    pub features: CpuFeatures,
    /// Discovered physical cores mapped to primary pinning targets.
    pub physical_cores: Vec<PhysicalCore>,
    /// Total number of logical execution threads.
    pub logical_core_count: usize,
    /// Last-level L3 cache size in bytes (if available).
    pub l3_cache_bytes: Option<u64>,
}

impl CpuTopology {
    /// Probes the host system CPU topology and features.
    pub fn detect() -> Result<Self, CpuTopologyError> {
        let features = CpuFeatures::detect();
        let (vendor_id, model_name) = Self::read_proc_cpuinfo();
        let physical_cores = Self::discover_physical_cores()?;
        let logical_core_count = num_cpus::get();
        let l3_cache_bytes = Self::read_l3_cache_size();

        Ok(Self {
            vendor_id,
            model_name,
            features,
            physical_cores,
            logical_core_count,
            l3_cache_bytes,
        })
    }

    /// Returns the list of logical CPU IDs to pin for decode threads (one per physical core).
    pub fn decode_pin_targets(&self) -> Vec<usize> {
        self.physical_cores.iter().map(|c| c.primary_cpu).collect()
    }

    /// Parses `/proc/cpuinfo` to extract vendor and model strings.
    fn read_proc_cpuinfo() -> (String, String) {
        let content = match fs::read_to_string("/proc/cpuinfo") {
            Ok(c) => c,
            Err(_) => return ("Unknown".to_string(), "Unknown CPU".to_string()),
        };

        let mut vendor = "Unknown".to_string();
        let mut model = "Unknown CPU".to_string();

        for line in content.lines() {
            if line.starts_with("vendor_id") && vendor == "Unknown" {
                if let Some((_, v)) = line.split_once(':') {
                    vendor = v.trim().to_string();
                }
            } else if line.starts_with("model name") && model == "Unknown CPU" {
                if let Some((_, m)) = line.split_once(':') {
                    model = m.trim().to_string();
                }
            }
        }

        (vendor, model)
    }

    /// Discovers physical cores and SMT siblings from `/sys/devices/system/cpu/cpu*/topology`.
    fn discover_physical_cores() -> Result<Vec<PhysicalCore>, CpuTopologyError> {
        let sys_cpu_dir = Path::new("/sys/devices/system/cpu");
        if !sys_cpu_dir.exists() {
            // Fallback for non-standard environments: use num_cpus
            let count = num_cpus::get_physical().max(1);
            let cores = (0..count)
                .map(|i| PhysicalCore {
                    core_id: i as u32,
                    package_id: 0,
                    primary_cpu: i,
                    smt_siblings: vec![i],
                })
                .collect();
            return Ok(cores);
        }

        let mut core_map: BTreeMap<(u32, u32), (usize, Vec<usize>)> = BTreeMap::new();

        let entries = fs::read_dir(sys_cpu_dir).map_err(|e| CpuTopologyError::Io {
            path: sys_cpu_dir.display().to_string(),
            source: e,
        })?;

        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if !name.starts_with("cpu") {
                continue;
            }
            let cpu_id_str = &name[3..];
            let cpu_id: usize = match cpu_id_str.parse() {
                Ok(id) => id,
                Err(_) => continue,
            };

            let top_dir = entry.path().join("topology");
            if !top_dir.exists() {
                continue;
            }

            let core_id: u32 = fs::read_to_string(top_dir.join("core_id"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(cpu_id as u32);

            let package_id: u32 = fs::read_to_string(top_dir.join("physical_package_id"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);

            let siblings_str = fs::read_to_string(top_dir.join("thread_siblings_list"))
                .unwrap_or_else(|_| cpu_id.to_string());
            let siblings = parse_cpu_list(&siblings_str)?;

            let entry = core_map
                .entry((package_id, core_id))
                .or_insert_with(|| (cpu_id, siblings.clone()));

            // Update primary_cpu to the minimum logical CPU ID in this physical core
            if cpu_id < entry.0 {
                entry.0 = cpu_id;
            }
            for s in siblings {
                if !entry.1.contains(&s) {
                    entry.1.push(s);
                }
            }
            entry.1.sort_unstable();
        }

        if core_map.is_empty() {
            let count = num_cpus::get_physical().max(1);
            let cores = (0..count)
                .map(|i| PhysicalCore {
                    core_id: i as u32,
                    package_id: 0,
                    primary_cpu: i,
                    smt_siblings: vec![i],
                })
                .collect();
            return Ok(cores);
        }

        let mut result = Vec::new();
        for ((package_id, core_id), (primary_cpu, mut smt_siblings)) in core_map {
            smt_siblings.sort_unstable();
            result.push(PhysicalCore {
                core_id,
                package_id,
                primary_cpu,
                smt_siblings,
            });
        }

        result.sort_by_key(|c| (c.package_id, c.core_id));
        Ok(result)
    }

    /// Reads L3 cache size from sysfs cache hierarchy for cpu0.
    fn read_l3_cache_size() -> Option<u64> {
        let cache_base = Path::new("/sys/devices/system/cpu/cpu0/cache");
        if !cache_base.exists() {
            return None;
        }

        let entries = fs::read_dir(cache_base).ok()?;
        for entry in entries.flatten() {
            let index_path = entry.path();
            let level_path = index_path.join("level");
            if let Ok(level_str) = fs::read_to_string(level_path) {
                if level_str.trim() == "3" {
                    let size_path = index_path.join("size");
                    if let Ok(size_str) = fs::read_to_string(size_path) {
                        return parse_cache_size(&size_str);
                    }
                }
            }
        }
        None
    }
}

/// Parses a Linux CPU list format like "0,2", "0-3", or "0,2-4,7" into a sorted `Vec<usize>`.
pub fn parse_cpu_list(input: &str) -> Result<Vec<usize>, CpuTopologyError> {
    let mut cpus = Vec::new();
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(cpus);
    }

    for part in trimmed.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let mut range = part.split('-');
            let start: usize = range
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| CpuTopologyError::InvalidCpuList(input.to_string()))?;
            let end: usize = range
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| CpuTopologyError::InvalidCpuList(input.to_string()))?;
            for cpu in start..=end {
                cpus.push(cpu);
            }
        } else {
            let cpu: usize = part
                .parse()
                .map_err(|_| CpuTopologyError::InvalidCpuList(input.to_string()))?;
            cpus.push(cpu);
        }
    }

    cpus.sort_unstable();
    cpus.dedup();
    Ok(cpus)
}

/// Parses a cache size string such as "3072K", "3M", "256K", or "3145728" into bytes.
pub fn parse_cache_size(input: &str) -> Option<u64> {
    let s = input.trim();
    if s.is_empty() {
        return None;
    }

    let (num_str, mult) = if let Some(stripped) = s.strip_suffix('K') {
        (stripped, 1024)
    } else if let Some(stripped) = s.strip_suffix('k') {
        (stripped, 1024)
    } else if let Some(stripped) = s.strip_suffix('M') {
        (stripped, 1024 * 1024)
    } else if let Some(stripped) = s.strip_suffix('m') {
        (stripped, 1024 * 1024)
    } else if let Some(stripped) = s.strip_suffix('G') {
        (stripped, 1024 * 1024 * 1024)
    } else {
        (s, 1)
    };

    num_str.parse::<u64>().ok().map(|n| n * mult)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpu_list_simple() {
        assert_eq!(parse_cpu_list("0,2").unwrap(), vec![0, 2]);
        assert_eq!(parse_cpu_list("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0,1,2,3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0-1,4-5").unwrap(), vec![0, 1, 4, 5]);
        assert_eq!(parse_cpu_list("4").unwrap(), vec![4]);
    }

    #[test]
    fn test_parse_cache_size() {
        assert_eq!(parse_cache_size("3072K"), Some(3072 * 1024));
        assert_eq!(parse_cache_size("3M"), Some(3 * 1024 * 1024));
        assert_eq!(parse_cache_size("256k"), Some(256 * 1024));
        assert_eq!(parse_cache_size("1048576"), Some(1048576));
    }

    #[test]
    fn test_live_topology() {
        let topo = CpuTopology::detect().expect("CPU topology detection should succeed");
        assert!(!topo.physical_cores.is_empty());
        assert!(topo.logical_core_count >= topo.physical_cores.len());
        // Verify primary cpu is in smt_siblings
        for core in &topo.physical_cores {
            assert!(core.smt_siblings.contains(&core.primary_cpu));
        }
    }
}
