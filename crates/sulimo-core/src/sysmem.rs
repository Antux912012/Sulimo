//! System memory and hugepage discovery for Súlimo.
//!
//! Inspects `/proc/meminfo` and Linux sysfs transparent hugepage configurations
//! to enforce the 5.2 GB static allocation ceiling and verify zero-copy capabilities.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Hard ceiling for unified engine RAM on Gen8–10 8GB hosts: 5.2 GB.
pub const MAX_RAM_BUDGET_BYTES: u64 = 5_200 * 1024 * 1024;

/// Errors encountered while probing system memory.
#[derive(Debug, Error)]
pub enum SysMemError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("missing required field '{0}' in /proc/meminfo")]
    MissingField(&'static str),
    #[error("invalid numeric value for field '{0}': {1}")]
    InvalidValue(&'static str, String),
}

/// Linux Transparent Huge Page (THP) mode from `/sys/kernel/mm/transparent_hugepage/enabled`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThpMode {
    Always,
    Madvise,
    Never,
    Unknown(String),
}

impl std::fmt::Display for ThpMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Always => write!(f, "always"),
            Self::Madvise => write!(f, "madvise"),
            Self::Never => write!(f, "never"),
            Self::Unknown(s) => write!(f, "unknown ({})", s),
        }
    }
}

/// Host system memory statistics and hugepage status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SysMem {
    /// Total host RAM in bytes.
    pub total_bytes: u64,
    /// Currently available host RAM in bytes.
    pub available_bytes: u64,
    /// Total swap space in bytes.
    pub swap_total_bytes: u64,
    /// Free swap space in bytes.
    pub swap_free_bytes: u64,
    /// Kernel transparent hugepage setting.
    pub thp_mode: ThpMode,
}

impl SysMem {
    /// Reads and parses live system memory information from `/proc/meminfo` and sysfs.
    pub fn read() -> Result<Self, SysMemError> {
        Self::read_from_paths(
            Path::new("/proc/meminfo"),
            Path::new("/sys/kernel/mm/transparent_hugepage/enabled"),
        )
    }

    /// Reads memory statistics from explicit file paths (allows testing with mock files).
    pub fn read_from_paths(meminfo_path: &Path, thp_path: &Path) -> Result<Self, SysMemError> {
        let meminfo_text = fs::read_to_string(meminfo_path).map_err(|e| SysMemError::Io {
            path: meminfo_path.display().to_string(),
            source: e,
        })?;

        let (total_bytes, available_bytes, swap_total_bytes, swap_free_bytes) =
            Self::parse_meminfo(&meminfo_text)?;

        let thp_mode = if thp_path.exists() {
            match fs::read_to_string(thp_path) {
                Ok(content) => Self::parse_thp_enabled(&content),
                Err(_) => ThpMode::Unknown("unreadable".to_string()),
            }
        } else {
            ThpMode::Unknown("not supported".to_string())
        };

        Ok(Self {
            total_bytes,
            available_bytes,
            swap_total_bytes,
            swap_free_bytes,
            thp_mode,
        })
    }

    /// Parses `/proc/meminfo` content into memory counts (in bytes).
    pub fn parse_meminfo(content: &str) -> Result<(u64, u64, u64, u64), SysMemError> {
        let mut total_kb = None;
        let mut available_kb = None;
        let mut swap_total_kb = None;
        let mut swap_free_kb = None;

        for line in content.lines() {
            let mut parts = line.split_whitespace();
            let key = match parts.next() {
                Some(k) => k.trim_end_matches(':'),
                None => continue,
            };
            let value_str = match parts.next() {
                Some(v) => v,
                None => continue,
            };

            let parse_val = |field: &'static str| -> Result<u64, SysMemError> {
                value_str
                    .parse::<u64>()
                    .map_err(|_| SysMemError::InvalidValue(field, value_str.to_string()))
            };

            match key {
                "MemTotal" => total_kb = Some(parse_val("MemTotal")?),
                "MemAvailable" => available_kb = Some(parse_val("MemAvailable")?),
                "SwapTotal" => swap_total_kb = Some(parse_val("SwapTotal")?),
                "SwapFree" => swap_free_kb = Some(parse_val("SwapFree")?),
                _ => {}
            }
        }

        let total_bytes = total_kb
            .ok_or(SysMemError::MissingField("MemTotal"))?
            .saturating_mul(1024);
        let available_bytes = available_kb
            .ok_or(SysMemError::MissingField("MemAvailable"))?
            .saturating_mul(1024);
        let swap_total_bytes = swap_total_kb.unwrap_or(0).saturating_mul(1024);
        let swap_free_bytes = swap_free_kb.unwrap_or(0).saturating_mul(1024);

        Ok((
            total_bytes,
            available_bytes,
            swap_total_bytes,
            swap_free_bytes,
        ))
    }

    /// Parses bracketed THP selection e.g. `[always] madvise never` or `always [madvise] never`.
    pub fn parse_thp_enabled(content: &str) -> ThpMode {
        for token in content.split_whitespace() {
            if token.starts_with('[') && token.ends_with(']') {
                let inner = &token[1..token.len() - 1];
                return match inner {
                    "always" => ThpMode::Always,
                    "madvise" => ThpMode::Madvise,
                    "never" => ThpMode::Never,
                    other => ThpMode::Unknown(other.to_string()),
                };
            }
        }
        ThpMode::Unknown(content.trim().to_string())
    }

    /// Returns whether the host has sufficient headroom for the 5.2 GB engine ceiling.
    pub fn fits_5_2gb_budget(&self) -> bool {
        self.total_bytes >= MAX_RAM_BUDGET_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_meminfo_valid() {
        let sample = "\
MemTotal:       16125304 kB
MemFree:         2368848 kB
MemAvailable:   10257408 kB
Buffers:          822684 kB
Cached:          7357120 kB
SwapTotal:       2097148 kB
SwapFree:        2097148 kB
";
        let (total, avail, swap_total, swap_free) = SysMem::parse_meminfo(sample).unwrap();
        assert_eq!(total, 16125304 * 1024);
        assert_eq!(avail, 10257408 * 1024);
        assert_eq!(swap_total, 2097148 * 1024);
        assert_eq!(swap_free, 2097148 * 1024);
    }

    #[test]
    fn test_parse_thp_modes() {
        assert_eq!(
            SysMem::parse_thp_enabled("[always] madvise never\n"),
            ThpMode::Always
        );
        assert_eq!(
            SysMem::parse_thp_enabled("always [madvise] never\n"),
            ThpMode::Madvise
        );
        assert_eq!(
            SysMem::parse_thp_enabled("always madvise [never]\n"),
            ThpMode::Never
        );
        assert_eq!(
            SysMem::parse_thp_enabled("something strange"),
            ThpMode::Unknown("something strange".to_string())
        );
    }

    #[test]
    fn test_live_sysmem() {
        let mem = SysMem::read().expect("live sysmem read must succeed on Linux host");
        assert!(mem.total_bytes > 0);
        assert!(mem.available_bytes > 0);
        assert!(mem.fits_5_2gb_budget());
    }
}
