//! Local model store and storage budget tracker for Súlimo.
//!
//! Enforces the 30.0 GB model storage quota (`MAX_MODELS_STORAGE_BUDGET_BYTES`),
//! discovers installed `.gguf` files, and safely manages deletion and metadata linkage.

use std::ffi::CString;
use std::fs;
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub use crate::model_manager::registry::MAX_MODELS_STORAGE_BUDGET_BYTES;
use crate::model_manager::registry::{default_registry, ModelMetadata};

/// Minimum free filesystem headroom required before starting a download (500 MiB).
pub const DISK_SAFETY_HEADROOM_BYTES: u64 = 500 * 1024 * 1024;

/// Errors returned by [`ModelStore`] operations.
#[derive(Error, Debug)]
pub enum ModelStoreError {
    #[error("I/O error in model store: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to query filesystem stats: {0}")]
    FsStats(String),

    #[error("Model not found in store: {0}")]
    NotFound(String),
}

/// Represents an installed model found on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledModel {
    /// Identifier: either the registry model ID if matched, or the filename without `.gguf`.
    pub id: String,
    /// Exact filename on disk (e.g. `qwen2.5-1.5b-instruct-q4_0.gguf`).
    pub filename: String,
    /// Absolute path to the model file on disk.
    pub path: PathBuf,
    /// Exact byte size on disk.
    pub size_bytes: u64,
    /// Whether this model matches an entry in Súlimo's curated registry.
    pub is_registered: bool,
    /// Attached curated metadata if registered.
    pub metadata: Option<ModelMetadata>,
}

/// Local model storage repository and quota manager.
#[derive(Debug, Clone)]
pub struct ModelStore {
    models_dir: PathBuf,
}

impl ModelStore {
    /// Initializes the model store, creating the destination directory if it does not exist.
    pub fn new(custom_dir: Option<PathBuf>) -> Result<Self, ModelStoreError> {
        let models_dir = custom_dir.unwrap_or_else(Self::default_models_dir);
        if !models_dir.exists() {
            fs::create_dir_all(&models_dir)?;
        }
        Ok(Self { models_dir })
    }

    /// Returns the resolved directory where models are stored.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Returns the default models directory path.
    ///
    /// Respects `SULIMO_MODELS_DIR` environment variable, falling back to
    /// `~/.local/share/sulimo/models`.
    pub fn default_models_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("SULIMO_MODELS_DIR") {
            PathBuf::from(dir)
        } else if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".local/share/sulimo/models")
        } else {
            PathBuf::from(".sulimo/models")
        }
    }

    /// Scans the model directory and returns all installed `.gguf` models,
    /// cross-referencing them against the curated registry.
    pub fn list_installed_models(&self) -> Result<Vec<InstalledModel>, ModelStoreError> {
        let mut installed = Vec::new();
        if !self.models_dir.exists() {
            return Ok(installed);
        }

        let registry = default_registry();

        for entry in fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let filename = match path.file_name().and_then(|s| s.to_str()) {
                Some(name) => name.to_string(),
                None => continue,
            };

            // Only inspect .gguf files (ignore temporary .part files)
            if !filename.ends_with(".gguf") {
                continue;
            }

            let metadata_res = fs::metadata(&path)?;
            let size_bytes = metadata_res.len();

            // Match against registry (case-insensitive filename or ID stem match)
            let matched_meta = registry.iter().find(|m| {
                m.filename.eq_ignore_ascii_case(&filename)
                    || path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|stem| stem.eq_ignore_ascii_case(&m.id))
                        .unwrap_or(false)
                    || path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|stem| stem.eq_ignore_ascii_case(m.filename.trim_end_matches(".gguf")))
                        .unwrap_or(false)
            });

            let (id, is_registered, metadata) = match matched_meta {
                Some(m) => (m.id.clone(), true, Some(m.clone())),
                None => {
                    let fallback_id = filename.trim_end_matches(".gguf").to_string();
                    (fallback_id, false, None)
                }
            };

            installed.push(InstalledModel {
                id,
                filename,
                path,
                size_bytes,
                is_registered,
                metadata,
            });
        }

        // Sort by id for deterministic ordering
        installed.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(installed)
    }

    /// Calculates total storage occupied by completed `.gguf` models in bytes.
    pub fn total_installed_bytes(&self) -> Result<u64, ModelStoreError> {
        let models = self.list_installed_models()?;
        let total: u64 = models.iter().map(|m| m.size_bytes).sum();
        Ok(total)
    }

    /// Calculates total storage occupied by completed `.gguf` models AND in-progress `.part` files.
    pub fn total_storage_used_bytes(&self) -> Result<u64, ModelStoreError> {
        if !self.models_dir.exists() {
            return Ok(0);
        }

        let mut total: u64 = 0;
        for entry in fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name.ends_with(".gguf") || name.ends_with(".part") {
                    total += fs::metadata(&path)?.len();
                }
            }
        }
        Ok(total)
    }

    /// Returns remaining available quota within the 30.0 GB model storage budget.
    pub fn remaining_quota_bytes(&self) -> Result<u64, ModelStoreError> {
        let used = self.total_storage_used_bytes()?;
        Ok(MAX_MODELS_STORAGE_BUDGET_BYTES.saturating_sub(used))
    }

    /// Locates an installed model by its registry ID or filename.
    pub fn get_model_path(&self, id_or_filename: &str) -> Option<PathBuf> {
        let direct_path = self.models_dir.join(id_or_filename);
        if direct_path.is_file() {
            return Some(direct_path);
        }

        // Try adding .gguf if missing
        if !id_or_filename.ends_with(".gguf") {
            let with_ext = self.models_dir.join(format!("{id_or_filename}.gguf"));
            if with_ext.is_file() {
                return Some(with_ext);
            }
        }

        // Search registry for matching ID or filename to resolve exact filename
        if let Some(meta) = default_registry().iter().find(|m| {
            m.id.eq_ignore_ascii_case(id_or_filename)
                || m.filename.eq_ignore_ascii_case(id_or_filename)
                || m.filename
                    .trim_end_matches(".gguf")
                    .eq_ignore_ascii_case(id_or_filename.trim_end_matches(".gguf"))
        }) {
            let meta_path = self.models_dir.join(&meta.filename);
            if meta_path.is_file() {
                return Some(meta_path);
            }
        }

        // Case-insensitive directory scan fallback
        if let Ok(entries) = fs::read_dir(&self.models_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.eq_ignore_ascii_case(id_or_filename)
                        || (!id_or_filename.ends_with(".gguf")
                            && name.eq_ignore_ascii_case(&format!("{id_or_filename}.gguf")))
                    {
                        return Some(entry.path());
                    }
                }
            }
        }

        None
    }

    /// Checks if a model is installed on disk.
    pub fn is_model_installed(&self, id_or_filename: &str) -> bool {
        self.get_model_path(id_or_filename).is_some()
    }

    /// Deletes an installed model and any corresponding `.part` temporary file.
    pub fn delete_model(&self, id_or_filename: &str) -> Result<bool, ModelStoreError> {
        let mut deleted = false;

        if let Some(path) = self.get_model_path(id_or_filename) {
            fs::remove_file(&path)?;
            deleted = true;

            // Also check and delete matching .part file if present
            let part_path = path.with_extension("gguf.part");
            if part_path.exists() {
                let _ = fs::remove_file(part_path);
            }
        }

        Ok(deleted)
    }

    /// Verifies the SHA-256 integrity of an installed model against the curated registry.
    pub fn verify_installed_model(&self, id_or_filename: &str) -> Result<bool, ModelStoreError> {
        let path = match self.get_model_path(id_or_filename) {
            Some(p) => p,
            None => return Err(ModelStoreError::NotFound(id_or_filename.to_string())),
        };

        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let meta = default_registry()
            .iter()
            .find(|m| m.filename == filename || m.id == id_or_filename)
            .cloned();

        let expected_sha256 = match meta {
            Some(m) => m.sha256,
            None => {
                // If not in registry, verification against known digest cannot be performed
                return Ok(true);
            }
        };

        let mut file = fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let actual_sha256 = format!("{:x}", hasher.finalize());
        Ok(actual_sha256.eq_ignore_ascii_case(&expected_sha256))
    }
}

/// Inspects the underlying filesystem using `statvfs` to determine available unprivileged bytes.
pub fn get_filesystem_free_bytes(path: &Path) -> Result<u64, ModelStoreError> {
    let mut check_path = path.to_path_buf();
    while !check_path.exists() {
        if let Some(parent) = check_path.parent() {
            check_path = parent.to_path_buf();
        } else {
            break;
        }
    }

    let c_path = CString::new(check_path.as_os_str().as_bytes())
        .map_err(|e| ModelStoreError::FsStats(format!("Invalid path string: {e}")))?;

    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if ret != 0 {
        return Err(ModelStoreError::FsStats(
            std::io::Error::last_os_error().to_string(),
        ));
    }

    let free_bytes = (stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64);
    Ok(free_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct TempTestDir(PathBuf);
    impl TempTestDir {
        fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("sulimo_test_{name}_{}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TempTestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn test_store_lifecycle_and_quota() {
        let temp_dir = TempTestDir::new("store_lifecycle");
        let store = ModelStore::new(Some(temp_dir.0.clone())).unwrap();

        assert_eq!(store.models_dir(), &temp_dir.0);
        let list = store.list_installed_models().unwrap();
        assert!(list.is_empty());
        assert_eq!(store.total_installed_bytes().unwrap(), 0);
        assert_eq!(
            store.remaining_quota_bytes().unwrap(),
            MAX_MODELS_STORAGE_BUDGET_BYTES
        );

        // Create a dummy model file matching registry
        let dummy_path = temp_dir.0.join("LFM2.5-230M-Q4_0.gguf");
        let payload = b"sulimo_dummy_payload_for_testing";
        let mut f = fs::File::create(&dummy_path).unwrap();
        f.write_all(payload).unwrap();
        drop(f);

        let list_after = store.list_installed_models().unwrap();
        assert_eq!(list_after.len(), 1);
        let installed = &list_after[0];
        assert_eq!(installed.id, "lfm2.5-230m");
        assert_eq!(installed.filename, "LFM2.5-230M-Q4_0.gguf");
        assert_eq!(installed.size_bytes, payload.len() as u64);
        assert!(installed.is_registered);
        assert!(installed.metadata.is_some());

        assert_eq!(store.total_installed_bytes().unwrap(), payload.len() as u64);
        assert_eq!(
            store.remaining_quota_bytes().unwrap(),
            MAX_MODELS_STORAGE_BUDGET_BYTES - payload.len() as u64
        );

        // Find model
        assert_eq!(
            store.get_model_path("lfm2.5-230m"),
            Some(dummy_path.clone())
        );
        assert_eq!(
            store.get_model_path("LFM2.5-230M-Q4_0.gguf"),
            Some(dummy_path.clone())
        );
        assert_eq!(
            store.get_model_path("lfm2.5-230m-q4_0.gguf"),
            Some(dummy_path.clone())
        );
        assert!(store.is_model_installed("lfm2.5-230m"));

        // Delete model
        assert!(store.delete_model("lfm2.5-230m").unwrap());
        assert!(!store.is_model_installed("lfm2.5-230m"));
        assert_eq!(store.total_installed_bytes().unwrap(), 0);
    }

    #[test]
    fn test_filesystem_free_bytes() {
        let free = get_filesystem_free_bytes(Path::new("/tmp")).unwrap();
        assert!(free > 0, "Filesystem free bytes should be positive");
    }
}
