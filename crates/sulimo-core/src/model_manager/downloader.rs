//! Resumable streaming model downloader with SHA-256 integrity verification.
//!
//! Features:
//! - Pre-flight disk space check via `statvfs` (requires `file_size + 500 MiB` free headroom).
//! - Model storage pool quota enforcement (`MAX_MODELS_STORAGE_BUDGET_BYTES` = 30.0 GB).
//! - Resumable HTTP downloads writing to `<filename>.part` using HTTP `Range: bytes=offset-`.
//! - Streaming SHA-256 calculation across the entire file during transit (O(1) RAM).
//! - Real-time progress, transfer speed, and ETA tracking.
//! - Cooperative cancellation via `Arc<AtomicBool>`.
//! - Atomic rename from `.part` to final `.gguf` strictly after SHA-256 match.

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::model_manager::registry::{find_model_by_id, ModelMetadata};
use crate::model_manager::store::{
    get_filesystem_free_bytes, ModelStore, DISK_SAFETY_HEADROOM_BYTES,
    MAX_MODELS_STORAGE_BUDGET_BYTES,
};

/// Lifecycle state of a model download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadState {
    /// Download initialized, pre-flight checks passing.
    Pending,
    /// Actively receiving streaming bytes from network.
    Downloading,
    /// Download finished, validating complete SHA-256 digest against registry.
    VerifyingChecksum,
    /// SHA-256 verified; atomically renamed to final `.gguf`.
    Completed,
    /// Cancelled by user or application shutdown.
    Cancelled,
    /// Download or verification failed.
    Failed,
}

/// Progress telemetry emitted during download.
#[derive(Debug, Clone, PartialEq)]
pub struct DownloadProgress {
    /// Number of bytes downloaded so far.
    pub downloaded_bytes: u64,
    /// Total expected size of the model in bytes.
    pub total_bytes: u64,
    /// Current transfer speed in bytes per second.
    pub speed_bps: f64,
    /// Estimated time remaining in seconds (if speed > 0).
    pub eta_seconds: Option<u64>,
    /// Progress fraction between 0.0 and 1.0.
    pub percent: f32,
    /// Current lifecycle state.
    pub state: DownloadState,
}

/// Errors returned by the model downloader.
#[derive(Error, Debug)]
pub enum DownloadError {
    #[error("Model not found in registry: {0}")]
    ModelNotFound(String),

    #[error("Insufficient disk space: {available} bytes available, required {required} bytes")]
    InsufficientDiskSpace { available: u64, required: u64 },

    #[error("Model storage quota exceeded: current used {current_used} bytes, model requires {needed} bytes, max quota {max_quota} bytes")]
    StorageQuotaExceeded {
        current_used: u64,
        needed: u64,
        max_quota: u64,
    },

    #[error("Model already installed at: {0}")]
    AlreadyInstalled(PathBuf),

    #[error("Network error: {0}")]
    Network(String),

    #[error("I/O error during download: {0}")]
    Io(#[from] std::io::Error),

    #[error("SHA-256 checksum mismatch for {model_id}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        model_id: String,
        expected: String,
        actual: String,
    },

    #[error("Download was cancelled")]
    Cancelled,

    #[error("Server returned HTTP error status: {0}")]
    HttpStatus(u16),

    #[error("Model store error: {0}")]
    Store(#[from] crate::model_manager::store::ModelStoreError),
}

/// Downloader client orchestrating streaming transfers, resumption, and verification.
pub struct ModelDownloader {
    store: ModelStore,
    client: ureq::Agent,
}

impl ModelDownloader {
    /// Creates a new downloader associated with the specified model store.
    pub fn new(store: ModelStore) -> Self {
        let client = ureq::builder()
            .timeout_connect(Duration::from_secs(15))
            .timeout_read(Duration::from_secs(30))
            .build();
        Self { store, client }
    }

    /// Returns a reference to the underlying model store.
    pub fn store(&self) -> &ModelStore {
        &self.store
    }

    /// Downloads a model by its registry ID.
    pub fn download_by_id<F>(
        &self,
        model_id: &str,
        cancel_flag: Arc<AtomicBool>,
        progress_cb: F,
    ) -> Result<PathBuf, DownloadError>
    where
        F: Fn(DownloadProgress) + Send + Sync + 'static,
    {
        let metadata = find_model_by_id(model_id)
            .ok_or_else(|| DownloadError::ModelNotFound(model_id.to_string()))?;
        self.download(&metadata, cancel_flag, progress_cb)
    }

    /// Downloads a model according to its metadata descriptor.
    pub fn download<F>(
        &self,
        meta: &ModelMetadata,
        cancel_flag: Arc<AtomicBool>,
        progress_cb: F,
    ) -> Result<PathBuf, DownloadError>
    where
        F: Fn(DownloadProgress) + Send + Sync + 'static,
    {
        let final_path = self.store.models_dir().join(&meta.filename);
        let part_path = self
            .store
            .models_dir()
            .join(format!("{}.part", meta.filename));

        // 1. Check if final file already exists and is complete
        if final_path.is_file() {
            let existing_final_len = fs::metadata(&final_path)?.len();
            if existing_final_len == meta.file_size_bytes {
                let p = DownloadProgress {
                    downloaded_bytes: meta.file_size_bytes,
                    total_bytes: meta.file_size_bytes,
                    speed_bps: 0.0,
                    eta_seconds: Some(0),
                    percent: 1.0,
                    state: DownloadState::Completed,
                };
                progress_cb(p);
                return Err(DownloadError::AlreadyInstalled(final_path));
            }
        }

        // 2. Pre-flight check: Dedicated 30.0 GB storage pool quota
        let current_used = self.store.total_storage_used_bytes()?;
        let existing_part_size = if part_path.exists() {
            fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        // Only additional bytes to be written count towards the quota limit
        let needed_bytes = meta.file_size_bytes.saturating_sub(existing_part_size);
        if current_used.saturating_add(needed_bytes) > MAX_MODELS_STORAGE_BUDGET_BYTES {
            return Err(DownloadError::StorageQuotaExceeded {
                current_used,
                needed: needed_bytes,
                max_quota: MAX_MODELS_STORAGE_BUDGET_BYTES,
            });
        }

        // 3. Pre-flight check: Filesystem available free space
        let free_disk = get_filesystem_free_bytes(self.store.models_dir())?;
        let required_disk = meta
            .file_size_bytes
            .saturating_add(DISK_SAFETY_HEADROOM_BYTES);
        if free_disk < required_disk {
            return Err(DownloadError::InsufficientDiskSpace {
                available: free_disk,
                required: required_disk,
            });
        }

        // 4. Initial progress notification
        progress_cb(DownloadProgress {
            downloaded_bytes: existing_part_size,
            total_bytes: meta.file_size_bytes,
            speed_bps: 0.0,
            eta_seconds: None,
            percent: (existing_part_size as f32 / meta.file_size_bytes as f32).clamp(0.0, 1.0),
            state: DownloadState::Pending,
        });

        // 5. Setup resumption & SHA-256 hasher
        let (mut file, mut hasher, mut downloaded_bytes) = if existing_part_size > 0
            && existing_part_size < meta.file_size_bytes
        {
            // Attempt HTTP range resumption
            let req = self
                .client
                .get(&meta.download_url)
                .set("User-Agent", "Sulimo/0.2.0")
                .set("Range", &format!("bytes={existing_part_size}-"));

            match req.call() {
                Ok(resp) if resp.status() == 206 => {
                    // Server supports HTTP Range (206 Partial Content)
                    // Hash existing bytes already on disk
                    let mut pre_hasher = Sha256::new();
                    let mut existing_file = fs::File::open(&part_path)?;
                    let mut buf = [0u8; 64 * 1024];
                    let mut read_total = 0;
                    while read_total < existing_part_size {
                        let to_read =
                            (existing_part_size - read_total).min(buf.len() as u64) as usize;
                        let n = existing_file.read(&mut buf[..to_read])?;
                        if n == 0 {
                            break;
                        }
                        pre_hasher.update(&buf[..n]);
                        read_total += n as u64;
                    }

                    let file = fs::OpenOptions::new().append(true).open(&part_path)?;

                    (file, pre_hasher, existing_part_size)
                }
                Ok(resp) if resp.status() == 200 => {
                    // Server ignored Range header and returned full payload from start
                    let file = fs::File::create(&part_path)?;
                    (file, Sha256::new(), 0)
                }
                Ok(resp) => return Err(DownloadError::HttpStatus(resp.status())),
                Err(ureq::Error::Status(code, _)) => return Err(DownloadError::HttpStatus(code)),
                Err(ureq::Error::Transport(e)) => {
                    return Err(DownloadError::Network(e.to_string()))
                }
            }
        } else {
            // Fresh download from byte 0
            let file = fs::File::create(&part_path)?;
            (file, Sha256::new(), 0)
        };

        // 6. Execute streaming network read loop
        let req = if downloaded_bytes > 0 {
            self.client
                .get(&meta.download_url)
                .set("User-Agent", "Sulimo/0.2.0")
                .set("Range", &format!("bytes={downloaded_bytes}-"))
        } else {
            self.client
                .get(&meta.download_url)
                .set("User-Agent", "Sulimo/0.2.0")
        };

        let resp = match req.call() {
            Ok(r) => r,
            Err(ureq::Error::Status(code, _)) => return Err(DownloadError::HttpStatus(code)),
            Err(ureq::Error::Transport(e)) => return Err(DownloadError::Network(e.to_string())),
        };

        let mut reader = resp.into_reader();
        let mut buffer = [0u8; 64 * 1024];

        let start_time = Instant::now();
        let mut last_progress_time = Instant::now();
        let mut bytes_since_last_progress = 0u64;
        let mut smoothed_speed_bps = 0.0f64;

        // Emit downloading state at stream initiation
        progress_cb(DownloadProgress {
            downloaded_bytes,
            total_bytes: meta.file_size_bytes,
            speed_bps: 0.0,
            eta_seconds: None,
            percent: (downloaded_bytes as f32 / meta.file_size_bytes as f32).clamp(0.0, 1.0),
            state: DownloadState::Downloading,
        });

        loop {
            // Check cooperative cancellation flag
            if cancel_flag.load(Ordering::Relaxed) {
                progress_cb(DownloadProgress {
                    downloaded_bytes,
                    total_bytes: meta.file_size_bytes,
                    speed_bps: 0.0,
                    eta_seconds: None,
                    percent: (downloaded_bytes as f32 / meta.file_size_bytes as f32)
                        .clamp(0.0, 1.0),
                    state: DownloadState::Cancelled,
                });
                return Err(DownloadError::Cancelled);
            }

            let bytes_read = match reader.read(&mut buffer) {
                Ok(0) => break, // EOF reached
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(DownloadError::Io(e)),
            };

            // Write chunk to file and update streaming hash
            file.write_all(&buffer[..bytes_read])?;
            hasher.update(&buffer[..bytes_read]);

            downloaded_bytes += bytes_read as u64;
            bytes_since_last_progress += bytes_read as u64;

            // Throttled progress callback (~100ms interval)
            let now = Instant::now();
            let elapsed_since_progress = now.duration_since(last_progress_time);
            if elapsed_since_progress >= Duration::from_millis(100) {
                let instant_speed = (bytes_since_last_progress as f64)
                    / elapsed_since_progress.as_secs_f64().max(0.001);

                // Exponential moving average for smooth display
                smoothed_speed_bps = if smoothed_speed_bps == 0.0 {
                    instant_speed
                } else {
                    0.7 * smoothed_speed_bps + 0.3 * instant_speed
                };

                let remaining_bytes = meta.file_size_bytes.saturating_sub(downloaded_bytes);
                let eta_seconds = if smoothed_speed_bps > 1000.0 {
                    Some((remaining_bytes as f64 / smoothed_speed_bps) as u64)
                } else {
                    None
                };

                progress_cb(DownloadProgress {
                    downloaded_bytes,
                    total_bytes: meta.file_size_bytes,
                    speed_bps: smoothed_speed_bps,
                    eta_seconds,
                    percent: (downloaded_bytes as f32 / meta.file_size_bytes as f32)
                        .clamp(0.0, 1.0),
                    state: DownloadState::Downloading,
                });

                last_progress_time = now;
                bytes_since_last_progress = 0;
            }
        }

        // 7. Flush buffers to disk
        file.sync_all()?;
        drop(file);

        // 8. Transition to verification
        progress_cb(DownloadProgress {
            downloaded_bytes,
            total_bytes: meta.file_size_bytes,
            speed_bps: 0.0,
            eta_seconds: Some(0),
            percent: 1.0,
            state: DownloadState::VerifyingChecksum,
        });

        // 9. Validate SHA-256 digest
        let actual_sha256 = format!("{:x}", hasher.finalize());
        if !actual_sha256.eq_ignore_ascii_case(&meta.sha256) {
            progress_cb(DownloadProgress {
                downloaded_bytes,
                total_bytes: meta.file_size_bytes,
                speed_bps: 0.0,
                eta_seconds: None,
                percent: 1.0,
                state: DownloadState::Failed,
            });
            return Err(DownloadError::ChecksumMismatch {
                model_id: meta.id.clone(),
                expected: meta.sha256.clone(),
                actual: actual_sha256,
            });
        }

        // 10. Atomic rename: strictly after SHA-256 match
        fs::rename(&part_path, &final_path)?;

        let final_elapsed = start_time.elapsed().as_secs_f64().max(0.001);
        let overall_avg_speed = (downloaded_bytes as f64) / final_elapsed;

        progress_cb(DownloadProgress {
            downloaded_bytes: meta.file_size_bytes,
            total_bytes: meta.file_size_bytes,
            speed_bps: overall_avg_speed,
            eta_seconds: Some(0),
            percent: 1.0,
            state: DownloadState::Completed,
        });

        Ok(final_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    struct TempTestDir(PathBuf);
    impl TempTestDir {
        fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("sulimo_test_dl_{name}_{}", std::process::id()));
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

    /// Spawns a lightweight HTTP test server on an ephemeral port.
    fn spawn_mock_server(
        payload: &'static [u8],
        hit_counter: Arc<AtomicUsize>,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let addr = format!("http://127.0.0.1:{port}");

        let handle = thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => break,
                };
                hit_counter.fetch_add(1, Ordering::SeqCst);

                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut range_header = None;

                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if line == "\r\n" || line == "\n" {
                        break;
                    }
                    if line.to_lowercase().starts_with("range:") {
                        range_header = Some(line.trim().to_string());
                    }
                    line.clear();
                }

                if let Some(ref range) = range_header {
                    // Range: bytes=X-
                    let parts: Vec<&str> = range.split('=').collect();
                    if parts.len() == 2 {
                        let offset_str = parts[1].trim_end_matches('-');
                        if let Ok(offset) = offset_str.parse::<usize>() {
                            if offset < payload.len() {
                                let slice = &payload[offset..];
                                let response = format!(
                                    "HTTP/1.1 206 Partial Content\r\n\
                                     Content-Length: {}\r\n\
                                     Content-Range: bytes {}-{}/{}\r\n\
                                     Connection: close\r\n\r\n",
                                    slice.len(),
                                    offset,
                                    payload.len() - 1,
                                    payload.len()
                                );
                                let _ = stream.write_all(response.as_bytes());
                                let _ = stream.write_all(slice);
                                continue;
                            }
                        }
                    }
                }

                // Standard 200 OK
                let response = format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\r\n",
                    payload.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.write_all(payload);
            }
        });

        (addr, handle)
    }

    #[test]
    fn test_full_download_sha256_match_and_rename() {
        let temp_dir = TempTestDir::new("full_download");
        let store = ModelStore::new(Some(temp_dir.0.clone())).unwrap();
        let downloader = ModelDownloader::new(store);

        static PAYLOAD: &[u8] = b"sulimo_streaming_test_model_payload_1234567890";
        let payload_hash = format!("{:x}", Sha256::digest(PAYLOAD));

        let hit_counter = Arc::new(AtomicUsize::new(0));
        let (server_url, _handle) = spawn_mock_server(PAYLOAD, hit_counter);

        let meta = ModelMetadata {
            id: "test-model".to_string(),
            display_name: "Test Model".to_string(),
            family: "Test".to_string(),
            architecture: crate::model_config::ModelArch::Llama,
            param_count: "1B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: format!("{server_url}/model.gguf"),
            filename: "test-model-q4_0.gguf".to_string(),
            file_size_bytes: PAYLOAD.len() as u64,
            sha256: payload_hash,
            default_context_length: 2048,
            estimated_ram_bytes_4k: 1024,
            prompt_template: crate::model_manager::registry::PromptTemplateType::ChatMl,
            description: "Test".to_string(),
        };

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let progress_events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let p_events_clone = progress_events.clone();

        let final_path = downloader
            .download(&meta, cancel_flag, move |p| {
                p_events_clone.lock().unwrap().push(p.state);
            })
            .expect("Download should succeed");

        assert!(final_path.is_file());
        assert_eq!(fs::read(&final_path).unwrap(), PAYLOAD);
        assert!(!temp_dir.0.join("test-model-q4_0.gguf.part").exists());

        let events = progress_events.lock().unwrap();
        assert!(events.contains(&DownloadState::Downloading));
        assert!(events.contains(&DownloadState::VerifyingChecksum));
        assert_eq!(events.last(), Some(&DownloadState::Completed));
    }

    #[test]
    fn test_corrupted_payload_checksum_mismatch_rejected() {
        let temp_dir = TempTestDir::new("corrupt_download");
        let store = ModelStore::new(Some(temp_dir.0.clone())).unwrap();
        let downloader = ModelDownloader::new(store);

        static CORRUPT_PAYLOAD: &[u8] = b"corrupted_bad_payload";
        let hit_counter = Arc::new(AtomicUsize::new(0));
        let (server_url, _handle) = spawn_mock_server(CORRUPT_PAYLOAD, hit_counter);

        let meta = ModelMetadata {
            id: "corrupt-model".to_string(),
            display_name: "Corrupt Model".to_string(),
            family: "Test".to_string(),
            architecture: crate::model_config::ModelArch::Llama,
            param_count: "1B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: format!("{server_url}/corrupt.gguf"),
            filename: "corrupt-model-q4_0.gguf".to_string(),
            file_size_bytes: CORRUPT_PAYLOAD.len() as u64,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            default_context_length: 2048,
            estimated_ram_bytes_4k: 1024,
            prompt_template: crate::model_manager::registry::PromptTemplateType::ChatMl,
            description: "Corrupt Test".to_string(),
        };

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let err = downloader
            .download(&meta, cancel_flag, |_| {})
            .expect_err("Should fail with checksum mismatch");

        match err {
            DownloadError::ChecksumMismatch { model_id, .. } => {
                assert_eq!(model_id, "corrupt-model");
            }
            other => panic!("Unexpected error: {:?}", other),
        }

        // Final .gguf must NOT exist
        assert!(!temp_dir.0.join("corrupt-model-q4_0.gguf").exists());
    }

    #[test]
    fn test_resumable_download() {
        let temp_dir = TempTestDir::new("resume_download");
        let store = ModelStore::new(Some(temp_dir.0.clone())).unwrap();
        let downloader = ModelDownloader::new(store);

        static FULL_PAYLOAD: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
        let full_hash = format!("{:x}", Sha256::digest(FULL_PAYLOAD));

        let hit_counter = Arc::new(AtomicUsize::new(0));
        let (server_url, _handle) = spawn_mock_server(FULL_PAYLOAD, hit_counter);

        let filename = "resumable-model.gguf";
        let part_path = temp_dir.0.join(format!("{filename}.part"));

        // Simulate a pre-existing partial download of the first 10 bytes
        let first_half = &FULL_PAYLOAD[..10];
        fs::write(&part_path, first_half).unwrap();

        let meta = ModelMetadata {
            id: "resumable-model".to_string(),
            display_name: "Resumable Model".to_string(),
            family: "Test".to_string(),
            architecture: crate::model_config::ModelArch::Llama,
            param_count: "1B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: format!("{server_url}/resume.gguf"),
            filename: filename.to_string(),
            file_size_bytes: FULL_PAYLOAD.len() as u64,
            sha256: full_hash,
            default_context_length: 2048,
            estimated_ram_bytes_4k: 1024,
            prompt_template: crate::model_manager::registry::PromptTemplateType::ChatMl,
            description: "Resumable Test".to_string(),
        };

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let final_path = downloader
            .download(&meta, cancel_flag, |_| {})
            .expect("Resumed download should succeed");

        assert_eq!(fs::read(&final_path).unwrap(), FULL_PAYLOAD);
        assert!(!part_path.exists());
    }

    #[test]
    fn test_cooperative_cancellation() {
        let temp_dir = TempTestDir::new("cancel_download");
        let store = ModelStore::new(Some(temp_dir.0.clone())).unwrap();
        let downloader = ModelDownloader::new(store);

        static BIG_PAYLOAD: &[u8] = &[0xAA; 1024 * 1024]; // 1 MiB
        let hit_counter = Arc::new(AtomicUsize::new(0));
        let (server_url, _handle) = spawn_mock_server(BIG_PAYLOAD, hit_counter);

        let meta = ModelMetadata {
            id: "cancel-model".to_string(),
            display_name: "Cancel Model".to_string(),
            family: "Test".to_string(),
            architecture: crate::model_config::ModelArch::Llama,
            param_count: "1B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: format!("{server_url}/cancel.gguf"),
            filename: "cancel-model.gguf".to_string(),
            file_size_bytes: BIG_PAYLOAD.len() as u64,
            sha256: "fake".to_string(),
            default_context_length: 2048,
            estimated_ram_bytes_4k: 1024,
            prompt_template: crate::model_manager::registry::PromptTemplateType::ChatMl,
            description: "Cancel Test".to_string(),
        };

        // Flag pre-cancelled
        let cancel_flag = Arc::new(AtomicBool::new(true));
        let err = downloader
            .download(&meta, cancel_flag, |_| {})
            .expect_err("Should be cancelled immediately");

        match err {
            DownloadError::Cancelled => {}
            other => panic!("Expected Cancelled error, got {:?}", other),
        }
    }

    #[test]
    fn test_quota_pre_flight_rejection() {
        let temp_dir = TempTestDir::new("quota_reject");
        let store = ModelStore::new(Some(temp_dir.0.clone())).unwrap();
        let downloader = ModelDownloader::new(store);

        // Oversized model exceeding 30.0 GB quota
        let meta = ModelMetadata {
            id: "huge-model".to_string(),
            display_name: "Huge Model".to_string(),
            family: "Test".to_string(),
            architecture: crate::model_config::ModelArch::Llama,
            param_count: "70B".to_string(),
            quantization: "Q4_0".to_string(),
            download_url: "http://127.0.0.1/fake".to_string(),
            filename: "huge-model.gguf".to_string(),
            file_size_bytes: 35_000_000_000,
            sha256: "fake".to_string(),
            default_context_length: 2048,
            estimated_ram_bytes_4k: 1024,
            prompt_template: crate::model_manager::registry::PromptTemplateType::ChatMl,
            description: "Quota Test".to_string(),
        };

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let err = downloader
            .download(&meta, cancel_flag, |_| {})
            .expect_err("Should reject due to storage quota");

        match err {
            DownloadError::StorageQuotaExceeded { max_quota, .. } => {
                assert_eq!(max_quota, MAX_MODELS_STORAGE_BUDGET_BYTES);
            }
            other => panic!("Expected StorageQuotaExceeded, got {:?}", other),
        }
    }
}
