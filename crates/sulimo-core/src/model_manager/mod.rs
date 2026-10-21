//! Model management subsystem for Súlimo.
//!
//! Provides the curated model registry, download management,
//! storage tracking, and file verification.

pub mod downloader;
pub mod registry;
pub mod store;

pub use downloader::{DownloadError, DownloadProgress, DownloadState, ModelDownloader};
pub use registry::{
    default_registry, find_model_by_id, format_bytes, ModelMetadata, PromptTemplateType,
};
pub use store::{
    get_filesystem_free_bytes, InstalledModel, ModelStore, ModelStoreError,
    DISK_SAFETY_HEADROOM_BYTES, MAX_MODELS_STORAGE_BUDGET_BYTES,
};
