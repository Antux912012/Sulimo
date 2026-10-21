//! Model management subsystem for Súlimo.
//!
//! Provides the curated model registry, download management,
//! storage tracking, and file verification.

pub mod registry;

pub use registry::{
    default_registry, find_model_by_id, format_bytes, ModelMetadata, PromptTemplateType,
};
