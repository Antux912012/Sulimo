//! Súlimo Engine orchestrator (@C:ENG).
//!
//! Bridges memory management, CPU inference, OpenCL execution, and telemetry.

pub mod capability;
pub mod engine;
pub mod forward_pass;
pub mod sampler;

pub use capability::{CapabilityError, CapabilityReport};
pub use engine::{Engine, EngineError, GenerationStats};
pub use forward_pass::{forward_token, ScratchBuffers};
pub use sampler::{sample_token, SamplingParams};
