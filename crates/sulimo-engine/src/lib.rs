//! Súlimo Engine orchestrator (@C:ENG).
//!
//! Bridges memory management, CPU inference, OpenCL execution, and telemetry.

pub mod bench;
pub mod capability;
pub mod engine;
pub mod forward_pass;
pub mod sampler;
pub mod scheduler;

pub use bench::{run_benchmark_suite, BenchmarkReport};
pub use capability::{CapabilityError, CapabilityReport};
pub use engine::{Engine, EngineError, GenerationStats};
pub use forward_pass::{forward_token, ScratchBuffers};
pub use sampler::{sample_token, SamplingParams};
#[cfg(feature = "opencl")]
pub use scheduler::OclEngineState;
pub use scheduler::{repack_context, PrefillBackend, SchedulerConfig};
