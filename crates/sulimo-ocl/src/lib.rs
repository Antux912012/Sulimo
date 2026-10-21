//! OpenCL runtime, SIMD8 FP16 kernel execution, and zero-spin waiting for Intel Gen8–11 iGPUs (@C:OCL-GEN9).

#[cfg(feature = "opencl")]
pub mod buffer;
#[cfg(feature = "opencl")]
pub mod gemm;
#[cfg(feature = "opencl")]
pub mod prefill;
pub mod probe;
#[cfg(feature = "opencl")]
pub mod program;
#[cfg(feature = "opencl")]
pub mod waiter;

#[cfg(feature = "opencl")]
pub use buffer::{active_cl_mem_count, OclBuffer};
#[cfg(feature = "opencl")]
pub use gemm::{gemm_q4_0_f16, gemm_q4_0_f16_scalar_reference, GemmProfile};
#[cfg(feature = "opencl")]
pub use prefill::{
    enqueue_add, enqueue_attention_prefill, enqueue_gemm_q4_0_f16, enqueue_kv_store,
    enqueue_logits_final, enqueue_rmsnorm, enqueue_rope, enqueue_silu_mul, execute_prefill_ubatch,
    OclKvCache, OclLayerTensors, OclModelTensors, OclPrefillScratch,
};
pub use probe::{probe, OclDeviceInfo, OclProbeResult};
#[cfg(feature = "opencl")]
pub use program::{KernelVariant, OclContext, OclError};
#[cfg(feature = "opencl")]
pub use waiter::wait_event_zero_spin;
