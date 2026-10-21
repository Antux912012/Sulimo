//! OpenCL runtime, SIMD8 FP16 kernel execution, and zero-spin waiting for Intel Gen8–11 iGPUs (@C:OCL-GEN9).

#[cfg(feature = "opencl")]
pub mod buffer;
#[cfg(feature = "opencl")]
pub mod gemm;
pub mod probe;
#[cfg(feature = "opencl")]
pub mod program;
#[cfg(feature = "opencl")]
pub mod waiter;

#[cfg(feature = "opencl")]
pub use buffer::OclBuffer;
#[cfg(feature = "opencl")]
pub use gemm::{gemm_q4_0_f16, gemm_q4_0_f16_scalar_reference, GemmProfile};
pub use probe::{probe, OclDeviceInfo, OclProbeResult};
#[cfg(feature = "opencl")]
pub use program::{KernelVariant, OclContext, OclError};
#[cfg(feature = "opencl")]
pub use waiter::wait_event_zero_spin;
