//! Súlimo CPU backend (@C:CPU).
//!
//! Provides topology detection, physical-core pinning, and AVX2/FMA execution primitives.

pub mod avx2;
pub mod scalar;
pub mod thread_pool;
pub mod topology;

pub use avx2::{
    dispatch_matvec_f16, dispatch_matvec_f32, dispatch_matvec_q4_0_q8_0, dispatch_rmsnorm,
};
pub use scalar::{
    causal_gqa_decode_scalar, matvec_f16_scalar, matvec_f32_scalar, matvec_q4_0_q8_0_scalar,
    rmsnorm_scalar, rope_head_scalar, softmax_scalar, swiglu_scalar, vec_dot_q4_0_q8_0_scalar,
    RopeMode,
};
pub use thread_pool::{pin_current_thread, ThreadPool};
pub use topology::{parse_cpu_list, CpuFeatures, CpuTopology, CpuTopologyError, PhysicalCore};
