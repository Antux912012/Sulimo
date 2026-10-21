//! Súlimo OpenCL backend (@C:OCL-GEN9).
//!
//! Provides hardware capability probing, OpenCL runtime management,
//! and FP16 SIMD8 kernels for Gen8–11 Intel graphics.

pub mod probe;

pub use probe::{probe, OclDeviceInfo, OclProbeResult};
