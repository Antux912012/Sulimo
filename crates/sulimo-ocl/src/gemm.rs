//! Q4_0 x F16 matrix-matrix multiplication execution on Intel Gen8–11 iGPU (@C:OCL-GEN9, §7 M3a).
//!
//! Enqueues GEMM kernels, sets up zero-copy buffer arguments, executes via OpenCL command queue,
//! and waits using the zero-spin OS parking waiter.

use crate::buffer::OclBuffer;
use crate::program::{KernelVariant, OclContext, OclError};
use crate::waiter::wait_event_zero_spin;
use half::f16;
use opencl3::kernel::ExecuteKernel;
use std::time::Instant;

/// Performance telemetry recorded during a GEMM kernel execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct GemmProfile {
    pub duration_seconds: f64,
    pub gflops: f64,
    pub kernel_variant: &'static str,
}

/// Executes Q4_0 SoA weights multiplied by F16 activations matrix: Y = X · W^T.
///
/// Shapes:
/// - `X`: [N, K] activations (N tokens, K input channels)
/// - `W`: [M, K] weights (M output features, K input channels) in Q4_0 SoA layout
/// - `Y`: [N, M] output activations
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_0_f16(
    ctx: &mut OclContext,
    scales_buf: &OclBuffer,
    quants_buf: &OclBuffer,
    x_buf: &OclBuffer,
    out_buf: &mut OclBuffer,
    m: usize,
    k: usize,
    n: usize,
    force_variant: Option<KernelVariant>,
) -> Result<GemmProfile, OclError> {
    assert_eq!(k % 32, 0, "K must be a multiple of 32 (Q4_0 block size)");
    assert_eq!(m % 8, 0, "M must be a multiple of 8 for SIMD8 sub-groups");

    let variant = force_variant.unwrap_or(ctx.active_variant);
    let kernel = match variant {
        KernelVariant::Intel => ctx
            .kernel_intel
            .as_mut()
            .unwrap_or(&mut ctx.kernel_portable),
        KernelVariant::Portable => &mut ctx.kernel_portable,
    };

    let m_i32 = m as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;

    let global_work_size = [m, n];
    let local_work_size = match variant {
        KernelVariant::Intel => [8, 1], // Exactly 1 sub-group of SIMD8 per work-group
        KernelVariant::Portable => [8, 1],
    };

    let start = Instant::now();

    let mut exec = ExecuteKernel::new(kernel);
    unsafe {
        exec.set_arg(&scales_buf.cl_mem())
            .set_arg(&quants_buf.cl_mem())
            .set_arg(&x_buf.cl_mem())
            .set_arg(&out_buf.cl_mem())
            .set_arg(&m_i32)
            .set_arg(&k_i32)
            .set_arg(&n_i32);
    }
    exec.set_global_work_sizes(&global_work_size);
    exec.set_local_work_sizes(&local_work_size);

    let event = unsafe {
        exec.enqueue_nd_range(&ctx.queue)
            .map_err(|e| OclError::Driver(format!("enqueue_nd_range: {e}")))?
    };

    // Zero-spin OS futex parking wait ($BATT, A9)
    wait_event_zero_spin(&ctx.queue, &event)?;

    let duration_seconds = start.elapsed().as_secs_f64();
    let total_flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let gflops = if duration_seconds > 0.0 {
        (total_flops / duration_seconds) / 1e9
    } else {
        0.0
    };

    let variant_name = match variant {
        KernelVariant::Intel => "Intel-SIMD8",
        KernelVariant::Portable => "Portable-FP16",
    };

    Ok(GemmProfile {
        duration_seconds,
        gflops,
        kernel_variant: variant_name,
    })
}

/// Scalar CPU reference implementation for Q4_0 SoA weights multiplied by F16 activations ($REF).
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_0_f16_scalar_reference(
    scales: &[f16],
    quants: &[u8],
    x: &[f16],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f16],
) {
    let num_blocks_per_row = k / 32;

    for token_n in 0..n {
        let x_row = &x[token_n * k..(token_n + 1) * k];

        for row_m in 0..m {
            let block_offset = row_m * num_blocks_per_row;
            let mut acc = 0.0f32;

            for b in 0..num_blocks_per_row {
                let d = scales[block_offset + b].to_f32();
                let q_slice = &quants[(block_offset + b) * 16..(block_offset + b + 1) * 16];
                let x_slice = &x_row[b * 32..(b + 1) * 32];

                let mut block_sum = 0.0f32;
                for i in 0..16 {
                    let q = q_slice[i];
                    let q_low = ((q & 0x0F) as i32 - 8) as f32;
                    let q_high = (((q >> 4) & 0x0F) as i32 - 8) as f32;

                    let x_low = x_slice[i].to_f32();
                    let x_high = x_slice[16 + i].to_f32();

                    block_sum += x_low * q_low + x_high * q_high;
                }

                acc += block_sum * d;
            }

            out[token_n * m + row_m] = f16::from_f32(acc);
        }
    }
}
