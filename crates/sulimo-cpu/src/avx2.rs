//! AVX2 and FMA accelerated SIMD kernels (@C:CPU, §6.2, $REF).
//!
//! Implements vectorized Q4_0·Q8_0 dot-products using AVX2 integer dot instructions
//! and `_mm256_fmadd_ps` accumulation, with dynamic runtime CPUID dispatch (A17).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::scalar::{
    matvec_f16_scalar, matvec_f32_scalar, matvec_q4_0_q8_0_scalar, rmsnorm_scalar,
};
use half::f16;
use sulimo_core::repack::Q4_0_BLOCK_SIZE;

/// Dispatches Q4_0·Q8_0 matrix-vector multiplication to AVX2+FMA or scalar fallback.
pub fn dispatch_matvec_q4_0_q8_0(
    w_scales: &[u8],
    w_quants: &[u8],
    x_q8: &[u8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                // SAFETY: Feature flags dynamically verified via CPUID before invocation.
                matvec_q4_0_q8_0_avx2(w_scales, w_quants, x_q8, rows, cols, out);
            }
            return;
        }
    }

    matvec_q4_0_q8_0_scalar(w_scales, w_quants, x_q8, rows, cols, out);
}

/// Dispatches RMSNorm to AVX2+FMA or scalar fallback.
pub fn dispatch_rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                // SAFETY: Feature flags dynamically verified via CPUID before invocation.
                rmsnorm_avx2(x, weight, eps, out);
            }
            return;
        }
    }

    rmsnorm_scalar(x, weight, eps, out);
}

/// Dispatches F32 matrix-vector multiplication to AVX2+FMA or scalar fallback.
pub fn dispatch_matvec_f32(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                // SAFETY: Feature flags dynamically verified via CPUID before invocation.
                matvec_f32_avx2(w, x, rows, cols, out);
            }
            return;
        }
    }

    matvec_f32_scalar(w, x, rows, cols, out);
}

/// Dispatches F16 matrix-vector multiplication to AVX2+FMA or scalar fallback.
pub fn dispatch_matvec_f16(w_bytes: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            unsafe {
                // SAFETY: Feature flags dynamically verified via CPUID before invocation.
                matvec_f16_avx2(w_bytes, x, rows, cols, out);
            }
            return;
        }
    }

    matvec_f16_scalar(w_bytes, x, rows, cols, out);
}

/// AVX2-accelerated single-row Q4_0(SoA)·Q8_0 dot product (§6.2).
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported by the host CPU, and
/// `w_scales`, `w_quants`, and `x_q8` contain at least `num_blocks` valid blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_q4_0_q8_0_avx2(
    w_scales: &[u8],
    w_quants: &[u8],
    x_q8: &[u8],
    num_blocks: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();

    for b in 0..num_blocks {
        let w_scale = f16::from_le_bytes([w_scales[b * 2], w_scales[b * 2 + 1]]).to_f32();
        let x_scale = f16::from_le_bytes([x_q8[b * 34], x_q8[b * 34 + 1]]).to_f32();
        let d = w_scale * x_scale;

        // Load 16 bytes (32 nibbles) of Q4_0 weights
        let w_ptr = w_quants.as_ptr().add(b * 16) as *const __m128i;
        let w128 = _mm_loadu_si128(w_ptr);

        let mask_low = _mm_set1_epi8(0x0F);
        let eight = _mm_set1_epi8(8);

        // Low nibbles: (w & 0x0F) - 8
        let w_low = _mm_sub_epi8(_mm_and_si128(w128, mask_low), eight);
        // High nibbles: ((w >> 4) & 0x0F) - 8
        let w_high = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(w128, 4), mask_low), eight);

        // Concatenate low (weights 0..15) and high (weights 16..31) into a 256-bit register
        let w256 = _mm256_set_m128i(w_high, w_low);

        // Load 32 bytes of Q8_0 activations
        let x_ptr = x_q8.as_ptr().add(b * 34 + 2) as *const __m256i;
        let x256 = _mm256_loadu_si256(x_ptr);

        // Multiply-add adjacent pairs of signed 8-bit integers
        let w_lo_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w256, 0));
        let w_hi_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w256, 1));

        let x_lo_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x256, 0));
        let x_hi_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x256, 1));

        let prod_lo = _mm256_madd_epi16(w_lo_16, x_lo_16);
        let prod_hi = _mm256_madd_epi16(w_hi_16, x_hi_16);

        let sum32 = _mm256_add_epi32(prod_lo, prod_hi);
        let sum_ps = _mm256_cvtepi32_ps(sum32);

        let d_vec = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(sum_ps, d_vec, acc);
    }

    // Horizontal reduction of acc
    hsum_256(acc)
}

/// AVX2-accelerated matrix-vector multiplication for Q4_0 (SoA).
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported by the host CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matvec_q4_0_q8_0_avx2(
    w_scales: &[u8],
    w_quants: &[u8],
    x_q8: &[u8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let blocks_per_row = cols / Q4_0_BLOCK_SIZE;
    let scales_row_stride = blocks_per_row * 2;
    let quants_row_stride = blocks_per_row * 16;

    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let scales_slice = &w_scales[r * scales_row_stride..(r + 1) * scales_row_stride];
        let quants_slice = &w_quants[r * quants_row_stride..(r + 1) * quants_row_stride];
        *out_val = vec_dot_q4_0_q8_0_avx2(scales_slice, quants_slice, x_q8, blocks_per_row);
    }
}

/// AVX2-accelerated RMSNorm with FMA vector reduction.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported by the host CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rmsnorm_avx2(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let len = x.len();
    let chunks = len / 8;
    let mut sum_sq_vec = _mm256_setzero_ps();

    for i in 0..chunks {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        sum_sq_vec = _mm256_fmadd_ps(xv, xv, sum_sq_vec);
    }

    let mut sum_sq = hsum_256(sum_sq_vec);
    for &x_val in x.iter().take(len).skip(chunks * 8) {
        sum_sq += x_val * x_val;
    }

    let mean = sum_sq / (len as f32);
    let scale = 1.0 / (mean + eps).sqrt();
    let scale_vec = _mm256_set1_ps(scale);

    for i in 0..chunks {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        let wv = _mm256_loadu_ps(weight.as_ptr().add(i * 8));
        let normed = _mm256_mul_ps(_mm256_mul_ps(xv, scale_vec), wv);
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), normed);
    }

    for i in (chunks * 8)..len {
        out[i] = x[i] * scale * weight[i];
    }
}

/// AVX2-accelerated F32 matrix-vector multiplication.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported by the host CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matvec_f32_avx2(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    let chunks = cols / 8;

    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let w_row = w.as_ptr().add(r * cols);
        let mut acc = _mm256_setzero_ps();

        for c in 0..chunks {
            let wv = _mm256_loadu_ps(w_row.add(c * 8));
            let xv = _mm256_loadu_ps(x.as_ptr().add(c * 8));
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }

        let mut sum = hsum_256(acc);
        for (c, &x_c) in x.iter().enumerate().take(cols).skip(chunks * 8) {
            sum += *w_row.add(c) * x_c;
        }
        *out_val = sum;
    }
}

/// AVX2-accelerated F16 matrix-vector multiplication using F16C conversion instructions.
///
/// # Safety
/// Caller must ensure AVX2, FMA, and F16C are supported by the host CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn matvec_f16_avx2(
    w_bytes: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let chunks = cols / 8;

    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let row_ptr = w_bytes.as_ptr().add(r * cols * 2);
        let mut acc = _mm256_setzero_ps();

        for c in 0..chunks {
            let half8 = _mm_loadu_si128(row_ptr.add(c * 16) as *const __m128i);
            let f32_vec = _mm256_cvtph_ps(half8);
            let xv = _mm256_loadu_ps(x.as_ptr().add(c * 8));
            acc = _mm256_fmadd_ps(f32_vec, xv, acc);
        }

        let mut sum = hsum_256(acc);
        for (c, &x_c) in x.iter().enumerate().take(cols).skip(chunks * 8) {
            let val = f16::from_le_bytes([*row_ptr.add(c * 2), *row_ptr.add(c * 2 + 1)]).to_f32();
            sum += val * x_c;
        }
        *out_val = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(final_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::vec_dot_q4_0_q8_0_scalar;
    use sulimo_core::kv_cache::quantize_f32_to_q8_0;
    use sulimo_core::repack::repack_q4_0_aos_to_soa;

    #[test]
    fn test_avx2_vs_scalar_q4_0_q8_0_precision() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        // Test over 8 blocks (256 weights)
        let num_blocks = 8;
        let num_weights = num_blocks * 32;

        let mut aos_data = Vec::with_capacity(num_blocks * 18);
        for b in 0..num_blocks {
            let scale = f16::from_f32(0.8 + (b as f32) * 0.1);
            aos_data.extend_from_slice(&scale.to_le_bytes());
            for i in 0..16 {
                let val = ((i * 3 + b) % 16) as u8;
                aos_data.push(val | ((15 - val) << 4));
            }
        }

        let mut w_scales = vec![0u8; num_blocks * 2];
        let mut w_quants = vec![0u8; num_blocks * 16];
        repack_q4_0_aos_to_soa(&aos_data, &mut w_scales, &mut w_quants).unwrap();

        let mut activations_f32 = vec![0.0f32; num_weights];
        for (i, item) in activations_f32.iter_mut().enumerate() {
            *item = ((i as f32) * 0.05).sin();
        }

        let mut x_q8 = vec![0u8; num_blocks * 34];
        quantize_f32_to_q8_0(&activations_f32, &mut x_q8);

        let scalar_result = vec_dot_q4_0_q8_0_scalar(&w_scales, &w_quants, &x_q8, num_blocks);
        let avx2_result =
            unsafe { vec_dot_q4_0_q8_0_avx2(&w_scales, &w_quants, &x_q8, num_blocks) };

        let diff = (scalar_result - avx2_result).abs();
        let rel_err = diff / (scalar_result.abs().max(1e-5));

        println!("Scalar dot: {scalar_result}, AVX2 dot: {avx2_result}, rel_err: {rel_err}");
        assert!(
            rel_err <= 1e-3,
            "AVX2 dot product relative error {rel_err} exceeds 1e-3 tolerance"
        );
    }
}
