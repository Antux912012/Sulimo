//! Scalar reference kernels for transformer execution (@C:CPU, §6.3, $REF).
//!
//! Provides mathematically exact, scalar f32 reference implementations for RMSNorm,
//! RoPE (NORM and NEOX modes), SwiGLU, Softmax, Q4_0·Q8_0 matrix-vector dot products,
//! and Grouped Query Attention (GQA).

use half::f16;
use sulimo_core::kv_cache::{dequantize_q8_0_to_f32, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
use sulimo_core::repack::Q4_0_BLOCK_SIZE;

/// RoPE rotation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeMode {
    /// Llama style: adjacent pairs `(x[2*i], x[2*i+1])` rotated.
    Norm,
    /// Qwen2 / GPT-NeoX style: split halves `(x[i], x[i + dim/2])` rotated.
    Neox,
}

/// In-place Root Mean Square Layer Normalization:
///
/// `y[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
pub fn rmsnorm_scalar(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());

    let mut sum_sq = 0.0f32;
    for &val in x {
        sum_sq += val * val;
    }

    let mean = sum_sq / (x.len() as f32);
    let scale = 1.0 / (mean + eps).sqrt();

    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Rotary Position Embedding (RoPE) application for a single attention head.
pub fn rope_head_scalar(
    vec: &mut [f32],
    pos: usize,
    _head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    mode: RopeMode,
    freq_factors: Option<&[f32]>,
) {
    let half_dim = rope_dim / 2;

    for i in 0..half_dim {
        let mut freq = 1.0 / freq_base.powf((2 * i) as f32 / (rope_dim as f32));
        if let Some(factors) = freq_factors {
            if i < factors.len() {
                freq *= factors[i];
            }
        }

        let theta = (pos as f32) * freq;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        match mode {
            RopeMode::Norm => {
                let v0 = vec[2 * i];
                let v1 = vec[2 * i + 1];
                vec[2 * i] = v0 * cos_theta - v1 * sin_theta;
                vec[2 * i + 1] = v0 * sin_theta + v1 * cos_theta;
            }
            RopeMode::Neox => {
                let v0 = vec[i];
                let v1 = vec[i + half_dim];
                vec[i] = v0 * cos_theta - v1 * sin_theta;
                vec[i + half_dim] = v0 * sin_theta + v1 * cos_theta;
            }
        }
    }
}

/// Numerically stable in-place Softmax over a vector.
pub fn softmax_scalar(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    let mut max_val = x[0];
    for &val in x.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }

    let mut sum = 0.0f32;
    for val in x.iter_mut() {
        let e = (*val - max_val).exp();
        *val = e;
        sum += e;
    }

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for val in x.iter_mut() {
            *val *= inv_sum;
        }
    }
}

/// SwiGLU activation: `out = silu(gate) * up`.
pub fn swiglu_scalar(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());

    for i in 0..gate.len() {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        out[i] = silu_g * up[i];
    }
}

/// Vector dot product between a Q4_0 (SoA) weight row and a Q8_0 activation vector (§6.2).
pub fn vec_dot_q4_0_q8_0_scalar(
    w_scales: &[u8],
    w_quants: &[u8],
    x_q8: &[u8],
    num_blocks: usize,
) -> f32 {
    let mut total_sum = 0.0f32;

    for b in 0..num_blocks {
        let w_scale = f16::from_le_bytes([w_scales[b * 2], w_scales[b * 2 + 1]]).to_f32();
        let x_scale = f16::from_le_bytes([x_q8[b * 34], x_q8[b * 34 + 1]]).to_f32();
        let d = w_scale * x_scale;

        let w_qs = &w_quants[b * 16..(b + 1) * 16];
        let x_qs = &x_q8[b * 34 + 2..(b + 1) * 34];

        let mut block_sum = 0i32;
        for i in 0..16 {
            let byte = w_qs[i];
            let w0 = (byte & 0x0F) as i32 - 8;
            let w1 = (byte >> 4) as i32 - 8;

            let x0 = (x_qs[i] as i8) as i32;
            let x1 = (x_qs[i + 16] as i8) as i32;

            block_sum += w0 * x0 + w1 * x1;
        }

        total_sum += (block_sum as f32) * d;
    }

    total_sum
}

/// Matrix-vector multiplication for Q4_0 (SoA) weights against Q8_0 activations.
pub fn matvec_q4_0_q8_0_scalar(
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

    for r in 0..rows {
        let scales_slice = &w_scales[r * scales_row_stride..(r + 1) * scales_row_stride];
        let quants_slice = &w_quants[r * quants_row_stride..(r + 1) * quants_row_stride];
        out[r] = vec_dot_q4_0_q8_0_scalar(scales_slice, quants_slice, x_q8, blocks_per_row);
    }
}

/// Matrix-vector multiplication for F32 weights against F32 activations.
pub fn matvec_f32_scalar(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for r in 0..rows {
        let mut sum = 0.0f32;
        let row_slice = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            sum += row_slice[c] * x[c];
        }
        out[r] = sum;
    }
}

/// Matrix-vector multiplication for F16 weights against F32 activations.
pub fn matvec_f16_scalar(w_bytes: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let mut sum = 0.0f32;
        let row_offset = r * cols * 2;
        for (c, &x_c) in x.iter().enumerate().take(cols) {
            let offset = row_offset + c * 2;
            let val = f16::from_le_bytes([w_bytes[offset], w_bytes[offset + 1]]).to_f32();
            sum += val * x_c;
        }
        *out_val = sum;
    }
}

/// Causal Grouped Query Attention (GQA) scalar reference for a single decode position.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_decode_scalar<FK, FV>(
    q: &[f32],
    get_k: FK,
    get_v: FV,
    context_len: usize,
    head_count: usize,
    head_count_kv: usize,
    head_dim: usize,
    scores_buf: &mut [f32],
    dequant_buf: &mut [f32],
    out: &mut [f32],
) where
    FK: Fn(usize) -> *const u8,
    FV: Fn(usize) -> *const u8,
{
    let heads_per_kv = head_count / head_count_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_bytes_len = (head_dim / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;

    for h in 0..head_count {
        let kv_head = h / heads_per_kv;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];

        // 1. Calculate attention scores: Q · Kᵀ * scale
        for (pos, score_pos) in scores_buf.iter_mut().enumerate().take(context_len) {
            let k_pos_ptr = get_k(pos);
            let k_head_offset = kv_head * head_bytes_len;
            let k_head_bytes = unsafe {
                // SAFETY: Pointer is valid within the KV cache bounds.
                std::slice::from_raw_parts(k_pos_ptr.add(k_head_offset), head_bytes_len)
            };

            dequantize_q8_0_to_f32(k_head_bytes, head_dim / Q8_0_BLOCK_SIZE, dequant_buf);

            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * dequant_buf[d];
            }
            *score_pos = dot * scale;
        }

        // 2. Softmax over context positions [0..context_len]
        softmax_scalar(&mut scores_buf[..context_len]);

        // 3. Weighted sum of past values: Scores · V
        let out_head = &mut out[h * head_dim..(h + 1) * head_dim];
        for out_val in out_head.iter_mut().take(head_dim) {
            *out_val = 0.0;
        }

        for (pos, &weight) in scores_buf.iter().enumerate().take(context_len) {
            let v_pos_ptr = get_v(pos);
            let v_head_offset = kv_head * head_bytes_len;
            let v_head_bytes = unsafe {
                // SAFETY: Pointer is valid within the KV cache bounds.
                std::slice::from_raw_parts(v_pos_ptr.add(v_head_offset), head_bytes_len)
            };

            dequantize_q8_0_to_f32(v_head_bytes, head_dim / Q8_0_BLOCK_SIZE, dequant_buf);

            for d in 0..head_dim {
                out_head[d] += weight * dequant_buf[d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_scalar() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        rmsnorm_scalar(&x, &w, 1e-5, &mut out);

        let mut sum_sq = 0.0f32;
        for &v in &out {
            sum_sq += v * v;
        }
        let norm = (sum_sq / 4.0).sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_softmax_scalar() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax_scalar(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }
}
