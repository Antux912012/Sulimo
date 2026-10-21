//! Quantization block repacking and dequantization references (@C:MMAP, A6).
//!
//! Transforms Q4_0 tensors from standard 18-byte Array-of-Structures (AoS) blocks
//! into Structure-of-Arrays (SoA) layout:
//! - Contiguous 64-byte aligned array of f16 scales (`d[]`)
//! - Contiguous 64-byte aligned array of packed 4-bit nibbles (`qs[]`)
//!
//! This layout ensures SIMD vector loads on AVX2 CPU and sub-group loads on Gen9 GPU
//! never cross block boundaries or suffer unaligned load penalties.

use half::f16;
use thiserror::Error;

/// Number of weights in a Q4_0 quant block.
pub const Q4_0_BLOCK_SIZE: usize = 32;

/// Raw byte size of an AoS Q4_0 block (2-byte f16 scale + 16-byte quants).
pub const Q4_0_AOS_BLOCK_BYTES: usize = 18;

/// Errors returned during block repacking.
#[derive(Debug, Error)]
pub enum RepackError {
    #[error("input AoS length {0} is not a multiple of {Q4_0_AOS_BLOCK_BYTES}")]
    InvalidAosLength(usize),
    #[error("output scales buffer too small: need {need}, got {got}")]
    ScalesBufferTooSmall { need: usize, got: usize },
    #[error("output quants buffer too small: need {need}, got {got}")]
    QuantsBufferTooSmall { need: usize, got: usize },
}

/// Computes the required SoA buffer sizes (in bytes) for a given number of weights.
pub fn q4_0_soa_sizes(num_weights: usize) -> (usize, usize) {
    assert_eq!(
        num_weights % Q4_0_BLOCK_SIZE,
        0,
        "num_weights must be a multiple of 32"
    );
    let num_blocks = num_weights / Q4_0_BLOCK_SIZE;
    let scales_bytes = num_blocks * 2; // f16
    let quants_bytes = num_blocks * 16; // 16 bytes (32 x 4-bit)
    (scales_bytes, quants_bytes)
}

/// Repacks a Q4_0 tensor from AoS (18-byte blocks) to SoA (separate scales and quants arrays).
pub fn repack_q4_0_aos_to_soa(
    src_aos: &[u8],
    dst_scales: &mut [u8],
    dst_quants: &mut [u8],
) -> Result<usize, RepackError> {
    if !src_aos.len().is_multiple_of(Q4_0_AOS_BLOCK_BYTES) {
        return Err(RepackError::InvalidAosLength(src_aos.len()));
    }

    let num_blocks = src_aos.len() / Q4_0_AOS_BLOCK_BYTES;
    let need_scales = num_blocks * 2;
    let need_quants = num_blocks * 16;

    if dst_scales.len() < need_scales {
        return Err(RepackError::ScalesBufferTooSmall {
            need: need_scales,
            got: dst_scales.len(),
        });
    }

    if dst_quants.len() < need_quants {
        return Err(RepackError::QuantsBufferTooSmall {
            need: need_quants,
            got: dst_quants.len(),
        });
    }

    for b in 0..num_blocks {
        let aos_offset = b * Q4_0_AOS_BLOCK_BYTES;
        // Copy 2-byte f16 scale
        dst_scales[b * 2..b * 2 + 2].copy_from_slice(&src_aos[aos_offset..aos_offset + 2]);
        // Copy 16-byte quants
        dst_quants[b * 16..b * 16 + 16].copy_from_slice(&src_aos[aos_offset + 2..aos_offset + 18]);
    }

    Ok(num_blocks)
}

/// Scalar reference implementation of Q4_0 dequantization from standard AoS layout.
pub fn dequantize_q4_0_aos(src_aos: &[u8], dst_weights: &mut [f32]) {
    assert_eq!(src_aos.len() % Q4_0_AOS_BLOCK_BYTES, 0);
    let num_blocks = src_aos.len() / Q4_0_AOS_BLOCK_BYTES;
    assert_eq!(dst_weights.len(), num_blocks * Q4_0_BLOCK_SIZE);

    for b in 0..num_blocks {
        let block_bytes = &src_aos[b * Q4_0_AOS_BLOCK_BYTES..(b + 1) * Q4_0_AOS_BLOCK_BYTES];
        let scale_bytes: [u8; 2] = [block_bytes[0], block_bytes[1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();
        let qs = &block_bytes[2..18];

        let w_out = &mut dst_weights[b * 32..(b + 1) * 32];
        for i in 0..16 {
            let byte = qs[i];
            let v0 = (byte & 0x0F) as i32 - 8;
            let v1 = (byte >> 4) as i32 - 8;
            w_out[i] = (v0 as f32) * scale;
            w_out[i + 16] = (v1 as f32) * scale;
        }
    }
}

/// Scalar reference implementation of Q4_0 dequantization from repacked SoA layout.
pub fn dequantize_q4_0_soa(
    scales: &[u8],
    quants: &[u8],
    num_blocks: usize,
    dst_weights: &mut [f32],
) {
    assert!(scales.len() >= num_blocks * 2);
    assert!(quants.len() >= num_blocks * 16);
    assert_eq!(dst_weights.len(), num_blocks * Q4_0_BLOCK_SIZE);

    for b in 0..num_blocks {
        let scale_bytes: [u8; 2] = [scales[b * 2], scales[b * 2 + 1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();
        let qs = &quants[b * 16..(b + 1) * 16];

        let w_out = &mut dst_weights[b * 32..(b + 1) * 32];
        for i in 0..16 {
            let byte = qs[i];
            let v0 = (byte & 0x0F) as i32 - 8;
            let v1 = (byte >> 4) as i32 - 8;
            w_out[i] = (v0 as f32) * scale;
            w_out[i + 16] = (v1 as f32) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repack_and_dequantize_roundtrip_bit_exact() {
        // Construct 4 synthetic Q4_0 AoS blocks (128 weights)
        let num_blocks = 4;
        let mut aos_data = Vec::with_capacity(num_blocks * 18);

        for b in 0..num_blocks {
            let scale = f16::from_f32(0.5 + (b as f32) * 0.25);
            aos_data.extend_from_slice(&scale.to_le_bytes());
            for i in 0..16 {
                let nibble_low = ((i + b) % 16) as u8;
                let nibble_high = ((i * 2 + b) % 16) as u8;
                aos_data.push(nibble_low | (nibble_high << 4));
            }
        }

        // Allocate SoA buffers
        let (scales_len, quants_len) = q4_0_soa_sizes(num_blocks * 32);
        let mut dst_scales = vec![0u8; scales_len];
        let mut dst_quants = vec![0u8; quants_len];

        let repacked_blocks =
            repack_q4_0_aos_to_soa(&aos_data, &mut dst_scales, &mut dst_quants).unwrap();
        assert_eq!(repacked_blocks, num_blocks);

        // Dequantize from AoS
        let mut weights_aos = vec![0.0f32; num_blocks * 32];
        dequantize_q4_0_aos(&aos_data, &mut weights_aos);

        // Dequantize from SoA
        let mut weights_soa = vec![0.0f32; num_blocks * 32];
        dequantize_q4_0_soa(&dst_scales, &dst_quants, num_blocks, &mut weights_soa);

        // Assert 100% bit-exact equality (§7 M1 Acceptance Criterion)
        for i in 0..weights_aos.len() {
            assert_eq!(
                weights_aos[i].to_bits(),
                weights_soa[i].to_bits(),
                "bit mismatch at weight index {}",
                i
            );
        }
    }
}
