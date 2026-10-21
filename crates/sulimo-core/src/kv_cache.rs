//! Circular ring KV cache quantized to Q8_0 (@C:COL, A11, A12).
//!
//! Stores key and value historical context quantized to Q8_0 (34 bytes per 32 values),
//! capping memory usage at ~180–490 MB for 4K–8K contexts on Gen8–10 hardware.

use crate::arena::{Arena, ArenaError, ArenaRegion, ArenaSlice};
use crate::model_config::ModelConfig;
use half::f16;
use thiserror::Error;

/// Number of elements per Q8_0 quantization block.
pub const Q8_0_BLOCK_SIZE: usize = 32;

/// Raw byte size of a Q8_0 block (2-byte f16 scale + 32 x 1-byte signed int).
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Errors returned by the KV cache.
#[derive(Debug, Error)]
pub enum KvCacheError {
    #[error("arena memory allocation error: {0}")]
    Arena(#[from] ArenaError),
    #[error("context position {pos} exceeds cache capacity {capacity}")]
    PositionOutOfBounds { pos: usize, capacity: usize },
}

/// A circular Q8_0 key-value context buffer for a multi-layer transformer model.
pub struct KvCache {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of key-value attention heads per layer.
    pub n_head_kv: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Total token capacity of the circular buffer.
    pub capacity: usize,
    /// Bytes allocated per token position across all layers.
    pub bytes_per_token: usize,
    /// Key cache memory slice: [n_layers, capacity, n_head_kv * head_dim] in Q8_0.
    k_slices: Vec<ArenaSlice>,
    /// Value cache memory slice: [n_layers, capacity, n_head_kv * head_dim] in Q8_0.
    v_slices: Vec<ArenaSlice>,
    /// Number of Q8_0 blocks per token position in a single layer for K or V.
    pub blocks_per_layer: usize,
}

impl KvCache {
    /// Allocates a new Q8_0 circular KV cache in the static arena.
    pub fn allocate(
        arena: &mut Arena,
        config: &ModelConfig,
        context_length: usize,
    ) -> Result<Self, KvCacheError> {
        let n_layers = config.block_count;
        let n_head_kv = config.head_count_kv;
        let head_dim = config.head_dim;
        let capacity = context_length;

        let kv_dim = n_head_kv * head_dim;
        assert_eq!(
            kv_dim % Q8_0_BLOCK_SIZE,
            0,
            "KV dimension must be a multiple of 32"
        );
        let blocks_per_layer = kv_dim / Q8_0_BLOCK_SIZE;
        let bytes_per_layer_pos = blocks_per_layer * Q8_0_BLOCK_BYTES;
        let layer_total_bytes = capacity * bytes_per_layer_pos;

        let bytes_per_token = 2 * n_layers * bytes_per_layer_pos;

        let mut k_slices = Vec::with_capacity(n_layers);
        let mut v_slices = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            let k = arena.alloc(layer_total_bytes, ArenaRegion::KvCache)?;
            let v = arena.alloc(layer_total_bytes, ArenaRegion::KvCache)?;
            k_slices.push(k);
            v_slices.push(v);
        }

        Ok(Self {
            n_layers,
            n_head_kv,
            head_dim,
            capacity,
            bytes_per_token,
            k_slices,
            v_slices,
            blocks_per_layer,
        })
    }

    /// Stores a key vector at the given layer and position, quantizing f32 to Q8_0.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        let ring_pos = pos % self.capacity;
        let block_bytes = self.blocks_per_layer * Q8_0_BLOCK_BYTES;
        let offset = ring_pos * block_bytes;

        let slice = self.k_slices[layer].as_mut_slice();
        quantize_f32_to_q8_0(key, &mut slice[offset..offset + block_bytes]);
    }

    /// Stores a value vector at the given layer and position, quantizing f32 to Q8_0.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        let ring_pos = pos % self.capacity;
        let block_bytes = self.blocks_per_layer * Q8_0_BLOCK_BYTES;
        let offset = ring_pos * block_bytes;

        let slice = self.v_slices[layer].as_mut_slice();
        quantize_f32_to_q8_0(value, &mut slice[offset..offset + block_bytes]);
    }

    /// Returns a raw pointer to past keys at `pos` in `layer`.
    #[inline]
    pub fn key_ptr(&self, layer: usize, pos: usize) -> *const u8 {
        let ring_pos = pos % self.capacity;
        let block_bytes = self.blocks_per_layer * Q8_0_BLOCK_BYTES;
        let offset = ring_pos * block_bytes;
        unsafe { self.k_slices[layer].as_ptr().add(offset) }
    }

    /// Returns a raw pointer to past values at `pos` in `layer`.
    #[inline]
    pub fn value_ptr(&self, layer: usize, pos: usize) -> *const u8 {
        let ring_pos = pos % self.capacity;
        let block_bytes = self.blocks_per_layer * Q8_0_BLOCK_BYTES;
        let offset = ring_pos * block_bytes;
        unsafe { self.v_slices[layer].as_ptr().add(offset) }
    }

    /// Returns the Q8_0 bytes for past keys in a given layer up to length `context_tokens`.
    pub fn key_bytes(&self, layer: usize, pos: usize) -> &[u8] {
        let ring_pos = pos % self.capacity;
        let block_bytes = self.blocks_per_layer * Q8_0_BLOCK_BYTES;
        let offset = ring_pos * block_bytes;
        &self.k_slices[layer].as_slice()[offset..offset + block_bytes]
    }

    /// Returns the Q8_0 bytes for past values in a given layer up to length `context_tokens`.
    pub fn value_bytes(&self, layer: usize, pos: usize) -> &[u8] {
        let ring_pos = pos % self.capacity;
        let block_bytes = self.blocks_per_layer * Q8_0_BLOCK_BYTES;
        let offset = ring_pos * block_bytes;
        &self.v_slices[layer].as_slice()[offset..offset + block_bytes]
    }
}

/// Quantizes an array of f32 activations into Q8_0 blocks (§6.2).
///
/// Formula:
/// - `d = max(|x|) / 127.0`
/// - `qs[i] = round(x[i] / d) as i8`
pub fn quantize_f32_to_q8_0(src: &[f32], dst_q8: &mut [u8]) {
    assert_eq!(src.len() % Q8_0_BLOCK_SIZE, 0);
    let num_blocks = src.len() / Q8_0_BLOCK_SIZE;
    assert!(dst_q8.len() >= num_blocks * Q8_0_BLOCK_BYTES);

    for b in 0..num_blocks {
        let src_block = &src[b * 32..(b + 1) * 32];
        let dst_block = &mut dst_q8[b * 34..(b + 1) * 34];

        // 1. Find max absolute value
        let mut max_abs = 0.0f32;
        for &val in src_block {
            let a = val.abs();
            if a > max_abs {
                max_abs = a;
            }
        }

        let d = max_abs / 127.0;
        let d_f16 = f16::from_f32(d);
        dst_block[0..2].copy_from_slice(&d_f16.to_le_bytes());

        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        for i in 0..32 {
            let q = (src_block[i] * id).round();
            let clamped = q.clamp(-128.0, 127.0) as i8;
            dst_block[2 + i] = clamped as u8;
        }
    }
}

/// Dequantizes Q8_0 blocks back to f32.
pub fn dequantize_q8_0_to_f32(src_q8: &[u8], num_blocks: usize, dst: &mut [f32]) {
    assert!(src_q8.len() >= num_blocks * Q8_0_BLOCK_BYTES);
    assert!(dst.len() >= num_blocks * Q8_0_BLOCK_SIZE);

    for b in 0..num_blocks {
        let block = &src_q8[b * 34..(b + 1) * 34];
        let scale_bytes: [u8; 2] = [block[0], block[1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();

        for i in 0..32 {
            let q = block[2 + i] as i8;
            dst[b * 32 + i] = (q as f32) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_and_dequantize_q8_0() {
        let mut src = [0.0f32; 64];
        for (i, val) in src.iter_mut().enumerate() {
            *val = (i as f32) * 0.1 - 3.2;
        }

        let mut q8_bytes = [0u8; 68]; // 2 blocks * 34 bytes
        quantize_f32_to_q8_0(&src, &mut q8_bytes);

        let mut dst = [0.0f32; 64];
        dequantize_q8_0_to_f32(&q8_bytes, 2, &mut dst);

        for i in 0..64 {
            let diff = (src[i] - dst[i]).abs();
            // Maximum quantization error in Q8_0 is d / 2 = max_abs / 254 ≈ 0.015
            assert!(diff < 0.05, "diff {} at index {} too large", diff, i);
        }
    }
}
