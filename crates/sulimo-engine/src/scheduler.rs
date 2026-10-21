//! Hybrid Asymmetric Scheduler coordinating CPU decode and iGPU prefill (@C:ENG, §7 M4, $BATT).
//!
//! Schedules prompt prefill on Intel Gen8–11 OpenCL iGPU when $N \ge T$ with zero-copy
//! unified buffers ($ZCOPY), and executes single-token autoregressive decoding on physical CPU
//! cores (AVX2+FMA) while the iGPU idles into RC6 deep sleep. Automatically handles context
//! repacking on capacity overflow (A12) and supports responsive cancellation.

use half::f16;
use std::sync::Arc;
use sulimo_core::arena::{ArenaRegion, ArenaSlice};
use sulimo_core::kv_cache::KvCache;
use sulimo_core::loader::{LoadedModel, LoadedTensor, LoadedTensorData, LoaderError};
#[cfg(feature = "opencl")]
use sulimo_ocl::{
    OclBuffer, OclContext, OclError, OclKvCache, OclLayerTensors, OclModelTensors,
    OclPrefillScratch,
};
use thiserror::Error;

/// Scheduler and execution errors.
#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("loader error: {0}")]
    Loader(#[from] LoaderError),
    #[cfg(feature = "opencl")]
    #[error("OpenCL error: {0}")]
    OpenCl(#[from] OclError),
    #[error("arena memory allocation error: {0}")]
    Arena(#[from] sulimo_core::arena::ArenaError),
    #[error("cancelled by user")]
    Cancelled,
}

/// Active execution backend selected for a prompt or prefill chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillBackend {
    GpuOpenCl,
    CpuAvx2,
}

/// Configuration knobs for the hybrid asymmetric scheduler (§4.5, §7 M4).
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Token threshold $T$: if prompt length $N \ge T$ and OpenCL is active, use GPU. Default: 8.
    pub gpu_threshold: usize,
    /// Maximum number of tokens per GPU prefill micro-batch. Default: 512.
    pub ubatch_size: usize,
    /// Whether OpenCL GPU prefill is enabled. Default: true.
    pub opencl_enabled: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: 8,
            ubatch_size: 512,
            opencl_enabled: true,
        }
    }
}

/// Pre-allocated GPU zero-copy buffer state for prompt prefill ($ZCOPY).
#[cfg(feature = "opencl")]
pub struct OclEngineState {
    pub ctx: Arc<OclContext>,
    pub model_tensors: OclModelTensors,
    pub kv_cache: OclKvCache,
    pub scratch: OclPrefillScratch,
    pub scratch_x_slice: ArenaSlice,
    pub scratch_logits_slice: ArenaSlice,
    pub ubatch_size: usize,
}

#[cfg(feature = "opencl")]
impl OclEngineState {
    /// Initializes zero-copy OpenCL buffers wrapping model weights, KV cache, and scratch arena ($ZCOPY).
    pub fn try_init(
        model: &mut LoadedModel,
        kv_cache: &KvCache,
        ubatch_size: usize,
    ) -> Result<Self, SchedulerError> {
        let ctx = OclContext::get_or_init()?;

        let cfg = &model.config;
        let embd = cfg.embedding_length;
        let ffn = cfg.feed_forward_length;
        let head_count = cfg.head_count;
        let head_count_kv = cfg.head_count_kv;
        let head_dim = cfg.head_dim;
        let vocab_size = cfg.vocab_size;
        let block_count = cfg.block_count;

        let q_dim = head_count * head_dim;
        let kv_dim = head_count_kv * head_dim;

        // 1. Wrap model weights in zero-copy OclLayerTensors
        let mut ocl_layers = Vec::with_capacity(block_count);
        for l in 0..block_count {
            let attn_norm = get_raw_ocl_buffer(&ctx, model, &format!("blk.{l}.attn_norm.weight"))?;
            let (attn_q_scales, attn_q_quants) =
                get_q4_ocl_buffers(&ctx, model, &format!("blk.{l}.attn_q.weight"))?;
            let (attn_k_scales, attn_k_quants) =
                get_q4_ocl_buffers(&ctx, model, &format!("blk.{l}.attn_k.weight"))?;
            let (attn_v_scales, attn_v_quants) =
                get_q4_ocl_buffers(&ctx, model, &format!("blk.{l}.attn_v.weight"))?;
            let (attn_out_scales, attn_out_quants) =
                get_q4_ocl_buffers(&ctx, model, &format!("blk.{l}.attn_output.weight"))?;
            let ffn_norm = get_raw_ocl_buffer(&ctx, model, &format!("blk.{l}.ffn_norm.weight"))?;
            let (ffn_gate_scales, ffn_gate_quants) =
                get_q4_ocl_buffers(&ctx, model, &format!("blk.{l}.ffn_gate.weight"))?;
            let (ffn_up_scales, ffn_up_quants) =
                get_q4_ocl_buffers(&ctx, model, &format!("blk.{l}.ffn_up.weight"))?;
            let (ffn_down_scales, ffn_down_quants) =
                get_q4_ocl_buffers(&ctx, model, &format!("blk.{l}.ffn_down.weight"))?;

            ocl_layers.push(OclLayerTensors {
                attn_norm,
                attn_q_scales,
                attn_q_quants,
                attn_k_scales,
                attn_k_quants,
                attn_v_scales,
                attn_v_quants,
                attn_out_scales,
                attn_out_quants,
                ffn_norm,
                ffn_gate_scales,
                ffn_gate_quants,
                ffn_up_scales,
                ffn_up_quants,
                ffn_down_scales,
                ffn_down_quants,
            });
        }

        let output_norm = get_raw_ocl_buffer(&ctx, model, "output_norm.weight")?;
        let output_tensor_name = if model.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let (output_scales, output_quants) = get_q4_ocl_buffers(&ctx, model, output_tensor_name)?;

        let model_tensors = OclModelTensors {
            output_norm,
            output_scales,
            output_quants,
            layers: ocl_layers,
        };

        // 2. Wrap shared ring KV cache in zero-copy OclKvCache handles ($ZCOPY)
        let mut k_buffers = Vec::with_capacity(block_count);
        let mut v_buffers = Vec::with_capacity(block_count);

        for l in 0..block_count {
            let k_slice = kv_cache.k_slice(l);
            let v_slice = kv_cache.v_slice(l);

            let k_buf = unsafe {
                OclBuffer::from_host_ptr(
                    &ctx.context,
                    &ctx.queue,
                    k_slice.as_ptr(),
                    k_slice.padded_size(),
                    false,
                )?
            };
            let v_buf = unsafe {
                OclBuffer::from_host_ptr(
                    &ctx.context,
                    &ctx.queue,
                    v_slice.as_ptr(),
                    v_slice.padded_size(),
                    false,
                )?
            };

            k_buffers.push(k_buf);
            v_buffers.push(v_buf);
        }

        let ocl_kv = OclKvCache {
            k_buffers,
            v_buffers,
            capacity: kv_cache.capacity,
            head_count_kv,
            head_dim,
        };

        // 3. Allocate zero-copy scratch memory in static arena ($MEM<8G)
        let max_n = ubatch_size;
        let x_slice = model.arena.alloc(max_n * embd * 2, ArenaRegion::Scratch)?;
        let h_slice = model.arena.alloc(max_n * embd * 2, ArenaRegion::Scratch)?;
        let q_slice = model.arena.alloc(max_n * q_dim * 2, ArenaRegion::Scratch)?;
        let k_slice = model
            .arena
            .alloc(max_n * kv_dim * 2, ArenaRegion::Scratch)?;
        let v_slice = model
            .arena
            .alloc(max_n * kv_dim * 2, ArenaRegion::Scratch)?;
        let attn_out_slice = model.arena.alloc(max_n * q_dim * 2, ArenaRegion::Scratch)?;
        let ffn_gate_slice = model.arena.alloc(max_n * ffn * 2, ArenaRegion::Scratch)?;
        let ffn_up_slice = model.arena.alloc(max_n * ffn * 2, ArenaRegion::Scratch)?;
        let ffn_swiglu_slice = model.arena.alloc(max_n * ffn * 2, ArenaRegion::Scratch)?;
        let ffn_down_slice = model.arena.alloc(max_n * embd * 2, ArenaRegion::Scratch)?;
        let logits_slice = model.arena.alloc(vocab_size * 4, ArenaRegion::Scratch)?;

        let wrap = |slice: &ArenaSlice| -> Result<OclBuffer, OclError> {
            unsafe {
                OclBuffer::from_host_ptr(
                    &ctx.context,
                    &ctx.queue,
                    slice.as_ptr(),
                    slice.padded_size(),
                    false,
                )
            }
        };

        let ocl_scratch = OclPrefillScratch {
            x: wrap(&x_slice)?,
            h: wrap(&h_slice)?,
            q: wrap(&q_slice)?,
            k: wrap(&k_slice)?,
            v: wrap(&v_slice)?,
            attn_out: wrap(&attn_out_slice)?,
            ffn_gate: wrap(&ffn_gate_slice)?,
            ffn_up: wrap(&ffn_up_slice)?,
            ffn_swiglu: wrap(&ffn_swiglu_slice)?,
            ffn_down: wrap(&ffn_down_slice)?,
            logits: wrap(&logits_slice)?,
            max_n,
        };

        Ok(Self {
            ctx,
            model_tensors,
            kv_cache: ocl_kv,
            scratch: ocl_scratch,
            scratch_x_slice: x_slice,
            scratch_logits_slice: logits_slice,
            ubatch_size,
        })
    }
}

#[cfg(feature = "opencl")]
impl Drop for OclEngineState {
    fn drop(&mut self) {
        self.ctx.finish();
    }
}

#[cfg(feature = "opencl")]
fn get_raw_ocl_buffer(
    ctx: &OclContext,
    model: &LoadedModel,
    name: &str,
) -> Result<OclBuffer, OclError> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| OclError::Driver(format!("missing tensor: {name}")))?;

    match &tensor.data {
        LoadedTensorData::Raw(slice) => unsafe {
            OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                slice.as_ptr(),
                slice.padded_size(),
                true,
            )
        },
        _ => Err(OclError::Driver(format!("tensor {name} is not raw"))),
    }
}

#[cfg(feature = "opencl")]
fn get_q4_ocl_buffers(
    ctx: &OclContext,
    model: &LoadedModel,
    name: &str,
) -> Result<(OclBuffer, OclBuffer), OclError> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| OclError::Driver(format!("missing tensor: {name}")))?;

    match &tensor.data {
        LoadedTensorData::Q4_0 { scales, quants, .. } => unsafe {
            let sc = OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                scales.as_ptr(),
                scales.padded_size(),
                true,
            )?;
            let qu = OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                quants.as_ptr(),
                quants.padded_size(),
                true,
            )?;
            Ok((sc, qu))
        },
        _ => Err(OclError::Driver(format!("tensor {name} is not Q4_0"))),
    }
}

/// Context repacking on capacity overflow (A12).
///
/// Evicts middle conversation turns while preserving initial system prompt tokens
/// and most recent conversational context, returning the new sequence of tokens.
pub fn repack_context(past_tokens: &[u32], capacity: usize) -> (Vec<u32>, usize) {
    let sys_len = (capacity / 8).clamp(4, 64).min(past_tokens.len());
    let recent_len = (capacity / 2).min(past_tokens.len().saturating_sub(sys_len));

    let mut new_tokens = Vec::with_capacity(sys_len + recent_len);
    new_tokens.extend_from_slice(&past_tokens[..sys_len]);
    new_tokens.extend_from_slice(&past_tokens[past_tokens.len() - recent_len..]);

    let new_pos = new_tokens.len();
    (new_tokens, new_pos)
}

/// Looks up token embedding vector and converts directly to FP16 in destination slice.
pub fn lookup_token_embedding_f16(
    tensor: &LoadedTensor,
    embd_dim: usize,
    token_id: u32,
    out_f16: &mut [f16],
) {
    let tid = token_id as usize;
    match &tensor.data {
        LoadedTensorData::Q4_0 {
            scales,
            quants,
            num_blocks,
        } => {
            let blocks_per_row = *num_blocks;
            let scales_stride = blocks_per_row * 2;
            let quants_stride = blocks_per_row * 16;

            let row_scales = &scales.as_slice()[tid * scales_stride..(tid + 1) * scales_stride];
            let row_quants = &quants.as_slice()[tid * quants_stride..(tid + 1) * quants_stride];

            for b in 0..blocks_per_row {
                let d = f16::from_le_bytes([row_scales[b * 2], row_scales[b * 2 + 1]]);
                let q_slice = &row_quants[b * 16..(b + 1) * 16];
                let out_offset = b * 32;

                for i in 0..16 {
                    let byte = q_slice[i];
                    let q_low = (byte & 0x0F) as i32 - 8;
                    let q_high = ((byte >> 4) & 0x0F) as i32 - 8;

                    out_f16[out_offset + i] = d * f16::from_f32(q_low as f32);
                    out_f16[out_offset + 16 + i] = d * f16::from_f32(q_high as f32);
                }
            }
        }
        LoadedTensorData::Raw(slice) => {
            let row_bytes = &slice.as_slice()[tid * embd_dim * 4..(tid + 1) * embd_dim * 4];
            for i in 0..embd_dim {
                let bytes: [u8; 4] = [
                    row_bytes[i * 4],
                    row_bytes[i * 4 + 1],
                    row_bytes[i * 4 + 2],
                    row_bytes[i * 4 + 3],
                ];
                let val_f32 = f32::from_le_bytes(bytes);
                out_f16[i] = f16::from_f32(val_f32);
            }
        }
    }
}
