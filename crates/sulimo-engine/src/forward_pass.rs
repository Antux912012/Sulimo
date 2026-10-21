//! Transformer forward pass execution without allocations on hot path (@C:ENG, §6.3, $NOALLOC).
//!
//! Orchestrates layer-by-layer inference: embedding lookup, RMSNorm, projection GEMVs,
//! RoPE rotation, Q8_0 KV cache storage, GQA attention, SwiGLU FFN, and logits output.

use sulimo_core::kv_cache::{quantize_f32_to_q8_0, KvCache, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
use sulimo_core::loader::{LoadedModel, LoadedTensor, LoadedTensorData};
use sulimo_core::model_config::{ModelArch, ModelConfig};
use sulimo_core::repack::{dequantize_q4_0_soa, Q4_0_BLOCK_SIZE};
use sulimo_cpu::avx2::{
    dispatch_matvec_f16, dispatch_matvec_f32, dispatch_matvec_q4_0_q8_0, dispatch_rmsnorm,
    dispatch_rmsnorm_gemma,
};
use sulimo_cpu::scalar::{
    causal_gqa_decode_scalar_ext, geglu_scalar, logit_softcap_scalar, rope_head_scalar,
    swiglu_scalar, RopeMode,
};
use sulimo_cpu::thread_pool::ThreadPool;

/// Pre-resolved tensor references for a single transformer layer to avoid hash lookups and string allocations.
pub struct LayerTensors<'a> {
    pub attn_norm: &'a LoadedTensor,
    pub attn_q: &'a LoadedTensor,
    pub attn_k: &'a LoadedTensor,
    pub attn_v: &'a LoadedTensor,
    pub attn_out: &'a LoadedTensor,
    pub ffn_norm: &'a LoadedTensor,
    pub ffn_gate: &'a LoadedTensor,
    pub ffn_up: &'a LoadedTensor,
    pub ffn_down: &'a LoadedTensor,
    pub attn_q_bias: Option<&'a LoadedTensor>,
    pub attn_k_bias: Option<&'a LoadedTensor>,
    pub attn_v_bias: Option<&'a LoadedTensor>,
    pub post_attn_norm: Option<&'a LoadedTensor>,
    pub post_ffn_norm: Option<&'a LoadedTensor>,
    pub attn_q_norm: Option<&'a LoadedTensor>,
    pub attn_k_norm: Option<&'a LoadedTensor>,
}

/// Pre-resolved tensor index across the full model ($NOALLOC).
pub struct ModelTensors<'a> {
    pub token_embd: &'a LoadedTensor,
    pub output_norm: &'a LoadedTensor,
    pub output: &'a LoadedTensor,
    pub layers: Vec<LayerTensors<'a>>,
}

impl<'a> ModelTensors<'a> {
    pub fn build(model: &'a LoadedModel) -> Self {
        let token_embd = model
            .tensors
            .get("token_embd.weight")
            .expect("missing token_embd.weight");
        let output_norm = model
            .tensors
            .get("output_norm.weight")
            .expect("missing output_norm.weight");
        let output = model.tensors.get("output.weight").unwrap_or(token_embd);

        let mut layers = Vec::with_capacity(model.config.block_count);
        for l in 0..model.config.block_count {
            let attn_norm = model
                .tensors
                .get(&format!("blk.{l}.attn_norm.weight"))
                .unwrap();
            let attn_q = model
                .tensors
                .get(&format!("blk.{l}.attn_q.weight"))
                .unwrap();
            let attn_k = model
                .tensors
                .get(&format!("blk.{l}.attn_k.weight"))
                .unwrap();
            let attn_v = model
                .tensors
                .get(&format!("blk.{l}.attn_v.weight"))
                .unwrap();
            let attn_out = model
                .tensors
                .get(&format!("blk.{l}.attn_output.weight"))
                .unwrap();
            let ffn_norm = model
                .tensors
                .get(&format!("blk.{l}.ffn_norm.weight"))
                .unwrap();
            let ffn_gate = model
                .tensors
                .get(&format!("blk.{l}.ffn_gate.weight"))
                .unwrap();
            let ffn_up = model
                .tensors
                .get(&format!("blk.{l}.ffn_up.weight"))
                .unwrap();
            let ffn_down = model
                .tensors
                .get(&format!("blk.{l}.ffn_down.weight"))
                .unwrap();

            let attn_q_bias = model.tensors.get(&format!("blk.{l}.attn_q.bias"));
            let attn_k_bias = model.tensors.get(&format!("blk.{l}.attn_k.bias"));
            let attn_v_bias = model.tensors.get(&format!("blk.{l}.attn_v.bias"));
            let post_attn_norm = model
                .tensors
                .get(&format!("blk.{l}.post_attention_norm.weight"));
            let post_ffn_norm = model.tensors.get(&format!("blk.{l}.post_ffn_norm.weight"));
            let attn_q_norm = model.tensors.get(&format!("blk.{l}.attn_q_norm.weight"));
            let attn_k_norm = model.tensors.get(&format!("blk.{l}.attn_k_norm.weight"));

            layers.push(LayerTensors {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_out,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                post_attn_norm,
                post_ffn_norm,
                attn_q_norm,
                attn_k_norm,
            });
        }

        Self {
            token_embd,
            output_norm,
            output,
            layers,
        }
    }
}
pub struct ScratchBuffers {
    pub x: Vec<f32>,
    pub h: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_swiglu: Vec<f32>,
    pub ffn_down: Vec<f32>,
    pub logits: Vec<f32>,
    pub q8_buf_embd: Vec<u8>,
    pub q8_buf_ffn: Vec<u8>,
    pub scores_buf: Vec<f32>,
    pub dequant_buf: Vec<f32>,
}

impl ScratchBuffers {
    /// Pre-allocates all intermediate activation vectors based on model hyperparameters.
    pub fn new(config: &ModelConfig) -> Self {
        let embd = config.embedding_length;
        let ffn = config.feed_forward_length;
        let head_dim = config.head_dim;
        let q_dim = config.head_count * head_dim;
        let kv_dim = config.head_count_kv * head_dim;
        let vocab = config.vocab_size;
        let ctx = config.context_length;

        let q8_embd_blocks = embd.div_ceil(Q8_0_BLOCK_SIZE);
        let q8_ffn_blocks = ffn.div_ceil(Q8_0_BLOCK_SIZE);

        Self {
            x: vec![0.0f32; embd],
            h: vec![0.0f32; embd],
            q: vec![0.0f32; q_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_out: vec![0.0f32; embd],
            ffn_gate: vec![0.0f32; ffn],
            ffn_up: vec![0.0f32; ffn],
            ffn_swiglu: vec![0.0f32; ffn],
            ffn_down: vec![0.0f32; embd],
            logits: vec![0.0f32; vocab],
            q8_buf_embd: vec![0u8; q8_embd_blocks * Q8_0_BLOCK_BYTES],
            q8_buf_ffn: vec![0u8; q8_ffn_blocks * Q8_0_BLOCK_BYTES],
            scores_buf: vec![0.0f32; ctx],
            dequant_buf: vec![0.0f32; head_dim],
        }
    }
}

/// Executes a single autoregressive token forward pass through all transformer layers ($NOALLOC).
pub fn forward_token<'s>(
    cfg: &ModelConfig,
    tensors: &ModelTensors,
    kv_cache: &mut KvCache,
    token_id: u32,
    pos: usize,
    scratch: &'s mut ScratchBuffers,
    _pool: &mut ThreadPool,
) -> &'s [f32] {
    let embd = cfg.embedding_length;
    let eps = cfg.rms_epsilon;
    let rope_mode = match cfg.arch {
        ModelArch::Llama => RopeMode::Norm,
        ModelArch::Qwen2 | ModelArch::Gemma2 | ModelArch::MiniCPM3 | ModelArch::LFM => {
            RopeMode::Neox
        }
    };

    // 1. Embedding lookup: x = token_embd[token_id]
    lookup_token_embedding(tensors.token_embd, embd, token_id, &mut scratch.x);
    if cfg.arch == ModelArch::Gemma2 {
        let scale = (embd as f32).sqrt();
        for val in scratch.x.iter_mut() {
            *val *= scale;
        }
    } else if let Some(scale) = cfg.scale_emb {
        for val in scratch.x.iter_mut() {
            *val *= scale;
        }
    }

    // 2. Transformer layers loop
    for (l, layer) in tensors.layers.iter().enumerate() {
        // Attention RMSNorm: h = rmsnorm(x, eps) * attn_norm
        apply_rmsnorm(cfg.arch, layer.attn_norm, &scratch.x, eps, &mut scratch.h);

        // Quantize h to Q8_0 for projection dot products
        quantize_f32_to_q8_0(&scratch.h, &mut scratch.q8_buf_embd);

        // Q, K, V linear projections: W · h
        apply_matvec(
            layer.attn_q,
            &scratch.q8_buf_embd,
            &scratch.h,
            &mut scratch.q,
        );
        apply_matvec(
            layer.attn_k,
            &scratch.q8_buf_embd,
            &scratch.h,
            &mut scratch.k,
        );
        apply_matvec(
            layer.attn_v,
            &scratch.q8_buf_embd,
            &scratch.h,
            &mut scratch.v,
        );

        // Optional QKV biases (Qwen2)
        add_bias(layer.attn_q_bias, &mut scratch.q);
        add_bias(layer.attn_k_bias, &mut scratch.k);
        add_bias(layer.attn_v_bias, &mut scratch.v);

        // Optional QK-Norm (MiniCPM)
        apply_head_norms(
            cfg.arch,
            layer.attn_q_norm,
            &mut scratch.q,
            cfg.head_count,
            cfg.head_dim,
            eps,
            &mut scratch.dequant_buf,
        );
        apply_head_norms(
            cfg.arch,
            layer.attn_k_norm,
            &mut scratch.k,
            cfg.head_count_kv,
            cfg.head_dim,
            eps,
            &mut scratch.dequant_buf,
        );

        // Apply RoPE rotation to Q and K
        for h in 0..cfg.head_count {
            let q_head = &mut scratch.q[h * cfg.head_dim..(h + 1) * cfg.head_dim];
            rope_head_scalar(
                q_head,
                pos,
                cfg.head_dim,
                cfg.rope_dimension_count,
                cfg.rope_freq_base,
                rope_mode,
                None,
            );
        }

        for h in 0..cfg.head_count_kv {
            let k_head = &mut scratch.k[h * cfg.head_dim..(h + 1) * cfg.head_dim];
            rope_head_scalar(
                k_head,
                pos,
                cfg.head_dim,
                cfg.rope_dimension_count,
                cfg.rope_freq_base,
                rope_mode,
                None,
            );
        }

        // Store K and V in Q8_0 circular ring KV cache
        kv_cache.store_key(l, pos, &scratch.k);
        kv_cache.store_value(l, pos, &scratch.v);

        // Causal GQA decode attention with zero heap allocations ($NOALLOC)
        let context_len = pos + 1;
        causal_gqa_decode_scalar_ext(
            &scratch.q,
            |p| kv_cache.key_ptr(l, p),
            |p| kv_cache.value_ptr(l, p),
            context_len,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.head_dim,
            &mut scratch.scores_buf,
            &mut scratch.dequant_buf,
            &mut scratch.attn_out,
            cfg.attn_logit_softcapping,
        );

        // Attention output projection: Wo · a
        quantize_f32_to_q8_0(&scratch.attn_out, &mut scratch.q8_buf_embd);
        apply_matvec(
            layer.attn_out,
            &scratch.q8_buf_embd,
            &scratch.attn_out,
            &mut scratch.h,
        );

        // Optional post-attention norm (Gemma 2)
        if let Some(post_norm) = layer.post_attn_norm {
            apply_rmsnorm(cfg.arch, post_norm, &scratch.h, eps, &mut scratch.attn_out);
            for i in 0..embd {
                scratch.x[i] += scratch.attn_out[i];
            }
        } else {
            // Residual addition: x = x + Wo · a
            for i in 0..embd {
                scratch.x[i] += scratch.h[i];
            }
        }

        // FFN RMSNorm: h = rmsnorm(x, eps) * ffn_norm
        apply_rmsnorm(cfg.arch, layer.ffn_norm, &scratch.x, eps, &mut scratch.h);

        // Quantize h to Q8_0 for FFN projections
        quantize_f32_to_q8_0(&scratch.h, &mut scratch.q8_buf_embd);

        // FFN projections: gate = Wgate · h, up = Wup · h
        apply_matvec(
            layer.ffn_gate,
            &scratch.q8_buf_embd,
            &scratch.h,
            &mut scratch.ffn_gate,
        );
        apply_matvec(
            layer.ffn_up,
            &scratch.q8_buf_embd,
            &scratch.h,
            &mut scratch.ffn_up,
        );

        // GeGLU for Gemma 2, SwiGLU for others
        if cfg.arch == ModelArch::Gemma2 {
            geglu_scalar(&scratch.ffn_gate, &scratch.ffn_up, &mut scratch.ffn_swiglu);
        } else {
            swiglu_scalar(&scratch.ffn_gate, &scratch.ffn_up, &mut scratch.ffn_swiglu);
        }

        // Down projection: Wdown · FFN_activation
        quantize_f32_to_q8_0(&scratch.ffn_swiglu, &mut scratch.q8_buf_ffn);
        apply_matvec(
            layer.ffn_down,
            &scratch.q8_buf_ffn,
            &scratch.ffn_swiglu,
            &mut scratch.ffn_down,
        );

        // Optional post-FFN norm (Gemma 2)
        if let Some(post_ffn) = layer.post_ffn_norm {
            apply_rmsnorm(cfg.arch, post_ffn, &scratch.ffn_down, eps, &mut scratch.h);
            for i in 0..embd {
                scratch.x[i] += scratch.h[i];
            }
        } else {
            // Residual addition: x = x + ffn_down
            for i in 0..embd {
                scratch.x[i] += scratch.ffn_down[i];
            }
        }
    }

    // 3. Final RMSNorm: x = rmsnorm(x, eps) * output_norm
    apply_rmsnorm(
        cfg.arch,
        tensors.output_norm,
        &scratch.x,
        eps,
        &mut scratch.h,
    );

    // 4. Logits projection: Wout · h
    quantize_f32_to_q8_0(&scratch.h, &mut scratch.q8_buf_embd);
    apply_matvec(
        tensors.output,
        &scratch.q8_buf_embd,
        &scratch.h,
        &mut scratch.logits,
    );

    // Optional final logit softcapping (Gemma 2)
    if let Some(cap) = cfg.final_logit_softcapping {
        logit_softcap_scalar(&mut scratch.logits, cap);
    }

    &scratch.logits
}

pub fn lookup_token_embedding(
    tensor: &LoadedTensor,
    embd_dim: usize,
    token_id: u32,
    out: &mut [f32],
) {
    let tid = token_id as usize;
    match &tensor.data {
        LoadedTensorData::Q4_0 { scales, quants, .. } => {
            let num_blocks = embd_dim / Q4_0_BLOCK_SIZE;
            let scales_stride = num_blocks * 2;
            let quants_stride = num_blocks * 16;

            let row_scales = &scales.as_slice()[tid * scales_stride..(tid + 1) * scales_stride];
            let row_quants = &quants.as_slice()[tid * quants_stride..(tid + 1) * quants_stride];

            dequantize_q4_0_soa(row_scales, row_quants, num_blocks, out);
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
                out[i] = f32::from_le_bytes(bytes);
            }
        }
    }
}

fn add_bias(tensor: Option<&LoadedTensor>, vec: &mut [f32]) {
    if let Some(t) = tensor {
        if let LoadedTensorData::Raw(slice) = &t.data {
            let raw_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const f32, slice.size() / 4)
            };
            let len = vec.len().min(raw_f32.len());
            for i in 0..len {
                vec[i] += raw_f32[i];
            }
        }
    }
}

fn apply_head_norms(
    arch: ModelArch,
    norm_tensor: Option<&LoadedTensor>,
    vec: &mut [f32],
    head_count: usize,
    head_dim: usize,
    eps: f32,
    buf: &mut [f32],
) {
    if let Some(t) = norm_tensor {
        if let LoadedTensorData::Raw(slice) = &t.data {
            let weights: &[f32] = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const f32, slice.size() / 4)
            };
            for h in 0..head_count {
                let head = &mut vec[h * head_dim..(h + 1) * head_dim];
                buf[..head_dim].copy_from_slice(head);
                if arch == ModelArch::Gemma2 {
                    dispatch_rmsnorm_gemma(&buf[..head_dim], weights, eps, head);
                } else {
                    dispatch_rmsnorm(&buf[..head_dim], weights, eps, head);
                }
            }
        }
    }
}

fn apply_rmsnorm(arch: ModelArch, tensor: &LoadedTensor, x: &[f32], eps: f32, out: &mut [f32]) {
    match &tensor.data {
        LoadedTensorData::Raw(slice) => {
            let raw_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const f32, slice.size() / 4)
            };
            if arch == ModelArch::Gemma2 {
                dispatch_rmsnorm_gemma(x, raw_f32, eps, out);
            } else {
                dispatch_rmsnorm(x, raw_f32, eps, out);
            }
        }
        _ => panic!("RMSNorm weight must be float"),
    }
}

fn apply_matvec(tensor: &LoadedTensor, x_q8: &[u8], x_f32: &[f32], out: &mut [f32]) {
    let rows = out.len();
    let cols = x_f32.len();

    match &tensor.data {
        LoadedTensorData::Q4_0 { scales, quants, .. } => {
            dispatch_matvec_q4_0_q8_0(scales.as_slice(), quants.as_slice(), x_q8, rows, cols, out);
        }
        LoadedTensorData::Raw(slice) => {
            if tensor.ggml_type == sulimo_core::GgmlType::F32 {
                let raw_f32: &[f32] = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const f32, slice.size() / 4)
                };
                dispatch_matvec_f32(raw_f32, x_f32, rows, cols, out);
            } else if tensor.ggml_type == sulimo_core::GgmlType::F16 {
                dispatch_matvec_f16(slice.as_slice(), x_f32, rows, cols, out);
            }
        }
    }
}
