//! Full-layer GPU prompt prefill execution pipeline (@C:OCL-GEN9, §7 M3b).
//!
//! Enqueues all layers across a prompt micro-batch on an in-order OpenCL queue,
//! writes Q8_0 KV vectors directly into shared host arena buffers ($ZCOPY),
//! and executes a single OS-level zero-spin wait at the end of all layers.

use crate::buffer::OclBuffer;
use crate::program::{KernelVariant, OclContext, OclError};
use crate::waiter::wait_event_zero_spin;
use opencl3::event::Event;
use opencl3::kernel::ExecuteKernel;
use sulimo_core::model_config::{ModelArch, ModelConfig};

/// Zero-copy OpenCL buffer handles for a single transformer layer.
pub struct OclLayerTensors {
    pub attn_norm: OclBuffer,
    pub attn_q_scales: OclBuffer,
    pub attn_q_quants: OclBuffer,
    pub attn_k_scales: OclBuffer,
    pub attn_k_quants: OclBuffer,
    pub attn_v_scales: OclBuffer,
    pub attn_v_quants: OclBuffer,
    pub attn_out_scales: OclBuffer,
    pub attn_out_quants: OclBuffer,
    pub ffn_norm: OclBuffer,
    pub ffn_gate_scales: OclBuffer,
    pub ffn_gate_quants: OclBuffer,
    pub ffn_up_scales: OclBuffer,
    pub ffn_up_quants: OclBuffer,
    pub ffn_down_scales: OclBuffer,
    pub ffn_down_quants: OclBuffer,
}

/// Zero-copy OpenCL buffer handles for all model weights across all layers.
pub struct OclModelTensors {
    pub output_norm: OclBuffer,
    pub output_scales: OclBuffer,
    pub output_quants: OclBuffer,
    pub layers: Vec<OclLayerTensors>,
}

/// Zero-copy OpenCL handles for the Q8_0 ring KV cache.
pub struct OclKvCache {
    pub k_buffers: Vec<OclBuffer>,
    pub v_buffers: Vec<OclBuffer>,
    pub capacity: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
}

/// Scratch activation buffers on the GPU for a micro-batch of tokens.
pub struct OclPrefillScratch {
    pub x: OclBuffer,
    pub h: OclBuffer,
    pub q: OclBuffer,
    pub k: OclBuffer,
    pub v: OclBuffer,
    pub attn_out: OclBuffer,
    pub ffn_gate: OclBuffer,
    pub ffn_up: OclBuffer,
    pub ffn_swiglu: OclBuffer,
    pub ffn_down: OclBuffer,
    pub logits: OclBuffer,
    pub max_n: usize,
}

/// Enqueues Q4_0 x F16 matrix multiplication without waiting (pipelined).
#[allow(clippy::too_many_arguments)]
pub fn enqueue_gemm_q4_0_f16(
    ctx: &OclContext,
    scales_buf: &OclBuffer,
    quants_buf: &OclBuffer,
    x_buf: &OclBuffer,
    out_buf: &mut OclBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Event, OclError> {
    let kernel = match ctx.active_variant {
        KernelVariant::Intel => ctx.kernel_intel.as_ref().unwrap_or(&ctx.kernel_portable),
        KernelVariant::Portable => &ctx.kernel_portable,
    };

    let m_i32 = m as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;

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
    exec.set_global_work_sizes(&[m, n]);
    exec.set_local_work_sizes(&[8, 1]);

    unsafe {
        exec.enqueue_nd_range(&ctx.queue)
            .map_err(|e| OclError::Driver(format!("enqueue gemm: {e}")))
    }
}

/// Enqueues RMSNorm over FP16 activations with FP32 weights.
pub fn enqueue_rmsnorm(
    ctx: &OclContext,
    x: &OclBuffer,
    w: &OclBuffer,
    out: &mut OclBuffer,
    d: usize,
    eps: f32,
    n: usize,
) -> Result<Event, OclError> {
    let d_i32 = d as i32;
    let n_i32 = n as i32;

    let mut exec = ExecuteKernel::new(&ctx.kernel_rmsnorm);
    unsafe {
        exec.set_arg(&x.cl_mem())
            .set_arg(&w.cl_mem())
            .set_arg(&out.cl_mem())
            .set_arg(&d_i32)
            .set_arg(&eps)
            .set_arg(&n_i32);
    }
    exec.set_global_work_sizes(&[n]);

    unsafe {
        exec.enqueue_nd_range(&ctx.queue)
            .map_err(|e| OclError::Driver(format!("enqueue rmsnorm: {e}")))
    }
}

/// Enqueues RoPE rotation on Q or K.
#[allow(clippy::too_many_arguments)]
pub fn enqueue_rope(
    ctx: &OclContext,
    vec: &mut OclBuffer,
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    mode: i32,
    start_pos: usize,
    n: usize,
) -> Result<Event, OclError> {
    let head_count_i32 = head_count as i32;
    let head_dim_i32 = head_dim as i32;
    let rope_dim_i32 = rope_dim as i32;
    let start_pos_i32 = start_pos as i32;
    let n_i32 = n as i32;

    let mut exec = ExecuteKernel::new(&ctx.kernel_rope);
    unsafe {
        exec.set_arg(&vec.cl_mem())
            .set_arg(&head_count_i32)
            .set_arg(&head_dim_i32)
            .set_arg(&rope_dim_i32)
            .set_arg(&freq_base)
            .set_arg(&mode)
            .set_arg(&start_pos_i32)
            .set_arg(&n_i32);
    }
    exec.set_global_work_sizes(&[n, head_count]);

    unsafe {
        exec.enqueue_nd_range(&ctx.queue)
            .map_err(|e| OclError::Driver(format!("enqueue rope: {e}")))
    }
}

/// Enqueues storing an activation vector into the Q8_0 ring KV cache buffer.
pub fn enqueue_kv_store(
    ctx: &OclContext,
    act: &OclBuffer,
    kv_cache: &mut OclBuffer,
    start_pos: usize,
    capacity: usize,
    blocks_per_pos: usize,
    n: usize,
) -> Result<Event, OclError> {
    let start_pos_i32 = start_pos as i32;
    let capacity_i32 = capacity as i32;
    let blocks_per_pos_i32 = blocks_per_pos as i32;
    let n_i32 = n as i32;

    let mut exec = ExecuteKernel::new(&ctx.kernel_kv_store);
    unsafe {
        exec.set_arg(&act.cl_mem())
            .set_arg(&kv_cache.cl_mem())
            .set_arg(&start_pos_i32)
            .set_arg(&capacity_i32)
            .set_arg(&blocks_per_pos_i32)
            .set_arg(&n_i32);
    }
    exec.set_global_work_sizes(&[n, blocks_per_pos]);

    unsafe {
        exec.enqueue_nd_range(&ctx.queue)
            .map_err(|e| OclError::Driver(format!("enqueue kv_store: {e}")))
    }
}

/// Enqueues causal Grouped Query Attention (GQA) prefill kernel.
#[allow(clippy::too_many_arguments)]
pub fn enqueue_attention_prefill(
    ctx: &OclContext,
    q: &OclBuffer,
    k_cache: &OclBuffer,
    v_cache: &OclBuffer,
    out: &mut OclBuffer,
    start_pos: usize,
    capacity: usize,
    head_count: usize,
    head_count_kv: usize,
    head_dim: usize,
    n: usize,
) -> Result<Event, OclError> {
    let start_pos_i32 = start_pos as i32;
    let capacity_i32 = capacity as i32;
    let head_count_i32 = head_count as i32;
    let head_count_kv_i32 = head_count_kv as i32;
    let head_dim_i32 = head_dim as i32;
    let n_i32 = n as i32;

    let mut exec = ExecuteKernel::new(&ctx.kernel_attention);
    unsafe {
        exec.set_arg(&q.cl_mem())
            .set_arg(&k_cache.cl_mem())
            .set_arg(&v_cache.cl_mem())
            .set_arg(&out.cl_mem())
            .set_arg(&start_pos_i32)
            .set_arg(&capacity_i32)
            .set_arg(&head_count_i32)
            .set_arg(&head_count_kv_i32)
            .set_arg(&head_dim_i32)
            .set_arg(&n_i32);
    }
    exec.set_global_work_sizes(&[n, head_count]);

    unsafe {
        exec.enqueue_nd_range(&ctx.queue)
            .map_err(|e| OclError::Driver(format!("enqueue attention: {e}")))
    }
}

/// Enqueues SwiGLU activation kernel: out = silu(gate) * up.
pub fn enqueue_silu_mul(
    ctx: &OclContext,
    gate: &OclBuffer,
    up: &OclBuffer,
    out: &mut OclBuffer,
    total_elements: usize,
) -> Result<Event, OclError> {
    let total_elements_i32 = total_elements as i32;

    let mut exec = ExecuteKernel::new(&ctx.kernel_silu_mul);
    unsafe {
        exec.set_arg(&gate.cl_mem())
            .set_arg(&up.cl_mem())
            .set_arg(&out.cl_mem())
            .set_arg(&total_elements_i32);
    }
    exec.set_global_work_sizes(&[total_elements]);

    unsafe {
        exec.enqueue_nd_range(&ctx.queue)
            .map_err(|e| OclError::Driver(format!("enqueue silu_mul: {e}")))
    }
}

/// Enqueues elementwise residual addition: a = a + b.
pub fn enqueue_add(
    ctx: &OclContext,
    a: &mut OclBuffer,
    b: &OclBuffer,
    total_elements: usize,
) -> Result<Event, OclError> {
    let total_elements_i32 = total_elements as i32;

    let mut exec = ExecuteKernel::new(&ctx.kernel_add);
    unsafe {
        exec.set_arg(&a.cl_mem())
            .set_arg(&b.cl_mem())
            .set_arg(&total_elements_i32);
    }
    exec.set_global_work_sizes(&[total_elements]);

    unsafe {
        exec.enqueue_nd_range(&ctx.queue)
            .map_err(|e| OclError::Driver(format!("enqueue add: {e}")))
    }
}

/// Enqueues final-position logits projection: logits = x_final · W^T.
#[allow(clippy::too_many_arguments)]
pub fn enqueue_logits_final(
    ctx: &OclContext,
    x_all: &OclBuffer,
    scales: &OclBuffer,
    quants: &OclBuffer,
    logits: &mut OclBuffer,
    vocab_size: usize,
    embd_dim: usize,
    token_idx: usize,
) -> Result<Event, OclError> {
    let m_i32 = vocab_size as i32;
    let k_i32 = embd_dim as i32;
    let token_idx_i32 = token_idx as i32;

    let mut exec = ExecuteKernel::new(&ctx.kernel_logits);
    unsafe {
        exec.set_arg(&x_all.cl_mem())
            .set_arg(&scales.cl_mem())
            .set_arg(&quants.cl_mem())
            .set_arg(&logits.cl_mem())
            .set_arg(&m_i32)
            .set_arg(&k_i32)
            .set_arg(&token_idx_i32);
    }
    exec.set_global_work_sizes(&[vocab_size]);

    unsafe {
        exec.enqueue_nd_range(&ctx.queue)
            .map_err(|e| OclError::Driver(format!("enqueue logits: {e}")))
    }
}

/// Executes full-layer prompt prefill for a micro-batch of $N$ tokens on the GPU (§7 M3b).
///
/// Enqueues all layers on an in-order command queue without blocking, and performs
/// **one** single OS futex wait at the very end ($BATT, A9).
#[allow(clippy::too_many_arguments)]
pub fn execute_prefill_ubatch(
    ctx: &OclContext,
    cfg: &ModelConfig,
    tensors: &OclModelTensors,
    kv_cache: &mut OclKvCache,
    scratch: &mut OclPrefillScratch,
    start_pos: usize,
    n: usize,
) -> Result<(), OclError> {
    let embd = cfg.embedding_length;
    let ffn = cfg.feed_forward_length;
    let head_count = cfg.head_count;
    let head_count_kv = cfg.head_count_kv;
    let head_dim = cfg.head_dim;
    let rope_dim = cfg.rope_dimension_count;
    let freq_base = cfg.rope_freq_base;
    let eps = cfg.rms_epsilon;
    let rope_mode = match cfg.arch {
        ModelArch::Llama => 0, // Norm
        ModelArch::Qwen2 => 1, // Neox
    };

    let blocks_per_pos = (head_count_kv * head_dim) / 32;

    // Loop through all transformer layers, enqueuing kernels consecutively without waiting
    for (l, layer) in tensors.layers.iter().enumerate() {
        // 1. Attention RMSNorm: h = rmsnorm(x, eps) * attn_norm
        enqueue_rmsnorm(
            ctx,
            &scratch.x,
            &layer.attn_norm,
            &mut scratch.h,
            embd,
            eps,
            n,
        )?;

        // 2. Q, K, V projections: W · h
        enqueue_gemm_q4_0_f16(
            ctx,
            &layer.attn_q_scales,
            &layer.attn_q_quants,
            &scratch.h,
            &mut scratch.q,
            head_count * head_dim,
            embd,
            n,
        )?;
        enqueue_gemm_q4_0_f16(
            ctx,
            &layer.attn_k_scales,
            &layer.attn_k_quants,
            &scratch.h,
            &mut scratch.k,
            head_count_kv * head_dim,
            embd,
            n,
        )?;
        enqueue_gemm_q4_0_f16(
            ctx,
            &layer.attn_v_scales,
            &layer.attn_v_quants,
            &scratch.h,
            &mut scratch.v,
            head_count_kv * head_dim,
            embd,
            n,
        )?;

        // 3. RoPE rotation on Q and K
        enqueue_rope(
            ctx,
            &mut scratch.q,
            head_count,
            head_dim,
            rope_dim,
            freq_base,
            rope_mode,
            start_pos,
            n,
        )?;
        enqueue_rope(
            ctx,
            &mut scratch.k,
            head_count_kv,
            head_dim,
            rope_dim,
            freq_base,
            rope_mode,
            start_pos,
            n,
        )?;

        // 4. Store K and V in zero-copy shared Q8_0 ring KV cache ($ZCOPY, @C:COL)
        enqueue_kv_store(
            ctx,
            &scratch.k,
            &mut kv_cache.k_buffers[l],
            start_pos,
            kv_cache.capacity,
            blocks_per_pos,
            n,
        )?;
        enqueue_kv_store(
            ctx,
            &scratch.v,
            &mut kv_cache.v_buffers[l],
            start_pos,
            kv_cache.capacity,
            blocks_per_pos,
            n,
        )?;

        // 5. Causal GQA prefill attention
        enqueue_attention_prefill(
            ctx,
            &scratch.q,
            &kv_cache.k_buffers[l],
            &kv_cache.v_buffers[l],
            &mut scratch.attn_out,
            start_pos,
            kv_cache.capacity,
            head_count,
            head_count_kv,
            head_dim,
            n,
        )?;

        // 6. Attention output projection: Wo · a
        enqueue_gemm_q4_0_f16(
            ctx,
            &layer.attn_out_scales,
            &layer.attn_out_quants,
            &scratch.attn_out,
            &mut scratch.h,
            embd,
            embd,
            n,
        )?;

        // 7. Residual addition: x = x + h
        enqueue_add(ctx, &mut scratch.x, &scratch.h, n * embd)?;

        // 8. FFN RMSNorm: h = rmsnorm(x, eps) * ffn_norm
        enqueue_rmsnorm(
            ctx,
            &scratch.x,
            &layer.ffn_norm,
            &mut scratch.h,
            embd,
            eps,
            n,
        )?;

        // 9. Gate and Up projections: gate = Wgate · h, up = Wup · h
        enqueue_gemm_q4_0_f16(
            ctx,
            &layer.ffn_gate_scales,
            &layer.ffn_gate_quants,
            &scratch.h,
            &mut scratch.ffn_gate,
            ffn,
            embd,
            n,
        )?;
        enqueue_gemm_q4_0_f16(
            ctx,
            &layer.ffn_up_scales,
            &layer.ffn_up_quants,
            &scratch.h,
            &mut scratch.ffn_up,
            ffn,
            embd,
            n,
        )?;

        // 10. SwiGLU: swiglu = silu(gate) * up
        enqueue_silu_mul(
            ctx,
            &scratch.ffn_gate,
            &scratch.ffn_up,
            &mut scratch.ffn_swiglu,
            n * ffn,
        )?;

        // 11. Down projection: ffn_down = Wdown · swiglu
        enqueue_gemm_q4_0_f16(
            ctx,
            &layer.ffn_down_scales,
            &layer.ffn_down_quants,
            &scratch.ffn_swiglu,
            &mut scratch.ffn_down,
            embd,
            ffn,
            n,
        )?;

        // 12. Residual addition: x = x + ffn_down
        enqueue_add(ctx, &mut scratch.x, &scratch.ffn_down, n * embd)?;
    }

    // 13. Output RMSNorm on final layer activations: h = rmsnorm(x, eps) * output_norm
    enqueue_rmsnorm(
        ctx,
        &scratch.x,
        &tensors.output_norm,
        &mut scratch.h,
        embd,
        eps,
        n,
    )?;

    // 14. Final-position logits projection ($REF): logits = h_final · Wout^T
    let final_token_idx = if n > 0 { n - 1 } else { 0 };
    let final_event = enqueue_logits_final(
        ctx,
        &scratch.h,
        &tensors.output_scales,
        &tensors.output_quants,
        &mut scratch.logits,
        cfg.vocab_size,
        embd,
        final_token_idx,
    )?;

    // 15. ONE SINGLE OS-level zero-spin wait at the end of the entire prefill pipeline ($BATT, A9)
    wait_event_zero_spin(&ctx.queue, &final_event)?;

    Ok(())
}
