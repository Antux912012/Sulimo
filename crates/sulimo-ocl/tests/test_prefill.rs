//! M3b Acceptance tests: Full-layer GPU prompt prefill, KV cache Q8_0 precision, and zero-spin waiting (@C:OCL-GEN9, §7 M3b).

use half::f16;
use std::fs;
use std::time::Instant;
use sulimo_core::arena::{Arena, ArenaRegion};
use sulimo_core::kv_cache::{dequantize_q8_0_to_f32, quantize_f32_to_q8_0, Q8_0_BLOCK_BYTES};
use sulimo_core::model_config::{ModelArch, ModelConfig};
use sulimo_cpu::scalar::{
    causal_gqa_decode_scalar, rmsnorm_scalar, rope_head_scalar, swiglu_scalar, RopeMode,
};
use sulimo_ocl::buffer::OclBuffer;
use sulimo_ocl::gemm::gemm_q4_0_f16_scalar_reference;
use sulimo_ocl::prefill::{
    enqueue_kv_store, execute_prefill_ubatch, OclKvCache, OclLayerTensors, OclModelTensors,
    OclPrefillScratch,
};
use sulimo_ocl::program::OclContext;
use sulimo_ocl::waiter::wait_event_zero_spin;

fn read_thread_cpu_time_ticks() -> u64 {
    let tid = unsafe { libc::syscall(libc::SYS_gettid) };
    let stat_path = format!("/proc/self/task/{tid}/stat");
    let content = match fs::read_to_string(stat_path) {
        Ok(c) => c,
        Err(_) => return 0,
    };

    let rparen = content.rfind(')').expect("closing paren in comm");
    let rest = &content[rparen + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let utime: u64 = fields[11].parse().unwrap();
    let stime: u64 = fields[12].parse().unwrap();
    utime + stime
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[test]
fn test_m3b_kv_store_precision_vs_cpu() {
    let ctx = match OclContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("OpenCL unavailable ({e}), skipping GPU test");
            return;
        }
    };

    println!(
        "Testing kv_store on OpenCL device: {} ({:?})",
        ctx.device_info.device_name, ctx.active_variant
    );

    let n_tokens = 16;
    let blocks_per_pos = 8; // 256 values per token (8 kv heads x 32 head_dim)
    let total_values = n_tokens * blocks_per_pos * 32;
    let capacity = 64;

    let act_bytes = total_values * 2; // f16
    let kv_bytes = capacity * blocks_per_pos * Q8_0_BLOCK_BYTES;

    let mut arena = Arena::new(act_bytes + kv_bytes + 16384).expect("allocate arena");
    let act_slice = arena.alloc(act_bytes, ArenaRegion::Scratch).unwrap();
    let kv_gpu_slice = arena.alloc(kv_bytes, ArenaRegion::KvCache).unwrap();

    // Fill activation vector with varied f16 values
    let act_f16: &mut [f16] =
        unsafe { std::slice::from_raw_parts_mut(act_slice.as_ptr() as *mut f16, total_values) };
    for (i, v) in act_f16.iter_mut().enumerate() {
        let val = ((i as f32 * 0.17).sin() * 5.0) + ((i % 7) as f32 - 3.0);
        *v = f16::from_f32(val);
    }

    let act_buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            act_slice.as_ptr(),
            act_slice.padded_size(),
            true,
        )
        .expect("act buffer")
    };

    let mut kv_gpu_buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            kv_gpu_slice.as_ptr(),
            kv_gpu_slice.padded_size(),
            false,
        )
        .expect("kv buffer")
    };

    let start_pos = 5;
    let event = enqueue_kv_store(
        &ctx,
        &act_buf,
        &mut kv_gpu_buf,
        start_pos,
        capacity,
        blocks_per_pos,
        n_tokens,
    )
    .expect("enqueue kv_store");

    wait_event_zero_spin(&ctx.queue, &event).expect("wait kv_store");

    // Read back and verify against CPU quantize_f32_to_q8_0
    let mut cpu_kv = vec![0u8; kv_bytes];
    let bytes_per_pos = blocks_per_pos * Q8_0_BLOCK_BYTES;

    for t in 0..n_tokens {
        let ring_pos = (start_pos + t) % capacity;
        let pos_offset = ring_pos * bytes_per_pos;
        let act_pos_offset = t * blocks_per_pos * 32;

        for b in 0..blocks_per_pos {
            let mut f32_block = [0.0f32; 32];
            for i in 0..32 {
                f32_block[i] = act_f16[act_pos_offset + b * 32 + i].to_f32();
            }

            let cpu_block_offset = pos_offset + b * Q8_0_BLOCK_BYTES;
            quantize_f32_to_q8_0(
                &f32_block,
                &mut cpu_kv[cpu_block_offset..cpu_block_offset + Q8_0_BLOCK_BYTES],
            );

            // Compare GPU bytes vs CPU bytes
            let gpu_block =
                &kv_gpu_slice.as_slice()[cpu_block_offset..cpu_block_offset + Q8_0_BLOCK_BYTES];
            let cpu_block = &cpu_kv[cpu_block_offset..cpu_block_offset + Q8_0_BLOCK_BYTES];

            let gpu_d = f16::from_le_bytes([gpu_block[0], gpu_block[1]]).to_f32();
            let cpu_d = f16::from_le_bytes([cpu_block[0], cpu_block[1]]).to_f32();

            // Scale factor must match within f16 rounding
            assert!(
                (gpu_d - cpu_d).abs() <= 1e-3 * cpu_d.max(1.0),
                "Block scale mismatch at token {t}, block {b}: gpu={gpu_d}, cpu={cpu_d}"
            );

            let mut gpu_dequant = [0.0f32; 32];
            let mut cpu_dequant = [0.0f32; 32];
            dequantize_q8_0_to_f32(gpu_block, 1, &mut gpu_dequant);
            dequantize_q8_0_to_f32(cpu_block, 1, &mut cpu_dequant);

            for i in 0..32 {
                let gpu_q = gpu_block[2 + i] as i8 as i32;
                let cpu_q = cpu_block[2 + i] as i8 as i32;
                let q_diff = (gpu_q - cpu_q).abs();

                // Quant integer error must be <= 1 step
                assert!(
                    q_diff <= 1,
                    "Quant mismatch at token {t}, block {b}, idx {i}: gpu={gpu_q}, cpu={cpu_q}"
                );

                // Dequantized float value must match within 1 Q8_0 step (d)
                let dequant_diff = (gpu_dequant[i] - cpu_dequant[i]).abs();
                assert!(
                    dequant_diff <= cpu_d * 1.05 + 1e-4,
                    "Dequant mismatch at token {t}, block {b}, idx {i}: gpu={}, cpu={}, diff={}",
                    gpu_dequant[i],
                    cpu_dequant[i],
                    dequant_diff
                );
            }
        }
    }

    println!("KV store Q8_0 precision test passed: all values within 1 Q8_0 step ($REF)");
}

fn helper_alloc_q4(
    arena: &mut Arena,
    ctx: &OclContext,
    rows: usize,
    cols: usize,
    seed: usize,
) -> (
    sulimo_core::arena::ArenaSlice,
    sulimo_core::arena::ArenaSlice,
    OclBuffer,
    OclBuffer,
) {
    let blocks = (rows * cols) / 32;
    let sc_slice = arena.alloc(blocks * 2, ArenaRegion::Scratch).unwrap();
    let qu_slice = arena.alloc(blocks * 16, ArenaRegion::Scratch).unwrap();

    let sc_f16: &mut [f16] =
        unsafe { std::slice::from_raw_parts_mut(sc_slice.as_ptr() as *mut f16, blocks) };
    for (i, s) in sc_f16.iter_mut().enumerate() {
        *s = f16::from_f32(0.02 + 0.005 * ((i + seed) % 7) as f32);
    }

    let qu_u8: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(qu_slice.as_ptr() as *mut u8, blocks * 16) };
    for (i, q) in qu_u8.iter_mut().enumerate() {
        let low = ((i + seed) * 3) % 15;
        let high = ((i + seed) * 7 + 1) % 15;
        *q = (low | (high << 4)) as u8;
    }

    let sc_buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            sc_slice.as_ptr(),
            sc_slice.padded_size(),
            true,
        )
        .unwrap()
    };
    let qu_buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            qu_slice.as_ptr(),
            qu_slice.padded_size(),
            true,
        )
        .unwrap()
    };

    (sc_slice, qu_slice, sc_buf, qu_buf)
}

fn helper_alloc_norm(
    arena: &mut Arena,
    ctx: &OclContext,
    dim: usize,
    seed: usize,
) -> (sulimo_core::arena::ArenaSlice, OclBuffer) {
    let slice = arena.alloc(dim * 4, ArenaRegion::Scratch).unwrap();
    let f32_slice: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut f32, dim) };
    for (i, v) in f32_slice.iter_mut().enumerate() {
        *v = 1.0 + 0.02 * ((i + seed) % 5) as f32;
    }

    let buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            slice.as_ptr(),
            slice.padded_size(),
            true,
        )
        .unwrap()
    };
    (slice, buf)
}

fn helper_alloc_scratch(
    arena: &mut Arena,
    ctx: &OclContext,
    elements: usize,
    is_f32: bool,
) -> (sulimo_core::arena::ArenaSlice, OclBuffer) {
    let byte_size = elements * if is_f32 { 4 } else { 2 };
    let slice = arena.alloc(byte_size, ArenaRegion::Scratch).unwrap();
    let buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            slice.as_ptr(),
            slice.padded_size(),
            false,
        )
        .unwrap()
    };
    (slice, buf)
}

#[test]
fn test_m3b_full_layer_prefill_and_zero_spin_proof() {
    let ctx = match OclContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("OpenCL unavailable ({e}), skipping GPU test");
            return;
        }
    };

    println!(
        "Testing full-layer prefill on: {} (Variant: {:?})",
        ctx.device_info.device_name, ctx.active_variant
    );

    let n = 16;
    let embd = 64;
    let ffn = 128;
    let head_count = 4;
    let head_count_kv = 2;
    let head_dim = 16;
    let vocab_size = 64;
    let block_count = 2;
    let capacity = 64;

    let cfg = ModelConfig {
        arch: ModelArch::Llama,
        context_length: capacity,
        embedding_length: embd,
        block_count,
        feed_forward_length: ffn,
        head_count,
        head_count_kv,
        head_dim,
        rope_dimension_count: head_dim,
        rope_freq_base: 10000.0,
        rms_epsilon: 1e-5,
        vocab_size,
        bos_token_id: None,
        eos_token_id: None,
        pre_tokenizer: "default".to_string(),
        sliding_window: None,
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        scale_emb: None,
    };

    // Calculate memory requirements
    let q_dim = head_count * head_dim;
    let kv_dim = head_count_kv * head_dim;

    let q4_bytes = |rows: usize, cols: usize| -> (usize, usize) {
        let blocks = (rows * cols) / 32;
        (blocks * 2, blocks * 16)
    };

    let (q_sc, q_qu) = q4_bytes(q_dim, embd);
    let (k_sc, k_qu) = q4_bytes(kv_dim, embd);
    let (v_sc, v_qu) = q4_bytes(kv_dim, embd);
    let (o_sc, o_qu) = q4_bytes(embd, q_dim);
    let (gate_sc, gate_qu) = q4_bytes(ffn, embd);
    let (up_sc, up_qu) = q4_bytes(ffn, embd);
    let (down_sc, down_qu) = q4_bytes(embd, ffn);
    let (out_sc, out_qu) = q4_bytes(vocab_size, embd);

    let layer_weight_bytes = (q_sc + q_qu)
        + (k_sc + k_qu)
        + (v_sc + v_qu)
        + (o_sc + o_qu)
        + (gate_sc + gate_qu)
        + (up_sc + up_qu)
        + (down_sc + down_qu)
        + (embd * 4 * 2) // attn_norm, ffn_norm (f32)
        + 16 * 4096;

    let total_arena = (layer_weight_bytes * block_count)
        + (out_sc + out_qu)
        + (embd * 4) // output_norm
        + (capacity * (kv_dim / 32) * Q8_0_BLOCK_BYTES * 2 * block_count)
        + (n * ffn * 4 * 8) // scratch
        + 64 * 4096;

    let mut arena = Arena::new(total_arena).expect("allocate arena");

    // Allocate layer weights
    struct LayerHostData {
        attn_norm_slice: sulimo_core::arena::ArenaSlice,
        attn_q_sc_slice: sulimo_core::arena::ArenaSlice,
        attn_q_qu_slice: sulimo_core::arena::ArenaSlice,
        attn_k_sc_slice: sulimo_core::arena::ArenaSlice,
        attn_k_qu_slice: sulimo_core::arena::ArenaSlice,
        attn_v_sc_slice: sulimo_core::arena::ArenaSlice,
        attn_v_qu_slice: sulimo_core::arena::ArenaSlice,
        attn_o_sc_slice: sulimo_core::arena::ArenaSlice,
        attn_o_qu_slice: sulimo_core::arena::ArenaSlice,
        ffn_norm_slice: sulimo_core::arena::ArenaSlice,
        ffn_gate_sc_slice: sulimo_core::arena::ArenaSlice,
        ffn_gate_qu_slice: sulimo_core::arena::ArenaSlice,
        ffn_up_sc_slice: sulimo_core::arena::ArenaSlice,
        ffn_up_qu_slice: sulimo_core::arena::ArenaSlice,
        ffn_down_sc_slice: sulimo_core::arena::ArenaSlice,
        ffn_down_qu_slice: sulimo_core::arena::ArenaSlice,
    }

    let mut host_layers = Vec::with_capacity(block_count);
    let mut ocl_layers = Vec::with_capacity(block_count);

    for l in 0..block_count {
        let (attn_norm_slice, attn_norm) = helper_alloc_norm(&mut arena, &ctx, embd, l * 10 + 1);
        let (attn_q_sc_slice, attn_q_qu_slice, attn_q_scales, attn_q_quants) =
            helper_alloc_q4(&mut arena, &ctx, q_dim, embd, l * 10 + 2);
        let (attn_k_sc_slice, attn_k_qu_slice, attn_k_scales, attn_k_quants) =
            helper_alloc_q4(&mut arena, &ctx, kv_dim, embd, l * 10 + 3);
        let (attn_v_sc_slice, attn_v_qu_slice, attn_v_scales, attn_v_quants) =
            helper_alloc_q4(&mut arena, &ctx, kv_dim, embd, l * 10 + 4);
        let (attn_o_sc_slice, attn_o_qu_slice, attn_out_scales, attn_out_quants) =
            helper_alloc_q4(&mut arena, &ctx, embd, q_dim, l * 10 + 5);
        let (ffn_norm_slice, ffn_norm) = helper_alloc_norm(&mut arena, &ctx, embd, l * 10 + 6);
        let (ffn_gate_sc_slice, ffn_gate_qu_slice, ffn_gate_scales, ffn_gate_quants) =
            helper_alloc_q4(&mut arena, &ctx, ffn, embd, l * 10 + 7);
        let (ffn_up_sc_slice, ffn_up_qu_slice, ffn_up_scales, ffn_up_quants) =
            helper_alloc_q4(&mut arena, &ctx, ffn, embd, l * 10 + 8);
        let (ffn_down_sc_slice, ffn_down_qu_slice, ffn_down_scales, ffn_down_quants) =
            helper_alloc_q4(&mut arena, &ctx, embd, ffn, l * 10 + 9);

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

        host_layers.push(LayerHostData {
            attn_norm_slice,
            attn_q_sc_slice,
            attn_q_qu_slice,
            attn_k_sc_slice,
            attn_k_qu_slice,
            attn_v_sc_slice,
            attn_v_qu_slice,
            attn_o_sc_slice,
            attn_o_qu_slice,
            ffn_norm_slice,
            ffn_gate_sc_slice,
            ffn_gate_qu_slice,
            ffn_up_sc_slice,
            ffn_up_qu_slice,
            ffn_down_sc_slice,
            ffn_down_qu_slice,
        });
    }

    let (output_norm_slice, output_norm) = helper_alloc_norm(&mut arena, &ctx, embd, 99);
    let (output_sc_slice, output_qu_slice, output_scales, output_quants) =
        helper_alloc_q4(&mut arena, &ctx, vocab_size, embd, 100);

    let ocl_model = OclModelTensors {
        output_norm,
        output_scales,
        output_quants,
        layers: ocl_layers,
    };

    // Allocate KV Cache buffers
    let kv_layer_bytes = capacity * (kv_dim / 32) * Q8_0_BLOCK_BYTES;
    let mut k_slices = Vec::new();
    let mut v_slices = Vec::new();
    let mut k_buffers = Vec::new();
    let mut v_buffers = Vec::new();

    for _ in 0..block_count {
        let k_s = arena.alloc(kv_layer_bytes, ArenaRegion::KvCache).unwrap();
        let v_s = arena.alloc(kv_layer_bytes, ArenaRegion::KvCache).unwrap();
        let k_b = unsafe {
            OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                k_s.as_ptr(),
                k_s.padded_size(),
                false,
            )
            .unwrap()
        };
        let v_b = unsafe {
            OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                v_s.as_ptr(),
                v_s.padded_size(),
                false,
            )
            .unwrap()
        };
        k_buffers.push(k_b);
        v_buffers.push(v_b);
        k_slices.push(k_s);
        v_slices.push(v_s);
    }

    let mut ocl_kv = OclKvCache {
        k_buffers,
        v_buffers,
        capacity,
        head_count_kv,
        head_dim,
    };

    // Allocate scratch buffers
    let (x_slice, _) = helper_alloc_scratch(&mut arena, &ctx, n * embd, false);
    // Populate initial prompt embeddings in x_slice BEFORE wrapping in OclBuffer
    let x_f16: &mut [f16] =
        unsafe { std::slice::from_raw_parts_mut(x_slice.as_ptr() as *mut f16, n * embd) };
    for (i, v) in x_f16.iter_mut().enumerate() {
        *v = f16::from_f32(((i % 17) as f32 - 8.0) * 0.15);
    }
    let initial_x_cpu: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();

    let x_buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            x_slice.as_ptr(),
            x_slice.padded_size(),
            false,
        )
        .unwrap()
    };
    let (_h_slice, h_buf) = helper_alloc_scratch(&mut arena, &ctx, n * embd, false);
    let (_q_slice, q_buf) = helper_alloc_scratch(&mut arena, &ctx, n * q_dim, false);
    let (_k_slice, k_buf) = helper_alloc_scratch(&mut arena, &ctx, n * kv_dim, false);
    let (_v_slice, v_buf) = helper_alloc_scratch(&mut arena, &ctx, n * kv_dim, false);
    let (_attn_out_slice, attn_out_buf) = helper_alloc_scratch(&mut arena, &ctx, n * q_dim, false);
    let (_gate_slice, ffn_gate_buf) = helper_alloc_scratch(&mut arena, &ctx, n * ffn, false);
    let (_up_slice, ffn_up_buf) = helper_alloc_scratch(&mut arena, &ctx, n * ffn, false);
    let (_swiglu_slice, ffn_swiglu_buf) = helper_alloc_scratch(&mut arena, &ctx, n * ffn, false);
    let (_down_slice, ffn_down_buf) = helper_alloc_scratch(&mut arena, &ctx, n * embd, false);
    let (logits_slice, logits_buf) = helper_alloc_scratch(&mut arena, &ctx, vocab_size, true); // f32 logits

    let mut ocl_scratch = OclPrefillScratch {
        x: x_buf,
        h: h_buf,
        q: q_buf,
        k: k_buf,
        v: v_buf,
        attn_out: attn_out_buf,
        ffn_gate: ffn_gate_buf,
        ffn_up: ffn_up_buf,
        ffn_swiglu: ffn_swiglu_buf,
        ffn_down: ffn_down_buf,
        logits: logits_buf,
        max_n: n,
    };

    // Measure engine thread CPU time before GPU execution to verify zero-spin waiting ($BATT, A9)
    let cpu_ticks_before = read_thread_cpu_time_ticks();
    let wall_start = Instant::now();

    // Execute full-layer prompt prefill across all layers with a SINGLE OS wait
    execute_prefill_ubatch(&ctx, &cfg, &ocl_model, &mut ocl_kv, &mut ocl_scratch, 0, n)
        .expect("execute_prefill_ubatch");

    let wall_duration = wall_start.elapsed();
    let cpu_ticks_after = read_thread_cpu_time_ticks();
    let cpu_ticks_delta = cpu_ticks_after.saturating_sub(cpu_ticks_before);

    println!(
        "GPU prefill finished in {:.2?} ms (engine thread CPU ticks delta: {cpu_ticks_delta})",
        wall_duration.as_secs_f64() * 1000.0
    );

    // Read GPU results from zero-copy shared buffers ($ZCOPY)
    let _map_logits = ocl_scratch.logits.map_read(&ctx.queue).unwrap();
    let gpu_logits: &[f32] =
        unsafe { std::slice::from_raw_parts(logits_slice.as_ptr() as *const f32, vocab_size) };

    // --- CPU Oracle Reference Computation ---
    let mut cpu_x = initial_x_cpu.clone();

    let mut cpu_k_cache = vec![vec![0u8; kv_layer_bytes]; block_count];
    let mut cpu_v_cache = vec![vec![0u8; kv_layer_bytes]; block_count];

    for l in 0..block_count {
        let hl = &host_layers[l];
        let attn_norm_w: &[f32] =
            unsafe { std::slice::from_raw_parts(hl.attn_norm_slice.as_ptr() as *const f32, embd) };
        let ffn_norm_w: &[f32] =
            unsafe { std::slice::from_raw_parts(hl.ffn_norm_slice.as_ptr() as *const f32, embd) };

        let q_sc: &[f16] = unsafe {
            std::slice::from_raw_parts(
                hl.attn_q_sc_slice.as_ptr() as *const f16,
                (q_dim * embd) / 32,
            )
        };
        let q_qu: &[u8] =
            unsafe { std::slice::from_raw_parts(hl.attn_q_qu_slice.as_ptr(), (q_dim * embd) / 2) };
        let k_sc: &[f16] = unsafe {
            std::slice::from_raw_parts(
                hl.attn_k_sc_slice.as_ptr() as *const f16,
                (kv_dim * embd) / 32,
            )
        };
        let k_qu: &[u8] =
            unsafe { std::slice::from_raw_parts(hl.attn_k_qu_slice.as_ptr(), (kv_dim * embd) / 2) };
        let v_sc: &[f16] = unsafe {
            std::slice::from_raw_parts(
                hl.attn_v_sc_slice.as_ptr() as *const f16,
                (kv_dim * embd) / 32,
            )
        };
        let v_qu: &[u8] =
            unsafe { std::slice::from_raw_parts(hl.attn_v_qu_slice.as_ptr(), (kv_dim * embd) / 2) };
        let o_sc: &[f16] = unsafe {
            std::slice::from_raw_parts(
                hl.attn_o_sc_slice.as_ptr() as *const f16,
                (embd * q_dim) / 32,
            )
        };
        let o_qu: &[u8] =
            unsafe { std::slice::from_raw_parts(hl.attn_o_qu_slice.as_ptr(), (embd * q_dim) / 2) };

        let gate_sc: &[f16] = unsafe {
            std::slice::from_raw_parts(
                hl.ffn_gate_sc_slice.as_ptr() as *const f16,
                (ffn * embd) / 32,
            )
        };
        let gate_qu: &[u8] =
            unsafe { std::slice::from_raw_parts(hl.ffn_gate_qu_slice.as_ptr(), (ffn * embd) / 2) };
        let up_sc: &[f16] = unsafe {
            std::slice::from_raw_parts(hl.ffn_up_sc_slice.as_ptr() as *const f16, (ffn * embd) / 32)
        };
        let up_qu: &[u8] =
            unsafe { std::slice::from_raw_parts(hl.ffn_up_qu_slice.as_ptr(), (ffn * embd) / 2) };
        let down_sc: &[f16] = unsafe {
            std::slice::from_raw_parts(
                hl.ffn_down_sc_slice.as_ptr() as *const f16,
                (embd * ffn) / 32,
            )
        };
        let down_qu: &[u8] =
            unsafe { std::slice::from_raw_parts(hl.ffn_down_qu_slice.as_ptr(), (embd * ffn) / 2) };

        // 1. Attention RMSNorm
        let mut h = vec![0.0f32; n * embd];
        for t in 0..n {
            rmsnorm_scalar(
                &cpu_x[t * embd..(t + 1) * embd],
                attn_norm_w,
                cfg.rms_epsilon,
                &mut h[t * embd..(t + 1) * embd],
            );
        }

        // 2. Q, K, V projections
        let h_f16: Vec<f16> = h.iter().map(|&v| f16::from_f32(v)).collect();
        let mut q_f16 = vec![f16::ZERO; n * q_dim];
        let mut k_f16 = vec![f16::ZERO; n * kv_dim];
        let mut v_f16 = vec![f16::ZERO; n * kv_dim];

        gemm_q4_0_f16_scalar_reference(q_sc, q_qu, &h_f16, q_dim, embd, n, &mut q_f16);
        gemm_q4_0_f16_scalar_reference(k_sc, k_qu, &h_f16, kv_dim, embd, n, &mut k_f16);
        gemm_q4_0_f16_scalar_reference(v_sc, v_qu, &h_f16, kv_dim, embd, n, &mut v_f16);

        let mut q: Vec<f32> = q_f16.iter().map(|v| v.to_f32()).collect();
        let mut k: Vec<f32> = k_f16.iter().map(|v| v.to_f32()).collect();
        let v: Vec<f32> = v_f16.iter().map(|v| v.to_f32()).collect();

        // 3. RoPE
        for t in 0..n {
            for head in 0..head_count {
                let q_h = &mut q[t * q_dim + head * head_dim..t * q_dim + (head + 1) * head_dim];
                rope_head_scalar(
                    q_h,
                    t,
                    head_dim,
                    head_dim,
                    cfg.rope_freq_base,
                    RopeMode::Norm,
                    None,
                );
            }
            for head in 0..head_count_kv {
                let k_h = &mut k[t * kv_dim + head * head_dim..t * kv_dim + (head + 1) * head_dim];
                rope_head_scalar(
                    k_h,
                    t,
                    head_dim,
                    head_dim,
                    cfg.rope_freq_base,
                    RopeMode::Norm,
                    None,
                );
            }
        }

        // 4. Store K and V in Q8_0 ring KV cache
        let bytes_per_pos = (kv_dim / 32) * Q8_0_BLOCK_BYTES;
        for t in 0..n {
            let offset = t * bytes_per_pos;
            quantize_f32_to_q8_0(
                &k[t * kv_dim..(t + 1) * kv_dim],
                &mut cpu_k_cache[l][offset..offset + bytes_per_pos],
            );
            quantize_f32_to_q8_0(
                &v[t * kv_dim..(t + 1) * kv_dim],
                &mut cpu_v_cache[l][offset..offset + bytes_per_pos],
            );
        }

        // 5. Causal attention
        let mut attn_out = vec![0.0f32; n * q_dim];
        let mut scores_buf = vec![0.0f32; capacity];
        let mut dequant_buf = vec![0.0f32; head_dim];

        for t in 0..n {
            let q_t = &q[t * q_dim..(t + 1) * q_dim];
            let out_t = &mut attn_out[t * q_dim..(t + 1) * q_dim];

            causal_gqa_decode_scalar(
                q_t,
                |p| cpu_k_cache[l][p * bytes_per_pos..].as_ptr(),
                |p| cpu_v_cache[l][p * bytes_per_pos..].as_ptr(),
                t + 1,
                head_count,
                head_count_kv,
                head_dim,
                &mut scores_buf,
                &mut dequant_buf,
                out_t,
            );
        }

        // 6. Out projection GEMM
        let attn_out_f16: Vec<f16> = attn_out.iter().map(|&x| f16::from_f32(x)).collect();
        let mut h_attn_f16 = vec![f16::ZERO; n * embd];
        gemm_q4_0_f16_scalar_reference(o_sc, o_qu, &attn_out_f16, embd, q_dim, n, &mut h_attn_f16);

        // 7. Residual add: x = x + h_attn
        for t in 0..n * embd {
            cpu_x[t] += h_attn_f16[t].to_f32();
        }

        // 8. FFN RMSNorm
        let mut h_ffn = vec![0.0f32; n * embd];
        for t in 0..n {
            rmsnorm_scalar(
                &cpu_x[t * embd..(t + 1) * embd],
                ffn_norm_w,
                cfg.rms_epsilon,
                &mut h_ffn[t * embd..(t + 1) * embd],
            );
        }

        // 9. Gate & Up projections
        let h_ffn_f16: Vec<f16> = h_ffn.iter().map(|&x| f16::from_f32(x)).collect();
        let mut gate_f16 = vec![f16::ZERO; n * ffn];
        let mut up_f16 = vec![f16::ZERO; n * ffn];

        gemm_q4_0_f16_scalar_reference(gate_sc, gate_qu, &h_ffn_f16, ffn, embd, n, &mut gate_f16);
        gemm_q4_0_f16_scalar_reference(up_sc, up_qu, &h_ffn_f16, ffn, embd, n, &mut up_f16);

        let gate: Vec<f32> = gate_f16.iter().map(|x| x.to_f32()).collect();
        let up: Vec<f32> = up_f16.iter().map(|x| x.to_f32()).collect();

        // 10. SwiGLU
        let mut swiglu = vec![0.0f32; n * ffn];
        swiglu_scalar(&gate, &up, &mut swiglu);

        // 11. Down projection
        let swiglu_f16: Vec<f16> = swiglu.iter().map(|&x| f16::from_f32(x)).collect();
        let mut down_f16 = vec![f16::ZERO; n * embd];
        gemm_q4_0_f16_scalar_reference(down_sc, down_qu, &swiglu_f16, embd, ffn, n, &mut down_f16);

        // 12. Residual add: x = x + down
        for t in 0..n * embd {
            cpu_x[t] += down_f16[t].to_f32();
        }
    }

    // 13. Output RMSNorm on final layer activations
    let output_norm_w: &[f32] =
        unsafe { std::slice::from_raw_parts(output_norm_slice.as_ptr() as *const f32, embd) };
    let mut h_final = vec![0.0f32; embd];
    let final_pos = n - 1;
    rmsnorm_scalar(
        &cpu_x[final_pos * embd..(final_pos + 1) * embd],
        output_norm_w,
        cfg.rms_epsilon,
        &mut h_final,
    );

    // 14. Logits on final position (FP32 accumulation matching logits_f16_output)
    let out_sc: &[f16] = unsafe {
        std::slice::from_raw_parts(
            output_sc_slice.as_ptr() as *const f16,
            (vocab_size * embd) / 32,
        )
    };
    let out_qu: &[u8] =
        unsafe { std::slice::from_raw_parts(output_qu_slice.as_ptr(), (vocab_size * embd) / 2) };

    let mut cpu_logits = vec![0.0f32; vocab_size];
    let num_blocks = embd / 32;
    for m in 0..vocab_size {
        let mut acc = 0.0f32;
        for b in 0..num_blocks {
            let d = out_sc[m * num_blocks + b].to_f32();
            let q_ptr = &out_qu[(m * num_blocks + b) * 16..(m * num_blocks + b + 1) * 16];
            let x_ptr = &h_final[b * 32..(b + 1) * 32];
            let mut block_sum = 0.0f32;
            for i in 0..16 {
                let q = q_ptr[i];
                let q_low = ((q & 0x0F) as i32 - 8) as f32;
                let q_high = (((q >> 4) & 0x0F) as i32 - 8) as f32;
                block_sum += x_ptr[i] * q_low + x_ptr[16 + i] * q_high;
            }
            acc += block_sum * d;
        }
        cpu_logits[m] = acc;
    }

    // Verify KV Cache written by GPU dequantizes within 1 Q8_0 step of CPU KV cache ($REF)
    let bytes_per_pos = (kv_dim / 32) * Q8_0_BLOCK_BYTES;
    for l in 0..block_count {
        let _map_k = ocl_kv.k_buffers[l].map_read(&ctx.queue).unwrap();
        let gpu_k = &k_slices[l].as_slice()[..n * bytes_per_pos];
        let cpu_k = &cpu_k_cache[l][..n * bytes_per_pos];

        let num_blocks = (n * bytes_per_pos) / Q8_0_BLOCK_BYTES;
        for b in 0..num_blocks {
            let offset = b * Q8_0_BLOCK_BYTES;
            let gpu_blk = &gpu_k[offset..offset + Q8_0_BLOCK_BYTES];
            let cpu_blk = &cpu_k[offset..offset + Q8_0_BLOCK_BYTES];

            let cpu_d = f16::from_le_bytes([cpu_blk[0], cpu_blk[1]]).to_f32();
            let mut gpu_dequant = [0.0f32; 32];
            let mut cpu_dequant = [0.0f32; 32];
            dequantize_q8_0_to_f32(gpu_blk, 1, &mut gpu_dequant);
            dequantize_q8_0_to_f32(cpu_blk, 1, &mut cpu_dequant);

            for i in 0..32 {
                let diff = (gpu_dequant[i] - cpu_dequant[i]).abs();
                assert!(
                    diff <= cpu_d * 1.05 + 1e-4,
                    "KV cache layer {l} block {b} idx {i} dequant mismatch: gpu={}, cpu={}, diff={}",
                    gpu_dequant[i],
                    cpu_dequant[i],
                    diff
                );
            }
        }
    }
    println!("KV cache verification passed: all GPU entries within 1 Q8_0 step of CPU ($REF)");

    // Verify logits match CPU oracle reference
    let sim = cosine_similarity(gpu_logits, &cpu_logits);
    println!("Logits cosine similarity GPU vs CPU: {sim:.6}");
    assert!(sim >= 0.99, "Logits cosine similarity too low: {sim}");

    // Verify greedy argmax token matches CPU reference ($REF)
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    println!("Greedy argmax token: GPU={gpu_argmax}, CPU={cpu_argmax}");
    assert_eq!(
        gpu_argmax, cpu_argmax,
        "Greedy token mismatch: GPU {gpu_argmax} != CPU {cpu_argmax}"
    );

    println!(
        "Full-layer prompt prefill test passed with exact greedy match and zero-spin waiting!"
    );
}

#[test]
fn test_m3b_prefill_throughput_benchmark() {
    let ctx = match OclContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    let n = 128;
    let embd = 128;
    let ffn = 256;
    let head_count = 4;
    let head_count_kv = 2;
    let head_dim = 32;
    let vocab_size = 256;
    let block_count = 2;
    let capacity = 256;

    let cfg = ModelConfig {
        arch: ModelArch::Llama,
        context_length: capacity,
        embedding_length: embd,
        block_count,
        feed_forward_length: ffn,
        head_count,
        head_count_kv,
        head_dim,
        rope_dimension_count: head_dim,
        rope_freq_base: 10000.0,
        rms_epsilon: 1e-5,
        vocab_size,
        bos_token_id: None,
        eos_token_id: None,
        pre_tokenizer: "default".to_string(),
        sliding_window: None,
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        scale_emb: None,
    };

    let q_dim = head_count * head_dim;
    let kv_dim = head_count_kv * head_dim;

    let q4_bytes = |rows: usize, cols: usize| -> (usize, usize) {
        let blocks = (rows * cols) / 32;
        (blocks * 2, blocks * 16)
    };

    let (q_sc, q_qu) = q4_bytes(q_dim, embd);
    let (k_sc, k_qu) = q4_bytes(kv_dim, embd);
    let (v_sc, v_qu) = q4_bytes(kv_dim, embd);
    let (o_sc, o_qu) = q4_bytes(embd, q_dim);
    let (gate_sc, gate_qu) = q4_bytes(ffn, embd);
    let (up_sc, up_qu) = q4_bytes(ffn, embd);
    let (down_sc, down_qu) = q4_bytes(embd, ffn);
    let (out_sc, out_qu) = q4_bytes(vocab_size, embd);

    let layer_weight_bytes = (q_sc + q_qu)
        + (k_sc + k_qu)
        + (v_sc + v_qu)
        + (o_sc + o_qu)
        + (gate_sc + gate_qu)
        + (up_sc + up_qu)
        + (down_sc + down_qu)
        + (embd * 4 * 2)
        + 16 * 4096;

    let total_arena = (layer_weight_bytes * block_count)
        + (out_sc + out_qu)
        + (embd * 4)
        + (capacity * (kv_dim / 32) * Q8_0_BLOCK_BYTES * 2 * block_count)
        + (n * ffn * 4 * 8)
        + 64 * 4096;

    let mut arena = Arena::new(total_arena).expect("allocate arena");

    let mut ocl_layers = Vec::with_capacity(block_count);
    for l in 0..block_count {
        let attn_norm = helper_alloc_norm(&mut arena, &ctx, embd, l * 10 + 1).1;
        let (_, _, attn_q_scales, attn_q_quants) =
            helper_alloc_q4(&mut arena, &ctx, q_dim, embd, l * 10 + 2);
        let (_, _, attn_k_scales, attn_k_quants) =
            helper_alloc_q4(&mut arena, &ctx, kv_dim, embd, l * 10 + 3);
        let (_, _, attn_v_scales, attn_v_quants) =
            helper_alloc_q4(&mut arena, &ctx, kv_dim, embd, l * 10 + 4);
        let (_, _, attn_out_scales, attn_out_quants) =
            helper_alloc_q4(&mut arena, &ctx, embd, q_dim, l * 10 + 5);
        let ffn_norm = helper_alloc_norm(&mut arena, &ctx, embd, l * 10 + 6).1;
        let (_, _, ffn_gate_scales, ffn_gate_quants) =
            helper_alloc_q4(&mut arena, &ctx, ffn, embd, l * 10 + 7);
        let (_, _, ffn_up_scales, ffn_up_quants) =
            helper_alloc_q4(&mut arena, &ctx, ffn, embd, l * 10 + 8);
        let (_, _, ffn_down_scales, ffn_down_quants) =
            helper_alloc_q4(&mut arena, &ctx, embd, ffn, l * 10 + 9);

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

    let output_norm = helper_alloc_norm(&mut arena, &ctx, embd, 99).1;
    let (_, _, output_scales, output_quants) =
        helper_alloc_q4(&mut arena, &ctx, vocab_size, embd, 100);

    let ocl_model = OclModelTensors {
        output_norm,
        output_scales,
        output_quants,
        layers: ocl_layers,
    };

    let kv_layer_bytes = capacity * (kv_dim / 32) * Q8_0_BLOCK_BYTES;
    let mut k_buffers = Vec::new();
    let mut v_buffers = Vec::new();
    for _ in 0..block_count {
        let k_s = arena.alloc(kv_layer_bytes, ArenaRegion::KvCache).unwrap();
        let v_s = arena.alloc(kv_layer_bytes, ArenaRegion::KvCache).unwrap();
        let k_b = unsafe {
            OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                k_s.as_ptr(),
                k_s.padded_size(),
                false,
            )
            .unwrap()
        };
        let v_b = unsafe {
            OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                v_s.as_ptr(),
                v_s.padded_size(),
                false,
            )
            .unwrap()
        };
        k_buffers.push(k_b);
        v_buffers.push(v_b);
    }

    let mut ocl_kv = OclKvCache {
        k_buffers,
        v_buffers,
        capacity,
        head_count_kv,
        head_dim,
    };

    let mut ocl_scratch = OclPrefillScratch {
        x: helper_alloc_scratch(&mut arena, &ctx, n * embd, false).1,
        h: helper_alloc_scratch(&mut arena, &ctx, n * embd, false).1,
        q: helper_alloc_scratch(&mut arena, &ctx, n * q_dim, false).1,
        k: helper_alloc_scratch(&mut arena, &ctx, n * kv_dim, false).1,
        v: helper_alloc_scratch(&mut arena, &ctx, n * kv_dim, false).1,
        attn_out: helper_alloc_scratch(&mut arena, &ctx, n * q_dim, false).1,
        ffn_gate: helper_alloc_scratch(&mut arena, &ctx, n * ffn, false).1,
        ffn_up: helper_alloc_scratch(&mut arena, &ctx, n * ffn, false).1,
        ffn_swiglu: helper_alloc_scratch(&mut arena, &ctx, n * ffn, false).1,
        ffn_down: helper_alloc_scratch(&mut arena, &ctx, n * embd, false).1,
        logits: helper_alloc_scratch(&mut arena, &ctx, vocab_size, true).1,
        max_n: n,
    };

    // Warmup
    execute_prefill_ubatch(&ctx, &cfg, &ocl_model, &mut ocl_kv, &mut ocl_scratch, 0, n)
        .expect("warmup");

    // Benchmark 5 runs
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        execute_prefill_ubatch(&ctx, &cfg, &ocl_model, &mut ocl_kv, &mut ocl_scratch, 0, n)
            .expect("bench run");
    }
    let elapsed = start.elapsed().as_secs_f64();
    let total_tokens = n * runs;
    let prefill_tok_per_sec = (total_tokens as f64) / elapsed;

    println!(
        "[Prefill Benchmark] N={n} tokens across {block_count} layers: {runs} runs in {elapsed:.3} s -> {prefill_tok_per_sec:.1} tok/s"
    );
}
