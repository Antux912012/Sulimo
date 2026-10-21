#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// RMSNorm over FP16 activations with FP32 weights: out = (x / rms(x)) * w
__kernel void rmsnorm_f16_f32w(
    __global const half *x,
    __global const float *w,
    __global half *out,
    const int D,
    const float eps,
    const int N
) {
    const int n = get_global_id(0);
    if (n >= N) return;

    __global const half *x_row = x + (n * D);
    __global half *out_row = out + (n * D);

    float sum_sq = 0.0f;
    for (int i = 0; i < D; i++) {
        float val = (float)x_row[i];
        sum_sq += val * val;
    }
    float mean = sum_sq / (float)D;
    float scale = 1.0f / sqrt(mean + eps);

    for (int i = 0; i < D; i++) {
        out_row[i] = (half)((float)x_row[i] * scale * w[i]);
    }
}

// RoPE rotation kernel for Q or K heads across all prompt tokens
__kernel void rope_f16(
    __global half *vec,
    const int head_count,
    const int head_dim,
    const int rope_dim,
    const float freq_base,
    const int mode,
    const int start_pos,
    const int N
) {
    const int n = get_global_id(0); // token index
    const int h = get_global_id(1); // head index
    if (n >= N || h >= head_count) return;

    const int pos = start_pos + n;
    const int half_dim = rope_dim / 2;
    __global half *head_ptr = vec + (n * head_count * head_dim) + (h * head_dim);

    for (int i = 0; i < half_dim; i++) {
        float freq = 1.0f / pow(freq_base, (float)(2 * i) / (float)rope_dim);
        float theta = (float)pos * freq;
        float cos_t = cos(theta);
        float sin_t = sin(theta);

        if (mode == 0) { // Norm (Llama adjacent pairs)
            float v0 = (float)head_ptr[2 * i];
            float v1 = (float)head_ptr[2 * i + 1];
            head_ptr[2 * i]     = (half)(v0 * cos_t - v1 * sin_t);
            head_ptr[2 * i + 1] = (half)(v0 * sin_t + v1 * cos_t);
        } else { // Neox (Qwen2 split halves)
            float v0 = (float)head_ptr[i];
            float v1 = (float)head_ptr[i + half_dim];
            head_ptr[i]            = (half)(v0 * cos_t - v1 * sin_t);
            head_ptr[i + half_dim] = (half)(v0 * sin_t + v1 * cos_t);
        }
    }
}

// Stores an FP16 activation slice into the Q8_0 ring KV cache buffer ($ZCOPY, @C:COL)
// Each block covers 32 activations -> 2 bytes f16 scale + 32 bytes i8 quants (34 bytes total)
__kernel void kv_store_q8_0(
    __global const half *act,
    __global uchar *kv_cache,
    const int start_pos,
    const int capacity,
    const int blocks_per_pos,
    const int N
) {
    const int n = get_global_id(0); // token index
    const int b = get_global_id(1); // block index
    if (n >= N || b >= blocks_per_pos) return;

    const int ring_pos = (start_pos + n) % capacity;
    const int bytes_per_pos = blocks_per_pos * 34;
    const int block_offset = ring_pos * bytes_per_pos + (b * 34);

    __global const half *in_ptr = act + (n * blocks_per_pos * 32) + (b * 32);
    __global uchar *out_ptr = kv_cache + block_offset;

    // 1. Find max absolute value
    float max_val = 0.0f;
    for (int i = 0; i < 32; i++) {
        float val = fabs((float)in_ptr[i]);
        if (val > max_val) {
            max_val = val;
        }
    }

    const float d = max_val / 127.0f;
    const half d_half = (half)d;
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

    // Write 2-byte f16 scale (little-endian)
    *(__global half *)out_ptr = d_half;

    // Write 32 signed 8-bit quantized values
    __global char *quants_out = (__global char *)(out_ptr + 2);
    for (int i = 0; i < 32; i++) {
        float v = (float)in_ptr[i];
        int q = (int)round(v * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        quants_out[i] = (char)q;
    }
}

// Causal Grouped Query Attention (GQA) prefill kernel ($REF)
// Q attends to all K and V up to current token position
__kernel void causal_gqa_attention_prefill(
    __global const half *q,
    __global const uchar *k_cache,
    __global const uchar *v_cache,
    __global half *out,
    const int start_pos,
    const int capacity,
    const int head_count,
    const int head_count_kv,
    const int head_dim,
    const int N
) {
    const int n = get_global_id(0); // token index
    const int h = get_global_id(1); // query head index
    if (n >= N || h >= head_count) return;

    const int heads_per_kv = head_count / head_count_kv;
    const int kv_head = h / heads_per_kv;
    const int current_pos = start_pos + n;
    const int num_keys = current_pos + 1;

    const float scale = 1.0f / sqrt((float)head_dim);
    const int blocks_per_head = head_dim / 32;
    const int head_bytes = blocks_per_head * 34;
    const int total_blocks_per_pos = head_count_kv * blocks_per_head;
    const int bytes_per_pos = total_blocks_per_pos * 34;

    __global const half *q_head = q + (n * head_count * head_dim) + (h * head_dim);

    // 1. Pass 1: compute attention scores Q · Kᵀ * scale and find max score for stable softmax
    float max_score = -1e30f;

    // Use private stack/local buffer for tokens within prompt
    for (int p = 0; p < num_keys; p++) {
        const int ring_pos = p % capacity;
        __global const uchar *k_ptr = k_cache + (ring_pos * bytes_per_pos) + (kv_head * head_bytes);

        float dot = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            __global const uchar *blk_ptr = k_ptr + (b * 34);
            const float d = (float)(*(__global const half *)blk_ptr);
            __global const char *qs = (__global const char *)(blk_ptr + 2);
            __global const half *qh = q_head + (b * 32);

            for (int i = 0; i < 32; i++) {
                dot += (float)qh[i] * ((float)qs[i] * d);
            }
        }

        float score = dot * scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    // 2. Pass 2: compute denominator sum of exp(score - max_score)
    float sum_exp = 0.0f;
    for (int p = 0; p < num_keys; p++) {
        const int ring_pos = p % capacity;
        __global const uchar *k_ptr = k_cache + (ring_pos * bytes_per_pos) + (kv_head * head_bytes);

        float dot = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            __global const uchar *blk_ptr = k_ptr + (b * 34);
            const float d = (float)(*(__global const half *)blk_ptr);
            __global const char *qs = (__global const char *)(blk_ptr + 2);
            __global const half *qh = q_head + (b * 32);

            for (int i = 0; i < 32; i++) {
                dot += (float)qh[i] * ((float)qs[i] * d);
            }
        }

        float score = dot * scale;
        sum_exp += exp(score - max_score);
    }

    const float inv_sum_exp = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    // 3. Pass 3: accumulate weighted values into output
    __global half *out_head = out + (n * head_count * head_dim) + (h * head_dim);

    for (int d_idx = 0; d_idx < head_dim; d_idx++) {
        out_head[d_idx] = 0.0h;
    }

    for (int p = 0; p < num_keys; p++) {
        const int ring_pos = p % capacity;
        __global const uchar *k_ptr = k_cache + (ring_pos * bytes_per_pos) + (kv_head * head_bytes);

        float dot = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            __global const uchar *blk_ptr = k_ptr + (b * 34);
            const float d = (float)(*(__global const half *)blk_ptr);
            __global const char *qs = (__global const char *)(blk_ptr + 2);
            __global const half *qh = q_head + (b * 32);

            for (int i = 0; i < 32; i++) {
                dot += (float)qh[i] * ((float)qs[i] * d);
            }
        }

        float score = dot * scale;
        float weight = exp(score - max_score) * inv_sum_exp;

        __global const uchar *v_ptr = v_cache + (ring_pos * bytes_per_pos) + (kv_head * head_bytes);
        for (int b = 0; b < blocks_per_head; b++) {
            __global const uchar *blk_ptr = v_ptr + (b * 34);
            const float d = (float)(*(__global const half *)blk_ptr);
            __global const char *qs = (__global const char *)(blk_ptr + 2);

            for (int i = 0; i < 32; i++) {
                float v_val = (float)qs[i] * d;
                out_head[b * 32 + i] += (half)(weight * v_val);
            }
        }
    }
}

// SwiGLU activation kernel: out = silu(gate) * up
__kernel void silu_mul_f16(
    __global const half *gate,
    __global const half *up,
    __global half *out,
    const int total_elements
) {
    const int idx = get_global_id(0);
    if (idx >= total_elements) return;

    float g = (float)gate[idx];
    float u = (float)up[idx];
    float silu = g / (1.0f + exp(-g));
    out[idx] = (half)(silu * u);
}

// Residual addition: a = a + b
__kernel void add_f16(
    __global half *a,
    __global const half *b,
    const int total_elements
) {
    const int idx = get_global_id(0);
    if (idx >= total_elements) return;

    out_add:
    a[idx] = (half)((float)a[idx] + (float)b[idx]);
}

// Logits projection kernel for final position: logits = x_final · W^T
__kernel void logits_f16_output(
    __global const half *x_all,
    __global const half *scales,
    __global const uchar *quants,
    __global float *logits,
    const int M, // vocab_size
    const int K, // embd_dim
    const int token_idx
) {
    const int m = get_global_id(0); // vocab row index
    if (m >= M) return;

    const int num_blocks = K / 32;
    __global const half *row_scales = scales + (m * num_blocks);
    __global const uchar *row_quants = quants + (m * num_blocks * 16);
    __global const half *x_final = x_all + (token_idx * K);

    float acc = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        const float d = (float)row_scales[b];
        __global const uchar *q_ptr = row_quants + (b * 16);
        __global const half *x_ptr = x_final + (b * 32);

        float block_sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            const uchar q = q_ptr[i];
            const float q_low = (float)((int)(q & 0x0F) - 8);
            const float q_high = (float)((int)(q >> 4) - 8);

            const float x_low = (float)x_ptr[i];
            const float x_high = (float)x_ptr[16 + i];

            block_sum += x_low * q_low + x_high * q_high;
        }

        acc += block_sum * d;
    }

    logits[m] = acc;
}
