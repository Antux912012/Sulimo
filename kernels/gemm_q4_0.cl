#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

// Portable Q4_0 x F16 GEMM kernel
// Output matrix: N x M (row-major: N tokens, M output features)
// Activations X: N x K (row-major: N tokens, K input channels)
// Weights W (Q4_0 SoA): M rows of length K
// scales: array of M * (K / 32) half values
// quants: array of M * (K / 32) * 16 uchar values
__kernel void gemm_q4_0_f16_portable(
    __global const half *scales,
    __global const uchar *quants,
    __global const half *x,
    __global half *out,
    const int M,
    const int K,
    const int N
) {
    const int m = get_global_id(0); // 0..M-1
    const int n = get_global_id(1); // 0..N-1

    if (m >= M || n >= N) {
        return;
    }

    const int num_blocks_per_row = K / 32;
    const int weight_row_block_offset = m * num_blocks_per_row;
    __global const half *row_scales = scales + weight_row_block_offset;
    __global const uchar *row_quants = quants + (weight_row_block_offset * 16);
    __global const half *token_x = x + (n * K);

    float acc = 0.0f;

    for (int b = 0; b < num_blocks_per_row; b++) {
        const float d = (float)row_scales[b];
        __global const uchar *q_ptr = row_quants + (b * 16);
        __global const half *x_ptr = token_x + (b * 32);

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

    out[n * M + m] = (half)acc;
}

#if defined(GEN8_PLUS_FP16) && defined(cl_intel_subgroups)
// Intel Gen8-11 SIMD8 specialized GEMM kernel using sub-groups and FP32 accumulation
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void gemm_q4_0_f16_intel(
    __global const half *scales,
    __global const uchar *quants,
    __global const half *x,
    __global half *out,
    const int M,
    const int K,
    const int N
) {
    const int m = get_global_id(0); // row index
    const int n = get_global_id(1); // token index

    if (m >= M || n >= N) {
        return;
    }

    const int num_blocks_per_row = K / 32;
    const int weight_row_block_offset = m * num_blocks_per_row;
    __global const half *row_scales = scales + weight_row_block_offset;
    __global const uchar *row_quants = quants + (weight_row_block_offset * 16);
    __global const half *token_x = x + (n * K);

    float acc = 0.0f;

    for (int b = 0; b < num_blocks_per_row; b++) {
        const float d = (float)row_scales[b];
        __global const uchar *q_ptr = row_quants + (b * 16);
        __global const half *x_ptr = token_x + (b * 32);

        float4 block_acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        for (int i = 0; i < 16; i += 4) {
            const uchar q0 = q_ptr[i + 0];
            const uchar q1 = q_ptr[i + 1];
            const uchar q2 = q_ptr[i + 2];
            const uchar q3 = q_ptr[i + 3];

            float4 q_low = (float4)(
                (float)((int)(q0 & 0x0F) - 8),
                (float)((int)(q1 & 0x0F) - 8),
                (float)((int)(q2 & 0x0F) - 8),
                (float)((int)(q3 & 0x0F) - 8)
            );

            float4 q_high = (float4)(
                (float)((int)(q0 >> 4) - 8),
                (float)((int)(q1 >> 4) - 8),
                (float)((int)(q2 >> 4) - 8),
                (float)((int)(q3 >> 4) - 8)
            );

            half4 x_low_h = vload4(0, x_ptr + i);
            half4 x_high_h = vload4(0, x_ptr + 16 + i);

            float4 x_low = (float4)((float)x_low_h.s0, (float)x_low_h.s1, (float)x_low_h.s2, (float)x_low_h.s3);
            float4 x_high = (float4)((float)x_high_h.s0, (float)x_high_h.s1, (float)x_high_h.s2, (float)x_high_h.s3);

            block_acc = fma(q_low, x_low, block_acc);
            block_acc = fma(q_high, x_high, block_acc);
        }

        float block_sum = block_acc.s0 + block_acc.s1 + block_acc.s2 + block_acc.s3;
        acc += block_sum * d;
    }

    out[n * M + m] = (half)acc;
}
#endif
