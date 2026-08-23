#include <metal_simdgroup_matrix>
#include <metal_stdlib>

// utils.h must come first: it provides the bfloat16_t alias that complex.h
// and the Steel headers reference (newer Metal toolchains no longer ship
// bfloat16_t as a builtin).
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"

using namespace metal;

// Starter interface map. Implement the named kernels at these checkpoints;
// their argument lists are defined by the matching C++ encoder you complete.
//
// Week 2, Day 3:
//   quantized_matmul_vanilla_w4a16_g128
//   quantized_matvec_x4_fast_w4a16_g128
// Week 2, Day 6:
//   quantized_matmul_simdgroup_w4a16_g128
// Week 2, Day 7:
//   quantized_matmul_simdgroup_splitk_w4a16_g128
//   quantized_matmul_splitk_reduce
// Week 3, Day 4:
//   quantized_embedding_w4a16_g128
//
// The x2/x8 tuning variants in the reference extension are deliberately not
// starter interfaces. Add an experimental variant only while running the
// optional scheduling comparison, then keep the selected course path.

template <typename T>
[[kernel]] void quantized_matmul_vanilla_w4a16_g128(
    device const T *scales [[buffer(0)]], device const T *biases [[buffer(1)]], device const T *a [[buffer(2)]],
    device const uint32_t *b [[buffer(3)]], device T *out [[buffer(4)]], device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]], device const int &K [[buffer(7)]],

    uint3 group_id [[threadgroup_position_in_grid]], uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
    const int bits = 4;
    const int group_size = 128;
    const int packs_per_item = 32 / bits;
    const int groups_per_row = N / group_size;

    const int i = group_id.x * threads_per_threadgroup.x + thread_id.x;
    const int k = group_id.y * threads_per_threadgroup.y + thread_id.y;

    float sum = 0;
    int scales_biases_loc = k * groups_per_row;

    const int mask = (1 << bits) - 1;

    if (i < M && k < K) {
        for (int group_idx = 0; group_idx < groups_per_row; group_idx++) {
            float scale = scales[scales_biases_loc + group_idx];
            float bias = biases[scales_biases_loc + group_idx];

            for (int pack_idx = 0; pack_idx < group_size / packs_per_item; pack_idx++) {
                int b_loc = k * (N / packs_per_item) + (group_idx * (group_size / packs_per_item)) + pack_idx;
                // take out packs_per_item values from b_loc
                uint32_t pack = b[b_loc];
                for (int bit_offset = 0; bit_offset < 32; bit_offset += 4) {
                    uint32_t b_value = (pack >> bit_offset) & mask;
                    float dequant_b_value = b_value * scale + bias;
                    int a_loc = i * (N) + (group_idx * group_size) + 8 * pack_idx + bit_offset / 4;
                    float a_value = a[a_loc];

                    sum += dequant_b_value * a_value;
                }
            }
        }
        int out_loc = i * K + k;
        // MLX's bfloat type has no implicit float conversion on newer
        // toolchains; cast once at the store.
        out[out_loc] = static_cast<T>(sum);
    }
}

instantiate_kernel("quantized_matmul_vanilla_w4a16_g128_f16", quantized_matmul_vanilla_w4a16_g128, half);
instantiate_kernel("quantized_matmul_vanilla_w4a16_g128_bf16", quantized_matmul_vanilla_w4a16_g128, bfloat16_t);

// Decode is a matrix-vector workload: M is usually 1 and every output row
// reduces over the same activation vector. One SIMD group cooperates on an
// output tile instead of assigning the whole reduction to one thread.
//
// Schedule (the Qwen-focused starting point from the book):
//   - 4 output columns per SIMD group,
//   - each lane loads 2 adjacent packed words (= 16 activations) once and
//     reuses them across those 4 outputs,
//   - activations are pre-scaled by 1/16^slot so the hot loop can mask the
//     packed uint16 weights directly without a shift per weight per row,
//   - the affine identity sum_j a_j*(s*q_j + b) = s*sum(a_j*q_j) + b*sum(a_j)
//     applies the bias once per group instead of once per value,
//   - lane partial sums combine with simd_sum; lane 0 stores.
// The C++ encoder launches 2 SIMD groups per threadgroup.
template <typename T>
[[kernel]] void quantized_matvec_x4_fast_w4a16_g128(
    device const T *scales [[buffer(0)]], device const T *biases [[buffer(1)]], device const T *a [[buffer(2)]],
    device const uint32_t *b [[buffer(3)]], device T *out [[buffer(4)]], device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]], device const int &K [[buffer(7)]],

    uint output_tile [[threadgroup_position_in_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int packs_per_item = 32 / bits;      // 8 int4 values per uint32
    constexpr int packs_per_lane = 2;              // 2 packed words per lane
    constexpr int values_per_lane = packs_per_item * packs_per_lane;  // 16 activations
    constexpr int outputs_per_simdgroup = 4;
    constexpr int simdgroups_per_threadgroup = 2;
    constexpr int outputs_per_threadgroup = outputs_per_simdgroup * simdgroups_per_threadgroup;

    const int column_tiles = (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
    const int row = output_tile / column_tiles;
    const int column_base = (output_tile - row * column_tiles) * outputs_per_threadgroup + simdgroup * outputs_per_simdgroup;
    if (row >= M || column_base >= K) {
        return;
    }

    const int packed_cols = N / packs_per_item;
    const int groups_per_row = N / group_size;
    const int a_base = row * N;

    float sums[outputs_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int packed_col = lane * packs_per_lane; packed_col < packed_cols; packed_col += 32 * packs_per_lane) {
        // Both packed words of a lane stay inside one quantization group
        // because a group holds 16 packed words and lanes move in pairs.
        const int group = packed_col / (group_size / packs_per_item);

        float scaled_activations[values_per_lane];
        float activation_sum = 0.0f;
#pragma clang loop unroll(full)
        for (int pack = 0; pack < packs_per_lane; ++pack) {
            const int activation_offset = a_base + (packed_col + pack) * packs_per_item;
#pragma clang loop unroll(full)
            for (int value = 0; value < packs_per_item; ++value) {
                const int local = pack * packs_per_item + value;
                const float activation = static_cast<float>(a[activation_offset + value]);
                activation_sum += activation;
                // Four adjacent W4 codes live in one uint16 at bit positions
                // 0/4/8/12. Scaling the activations once lets the hot loop use
                // masks directly: (a / 16^s) * (q << 4s) == a * q with no
                // per-weight shift.
                scaled_activations[local] = activation / static_cast<float>(1 << ((value & 3) * 4));
            }
        }

#pragma clang loop unroll(full)
        for (int output = 0; output < outputs_per_simdgroup; ++output) {
            const int column = column_base + output;
            if (column >= K) {
                continue;
            }
            const int parameter_index = column * groups_per_row + group;
            const float scale = static_cast<float>(scales[parameter_index]);
            const float bias = static_cast<float>(biases[parameter_index]);

            // Two adjacent packed words read as one uint32 pair of uint16
            // lanes (little-endian: low uint16 holds codes 0-3).
            const device uint16_t *packed =
                reinterpret_cast<const device uint16_t *>(b + column * packed_cols + packed_col);
            float quantized_dot = 0.0f;
#pragma clang loop unroll(full)
            for (int sub = 0; sub < values_per_lane / 4; ++sub) {
                const uint16_t weights = packed[sub];
                const int local = sub * 4;
                quantized_dot += scaled_activations[local] * (weights & 0x000f) +
                                 scaled_activations[local + 1] * (weights & 0x00f0) +
                                 scaled_activations[local + 2] * (weights & 0x0f00) +
                                 scaled_activations[local + 3] * (weights & 0xf000);
            }
            sums[output] += scale * quantized_dot + bias * activation_sum;
        }
    }

#pragma clang loop unroll(full)
    for (int output = 0; output < outputs_per_simdgroup; ++output) {
        sums[output] = simd_sum(sums[output]);
    }
    if (lane == 0) {
#pragma clang loop unroll(full)
        for (int output = 0; output < outputs_per_simdgroup; ++output) {
            const int column = column_base + output;
            if (column < K) {
                out[row * K + column] = static_cast<T>(sums[output]);
            }
        }
    }
}

instantiate_kernel("quantized_matvec_x4_fast_w4a16_g128_f16", quantized_matvec_x4_fast_w4a16_g128, half);
instantiate_kernel("quantized_matvec_x4_fast_w4a16_g128_bf16", quantized_matvec_x4_fast_w4a16_g128, bfloat16_t);

// Week 3, Day 4: quantized embedding lookup as one kernel. One thread per
// output element gathers its packed word, extracts the code, and applies the
// row's scale/bias directly - no dense dequantized table is materialized.
template <typename T, typename IndexT>
[[kernel]] void quantized_embedding_w4a16_g128(
    device const IndexT* indices [[buffer(0)]],
    device const T* scales [[buffer(1)]],
    device const T* biases [[buffer(2)]],
    device const uint32_t* weights [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &tokens [[buffer(5)]],
    device const int &dim [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int packs_per_item = 32 / bits;
    constexpr uint32_t mask = (1 << bits) - 1;
    if (index >= tokens * dim) {
        return;
    }
    const int token = index / dim;
    const int column = index - token * dim;
    const int row = indices[token];
    const int packed_cols = dim / packs_per_item;
    const int groups_per_row = dim / group_size;
    const uint32_t packed =
        weights[row * packed_cols + column / packs_per_item];
    const int shift = (column % packs_per_item) * bits;
    const float quantized = static_cast<float>((packed >> shift) & mask);
    const float scale = static_cast<float>(
        scales[row * groups_per_row + column / group_size]);
    const float bias = static_cast<float>(
        biases[row * groups_per_row + column / group_size]);
    out[index] = static_cast<T>(quantized * scale + bias);
}

instantiate_kernel("quantized_embedding_w4a16_g128_f16_i32", quantized_embedding_w4a16_g128, half, int32_t);
instantiate_kernel("quantized_embedding_w4a16_g128_bf16_i32", quantized_embedding_w4a16_g128, bfloat16_t, int32_t);
instantiate_kernel("quantized_embedding_w4a16_g128_f16_u32", quantized_embedding_w4a16_g128, half, uint32_t);
instantiate_kernel("quantized_embedding_w4a16_g128_bf16_u32", quantized_embedding_w4a16_g128, bfloat16_t, uint32_t);


// Week 2, Day 6: cooperative SIMD-matrix prefill.
//
// Prefill reuses each weight across many activation rows, so a 32x32x32 tile
// replaces the one-thread-per-output vanilla schedule. Four SIMD groups (128
// threads) cooperate on one tile:
//   1. a Steel BlockLoader stages the 32x32 activation tile into padded
//      threadgroup memory (stride 40 avoids bank conflicts),
//   2. four threads per output column unpack/dequantize the matching weight
//      tile into shared storage, hoisting one scale/bias load per column per
//      128-value quantization group (four 32-value reduction steps),
//   3. four 16x16 quadrants accumulate from 8x8 matrix fragments in FP32,
//   4. the result tile stores once with tail guards.
// The same helper serves Split-K (Day 7): partitions bound the reduction and
// offset the output plane instead of changing the schedule.
template <typename T, typename OutT>
inline void quantized_matmul_block_w4a16_g128(
    device const T* scales,
    device const T* biases,
    device const T* a,
    device const uint32_t* b,
    device OutT* out,
    const int M,
    const int N,
    const int K,
    const int reduction_start,
    const int reduction_end,
    const int output_offset,
    threadgroup T* activation_tile,
    threadgroup T* weight_tile,
    threadgroup T* quantization_parameters,
    uint3 group_id,
    uint thread_id,
    uint simdgroup,
    uint lane) {
    constexpr int output_block_size = 32;
    constexpr int reduction_block_size = 32;
    constexpr int padded_reduction_size = 40;
    constexpr int group_size = 128;
    constexpr int packs_per_item = 8;
    constexpr uint32_t mask = 0xf;
    const int row_base = group_id.y * output_block_size;
    const int column_base = group_id.x * output_block_size;
    const int packed_cols = N / packs_per_item;
    const int groups_per_row = N / group_size;

    using block_mma = mlx::steel::BlockMMA<
        T, OutT, output_block_size, output_block_size,
        reduction_block_size, 2, 2, false, true,
        padded_reduction_size, padded_reduction_size>;
    using activation_loader = mlx::steel::BlockLoader<
        T, output_block_size, reduction_block_size,
        padded_reduction_size, 1, 128>;
    block_mma mma(simdgroup, lane);
    activation_loader load_activation(
        a + row_base * N + reduction_start,
        N,
        activation_tile,
        simdgroup,
        lane);

    // Threads 4j..4j+3 unpack column j's 32 weights for this step; the scale
    // and bias arrive from shared memory, refreshed once per quantization
    // group instead of once per 32-value step.
    const int weight_output = thread_id / 4;
    const int weight_pack = thread_id % 4;
    const int output_column = column_base + weight_output;
    const bool valid_output = output_column < K;
    device const uint32_t* weight_source = valid_output
        ? b + output_column * packed_cols + reduction_start / packs_per_item +
            weight_pack
        : b;
    threadgroup T* weight_destination =
        weight_tile + weight_output * padded_reduction_size +
        weight_pack * packs_per_item;
    int quantization_group_step = 0;

    if (thread_id < output_block_size) {
        const int parameter_output = column_base + thread_id;
        const bool valid_parameter = parameter_output < K;
        const int parameter_index =
            parameter_output * groups_per_row + reduction_start / group_size;
        quantization_parameters[thread_id] =
            valid_parameter ? scales[parameter_index] : T(0);
        quantization_parameters[output_block_size + thread_id] =
            valid_parameter ? biases[parameter_index] : T(0);
    }

    for (int reduction_base = reduction_start;
         reduction_base < reduction_end;
         reduction_base += reduction_block_size) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_base + output_block_size <= M) {
            load_activation.load_unsafe();
        } else {
            load_activation.load_safe(short2(
                reduction_block_size,
                max(0, M - row_base)));
        }

        const uint32_t packed = valid_output ? *weight_source : 0;
        const float scale =
            static_cast<float>(quantization_parameters[weight_output]);
        const float bias = static_cast<float>(
            quantization_parameters[output_block_size + weight_output]);
#pragma clang loop unroll(full)
        for (int value = 0; value < packs_per_item; ++value) {
            const float quantized =
                static_cast<float>((packed >> (value * 4)) & mask);
            weight_destination[value] =
                static_cast<T>(quantized * scale + bias);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.mma(activation_tile, weight_tile);

        load_activation.next();
        weight_source += reduction_block_size / packs_per_item;
        quantization_group_step += reduction_block_size;
        if (quantization_group_step == group_size) {
            // Crossed a 128-value group boundary: fetch the next scale/bias.
            quantization_group_step = 0;
            const int next_reduction_base =
                reduction_base + reduction_block_size;
            if (next_reduction_base < reduction_end &&
                thread_id < output_block_size) {
                const int parameter_output = column_base + thread_id;
                const bool valid_parameter = parameter_output < K;
                const int parameter_index = parameter_output * groups_per_row +
                    next_reduction_base / group_size;
                quantization_parameters[thread_id] =
                    valid_parameter ? scales[parameter_index] : T(0);
                quantization_parameters[output_block_size + thread_id] =
                    valid_parameter ? biases[parameter_index] : T(0);
            }
        }
    }

    const short valid_rows = min(output_block_size, M - row_base);
    const short valid_columns = min(output_block_size, K - column_base);
    mma.store_result_safe(
        out + output_offset + row_base * K + column_base,
        K,
        short2(valid_columns, valid_rows));
}

template <typename T>
[[kernel]] void quantized_matmul_simdgroup_w4a16_g128(
    device const T *scales [[buffer(0)]], device const T *biases [[buffer(1)]],
    device const T *a [[buffer(2)]], device const uint32_t *b [[buffer(3)]],
    device T *out [[buffer(4)]], device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]], device const int &K [[buffer(7)]],

    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup T activation_tile[32 * 40];
    threadgroup T weight_tile[32 * 40];
    threadgroup T quantization_parameters[2 * 32];
    quantized_matmul_block_w4a16_g128(
        scales, biases, a, b, out, M, N, K, 0, N, 0,
        activation_tile, weight_tile, quantization_parameters,
        group_id, thread_id, simdgroup, lane);
}

// Week 2, Day 7: Split-K accumulation. group_id.z picks the partition; every
// partition covers an equal, 128-aligned slice of the reduction and writes
// its own [M, K] plane of partials without atomics.
template <typename T>
[[kernel]] void quantized_matmul_simdgroup_splitk_w4a16_g128(
    device const T *scales [[buffer(0)]], device const T *biases [[buffer(1)]],
    device const T *a [[buffer(2)]], device const uint32_t *b [[buffer(3)]],
    device T *partials [[buffer(4)]], device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]], device const int &K [[buffer(7)]],
    device const int &partition_size [[buffer(8)]],
    device const int &partition_stride [[buffer(9)]],

    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup T activation_tile[32 * 40];
    threadgroup T weight_tile[32 * 40];
    threadgroup T quantization_parameters[2 * 32];
    const int reduction_start = group_id.z * partition_size;
    quantized_matmul_block_w4a16_g128(
        scales, biases, a, b, partials, M, N, K,
        reduction_start, reduction_start + partition_size,
        group_id.z * partition_stride, activation_tile, weight_tile,
        quantization_parameters, group_id, thread_id, simdgroup, lane);
}

// Week 2, Day 7: merge the partial planes. One thread per output element;
// the sum accumulates in FP32 and casts to the model dtype once.
template <typename T>
[[kernel]] void quantized_matmul_splitk_reduce(
    device const T *partials [[buffer(0)]], device T *out [[buffer(1)]],
    device const int &elements [[buffer(2)]],
    device const int &split_k [[buffer(3)]],

    uint index [[thread_position_in_grid]]) {
    if (index >= static_cast<uint>(elements)) {
        return;
    }
    float sum = 0.0f;
    for (int partition = 0; partition < split_k; ++partition) {
        sum += static_cast<float>(partials[partition * elements + index]);
    }
    out[index] = static_cast<T>(sum);
}

instantiate_kernel("quantized_matmul_simdgroup_w4a16_g128_f16", quantized_matmul_simdgroup_w4a16_g128, half);
instantiate_kernel("quantized_matmul_simdgroup_w4a16_g128_bf16", quantized_matmul_simdgroup_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matmul_simdgroup_splitk_w4a16_g128_f16", quantized_matmul_simdgroup_splitk_w4a16_g128, half);
instantiate_kernel("quantized_matmul_simdgroup_splitk_w4a16_g128_bf16", quantized_matmul_simdgroup_splitk_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matmul_splitk_reduce_f16", quantized_matmul_splitk_reduce, half);
instantiate_kernel("quantized_matmul_splitk_reduce_bf16", quantized_matmul_splitk_reduce, bfloat16_t);