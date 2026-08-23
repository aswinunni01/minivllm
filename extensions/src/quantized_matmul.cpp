#include <mlx/utils.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tiny_llm_ext.h"

// Provides mx::metal::device/get_command_encoder and pulls in the MTL::
// types used by the dispatch code below.
#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 2, Day 3. Days 6 and 7 extend the dispatch policy behind this API.
mx::array quantized_matmul(const mx::array &scales, const mx::array &biases, const int group_size, const int bits,
                           const mx::array &a, const mx::array &b, const bool transpose_b, const bool use_simdgroup,
                           const bool use_split_k, mx::StreamOrDevice s) {
    // Validate shapes
    // scales , bias -> K x N/G
    // a -> M x N
    // b -> K x N , quantized to K x N / (32/bits)
    // dtype of a, scales, bias -> fp16
    // dtype of b -> uint32

    if (a.dtype() != scales.dtype()) {
        throw std::runtime_error("a must be the same dtype as scales");
    }

    if (b.dtype() != mx::uint32) {
        throw std::runtime_error("b must be quantized and packed into uint32");
    }

    int pack_factor = 32 / bits;
    if (b.shape()[1] * pack_factor != a.shape()[1]) {
        throw std::runtime_error("Weight dim mismatch b.shape[-1]*pack_factor != a.shape[-1]");
    }

    if (scales.shape()[0] != biases.shape()[0]) {
        throw std::runtime_error("Scales and bias must have compatible shapes");
    }

    if (scales.shape()[0] != b.shape()[0]) {
        throw std::runtime_error("Scales and b must have compatible shapes");
    }

    // Both sides count the logical N dimension: scales hold N/group_size
    // entries per row, b holds N/pack_factor packed uint32 words per row.
    if (scales.shape()[1] * group_size != b.shape()[1] * pack_factor) {
        throw std::runtime_error("Scales and b must have compatible shapes");
    }

    // The Week 2 Metal schedules hard-code the W4A16 g128 layout.
    if (group_size != 128 || bits != 4) {
        throw std::runtime_error("quantized_matmul: course kernels require group_size=128 and bits=4");
    }
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: only transpose_b=true is supported");
    }

    // output_shape = M x K

    mx::Shape output_shape = {a.shape()[0], b.shape()[0]};

    return mx::array(output_shape, a.dtype(),
                     std::make_shared<QuantizedMatmul>(mlx::core::to_stream(s), use_simdgroup, use_split_k),
                     {scales, biases, a, b});
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("QuantizedMatmul::eval_cpu the extension is GPU-only");
}

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];

    // Metal receives raw buffers, so row-contiguous layout is a correctness
    // condition, not just a performance one. The Python wrapper normalizes
    // with mx.contiguous; assert it here as well.
    if (!a.flags().row_contiguous || !b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a and b must be row contiguous");
    }

    const int M = a.shape()[0];
    const int N = a.shape()[1];
    const int K = b.shape()[0];

    auto &s = stream();
    auto &d = mx::metal::device(s.device);

    auto library = d.get_library("tiny_llm_ext");

    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // Week 2 dispatch policy, extended per checkpoint:
    //   Day 3: decode-shaped inputs (M <= 8) take the SIMD-group matvec;
    //          everything else kept my vanilla matrix grid as the control.
    //   Day 6: with use_simdgroup_, matrix shapes take the cooperative
    //          32x32x32 SIMD-matrix schedule.
    //   Day 7: with use_split_k_, under-filled grids split the reduction
    //          until the grid is occupied; everything else stays unsplit.
    const bool use_matvec = use_simdgroup_ && M <= 8;

    int split_k = 1;
    if (use_split_k_ && use_simdgroup_ && !use_matvec) {
        constexpr int block_size = 32;
        constexpr int target_threadgroups = 320;
        constexpr int max_split_k = 16;
        const int row_blocks = (M + block_size - 1) / block_size;
        const int column_blocks = (K + block_size - 1) / block_size;
        const int threadgroups = row_blocks * column_blocks;
        split_k = std::min({max_split_k,
                            std::max(1, target_threadgroups / std::max(threadgroups, 1)),
                            N / 128});
        // Partitions must start and end on quantization-group boundaries.
        while (split_k > 1 && N % (split_k * 128) != 0) {
            split_k--;
        }
    }
    const bool use_split_k = split_k > 1;

    const char *kernel_name;
    if (use_matvec) {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matvec_x4_fast_w4a16_g128_f16"
                                                 : "quantized_matvec_x4_fast_w4a16_g128_bf16";
    } else if (use_split_k) {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_simdgroup_splitk_w4a16_g128_f16"
                                                 : "quantized_matmul_simdgroup_splitk_w4a16_g128_bf16";
    } else if (use_simdgroup_) {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_simdgroup_w4a16_g128_f16"
                                                 : "quantized_matmul_simdgroup_w4a16_g128_bf16";
    } else {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_vanilla_w4a16_g128_f16"
                                                 : "quantized_matmul_vanilla_w4a16_g128_bf16";
    }

    auto kernel = d.get_kernel(kernel_name, library);

    auto &compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);

    if (use_split_k) {
        // Accumulation: one [M, K] partial plane per partition.
        compute_encoder.set_bytes(M, 5);
        compute_encoder.set_bytes(N, 6);
        compute_encoder.set_bytes(K, 7);
        auto partial_shape = out.shape();
        partial_shape.insert(partial_shape.begin(), split_k);
        mx::array partials(partial_shape, out.dtype(), nullptr, {});
        partials.set_data(mx::allocator::malloc(partials.nbytes()));
        compute_encoder.add_temporary(partials);
        compute_encoder.set_output_array(partials, 4);

        constexpr int tile = 32;
        const int row_blocks = (M + tile - 1) / tile;
        const int column_blocks = (K + tile - 1) / tile;
        const int partition_size = N / split_k;
        const int partition_stride = M * K;
        compute_encoder.set_bytes(partition_size, 8);
        compute_encoder.set_bytes(partition_stride, 9);
        compute_encoder.dispatch_threadgroups(MTL::Size(column_blocks, row_blocks, split_k),
                                              MTL::Size(128, 1, 1));

        // Reduction: one thread per output element, FP32 sum, single cast.
        const char *reduce_name =
            out.dtype() == mx::float16 ? "quantized_matmul_splitk_reduce_f16"
                                       : "quantized_matmul_splitk_reduce_bf16";
        auto reduce_kernel = d.get_kernel(reduce_name, library);
        compute_encoder.set_compute_pipeline_state(reduce_kernel);
        compute_encoder.set_input_array(partials, 0);
        compute_encoder.set_output_array(out, 1);
        const int elements = M * K;
        compute_encoder.set_bytes(elements, 2);
        compute_encoder.set_bytes(split_k, 3);
        const int threads = std::min<int>(reduce_kernel->maxTotalThreadsPerThreadgroup(), 256);
        compute_encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(threads, 1, 1));
        return;
    }

    compute_encoder.set_output_array(out, 4);

    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    if (use_matvec) {
        // Four output columns per SIMD group, two packed words per lane,
        // two SIMD groups per threadgroup: each lane loads 16 adjacent
        // activations once and reuses them across four weight rows.
        constexpr int outputs_per_simdgroup = 4;
        constexpr int simdgroups_per_threadgroup = 2;
        const int outputs_per_threadgroup = simdgroups_per_threadgroup * outputs_per_simdgroup;
        const int column_tiles = (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
        compute_encoder.dispatch_threadgroups(MTL::Size(M * column_tiles, 1, 1),
                                              MTL::Size(simdgroups_per_threadgroup * 32, 1, 1));
        return;
    }

    if (use_simdgroup_) {
        // Cooperative 32x32x32 tiles: four SIMD groups per threadgroup.
        constexpr int tile = 32;
        const int row_blocks = (M + tile - 1) / tile;
        const int column_blocks = (K + tile - 1) / tile;
        compute_encoder.dispatch_threadgroups(MTL::Size(column_blocks, row_blocks, 1),
                                              MTL::Size(128, 1, 1));
        return;
    }

    // Vanilla control grid, kept from my Day 3 bring-up.
    MTL::Size group_dims = MTL::Size(16, 16, 1);
    MTL::Size grid_dims = MTL::Size((M + 15) / 16, (K + 15) / 16, 1);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// Week 3, Day 4. One dispatch replaces the readable selected-row lookup.
mx::array quantized_embedding(const mx::array &indices, const mx::array &scales, const mx::array &biases,
                              const mx::array &weight, int group_size, int bits, mx::StreamOrDevice s) {
    if ((indices.dtype() != mx::int32 && indices.dtype() != mx::uint32) || weight.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_embedding: indices and weight must use 32-bit integers");
    }
    if (scales.dtype() != biases.dtype() || (scales.dtype() != mx::float16 && scales.dtype() != mx::bfloat16)) {
        throw std::runtime_error("quantized_embedding: scales and biases must have the same 16-bit dtype");
    }
    if (group_size != 128 || bits != 4 || scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_embedding: expected 4-bit weights with group size 128");
    }
    const int dim = weight.shape()[1] * (32 / bits);
    if (scales.shape()[0] != weight.shape()[0] || scales.shape()[1] != dim / group_size) {
        throw std::runtime_error("quantized_embedding: incompatible parameter shapes");
    }
    auto out_shape = indices.shape();
    out_shape.push_back(dim);
    return mx::array(out_shape, scales.dtype(), std::make_shared<QuantizedEmbedding>(to_stream(s)),
                     {indices, scales, biases, weight});
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("quantized_embedding: the course extension is GPU-only");
}

#ifdef _METAL_
void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &indices = inputs[0];
    const auto &scales = inputs[1];
    const auto &biases = inputs[2];
    const auto &weight = inputs[3];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    const bool unsigned_indices = indices.dtype() == mx::uint32;
    const char *kernel_name;
    if (out.dtype() == mx::float16) {
        kernel_name =
            unsigned_indices ? "quantized_embedding_w4a16_g128_f16_u32" : "quantized_embedding_w4a16_g128_f16_i32";
    } else {
        kernel_name =
            unsigned_indices ? "quantized_embedding_w4a16_g128_bf16_u32" : "quantized_embedding_w4a16_g128_bf16_i32";
    }
    auto kernel = d.get_kernel(kernel_name, d.get_library("tiny_llm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(indices, 0);
    encoder.set_input_array(scales, 1);
    encoder.set_input_array(biases, 2);
    encoder.set_input_array(weight, 3);
    encoder.set_output_array(out, 4);
    const int tokens = indices.size();
    const int dim = out.shape().back();
    encoder.set_bytes(tokens, 5);
    encoder.set_bytes(dim, 6);
    const int threads = std::min<int>(kernel->maxTotalThreadsPerThreadgroup(), 256);
    encoder.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(threads, 1, 1));
}
#else
void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("QuantizedEmbedding has no GPU implementation.");
}
#endif

}  // namespace tiny_llm_ext
