from typing import Any

import mlx.core as mx

from extensions import tiny_llm_ext


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w.astype(mx.bfloat16)


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
    ):
        self.scales = scales       # (K, N/G) bfp16
        self.biases = biases       # (K, N/G) bfp16
        self.group_size = group_size
        self.bits = bits           # Number of quantization bits (4 bit / 16 bit)
        self.weight = weight        # (K, N/8) uint32
        self.use_simdgroup_matmul = use_simdgroup_matmul
        self.use_simdgroup_matvec = use_simdgroup_matvec
        self.use_split_k_matmul = use_split_k_matmul

    @staticmethod
    def from_mlx_layer(
        mlx_layer: Any,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
    ) -> "QuantizedWeights":
        biases = mlx_layer.biases
        return QuantizedWeights(
            scales=mlx_layer.scales.astype(mx.bfloat16),
            biases=None if biases is None else biases.astype(mx.bfloat16),
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
            use_simdgroup_matmul=use_simdgroup_matmul,
            use_simdgroup_matvec=use_simdgroup_matvec,
            use_split_k_matmul=use_split_k_matmul,
        )


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
    use_simdgroup: bool = False,
    use_split_k: bool = False,
) -> mx.array:
    # General dispatcher: always goes through our extension primitive, which
    # picks the SIMD-group matvec schedule for decode-shaped inputs (M <= 8)
    # and a matrix schedule otherwise. Leading dimensions of `a` are folded
    # into rows so the kernel only ever sees a 2D problem.
    *batch, D = a.shape
    a = a.reshape(-1, D)
    result = tiny_llm_ext.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(a),
        mx.contiguous(b),
        transpose_b,
        use_simdgroup,
        use_split_k,
    )
    return result.reshape(*batch, -1)


def dequantize_weights(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    group_size: int,
    bits: int,
) -> mx.array:
    
    # weight => *(... , K, N / 8)
    ## Convert this into (..., K, N) uint4s
    ## Shift by 0, 4, 8, 12, 16, 20, 24, 28 & 0b1111
    shift = mx.arange(32//bits) * bits # (8,)
    weight = weight[..., None] # (..., K, N/8, 1)
    mask = (1<<bits) - 1
    unpacked_weights = (weight >> shift) & mask
    unpacked_weights = unpacked_weights.reshape(*unpacked_weights.shape[:-2], unpacked_weights.shape[-2] * unpacked_weights.shape[-1]) # (..., K, N)
    scales = mx.repeat(scales, group_size, axis=-1)
    if biases is None:
        return unpacked_weights * scales
        
    biases = mx.repeat(biases, group_size, axis=-1)
    dequantized_weights = unpacked_weights * scales + biases
    return dequantized_weights


    





def quantized_matvec_custom(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    # Decode path (M <= 8). The extension defaults to use_simdgroup=True, so
    # this lands on the SIMD-group matvec kernel.
    *batch, D = a.shape
    a = a.reshape(-1, D)
    if a.shape[0] > 8:
        raise ValueError("quantized_matvec_custom supports at most 8 input rows")
    result = tiny_llm_ext.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(a),
        mx.contiguous(b),
        transpose_b,
    )
    return result.reshape(*batch, -1)


def quantized_matmul_vanilla(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    # Inspectable one-thread-per-output control; kept callable so the
    # optimized matvec can be compared directly against it.
    return quantized_matmul(
        scales,
        biases,
        group_size,
        bits,
        a,
        b,
        transpose_b,
        use_simdgroup=False,
    )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:

    # Decode-shaped rows (flattened M <= 8) take the SIMD-group matvec;
    # matrix-shaped prefill takes the general dispatcher.
    rows = 1
    for size in x.shape[:-1]:
        rows *= size

    if rows <= 8 and w.use_simdgroup_matvec:
        result = quantized_matvec_custom(
            w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
        )
    else:
        result = quantized_matmul(
            w.scales,
            w.biases,
            w.group_size,
            w.bits,
            x,
            w.weight,
            True,
            use_simdgroup=w.use_simdgroup_matmul,
            use_split_k=w.use_split_k_matmul,
        )

    if bias is not None:
        result = result + bias

    return result

