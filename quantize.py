from typing import Any

import mlx.core as mx


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
    pass


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
    pass


def quantized_matmul_vanilla(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    pass


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    
    return quantized_matmul(w.scales, w.biases, w.group_size, w.bits, x, w.weight)

