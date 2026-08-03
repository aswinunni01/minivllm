import pytest
import mlx.core as mx
import mlx.nn as nn

from tiny_llm import quantize


def test_dequantize_weights():

    original_weights = mx.random.normal((3, 256), dtype=mx.float16)
    group_size = 128
    bits = 4

    packed_weights, scales, biases = mx.quantize(original_weights, group_size, bits)

    unpacked_weights = quantize.dequantize_weights(packed_weights, scales, biases, group_size, bits)
    print(packed_weights[0][1] , unpacked_weights[0][1])
    print(f"original_weights shape: {original_weights.shape} packed_weights shape: {packed_weights.shape} unpacked_weights shape: {unpacked_weights.shape}")
