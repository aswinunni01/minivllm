import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    dot = mx.matmul(x, w.T) 
    if(bias is not None):
        dot = dot + bias

    return dot


def silu(x: mx.array) -> mx.array:
    z = mx.exp(-mx.abs(x))

    condition = (x >= 0)

    sigmoid = mx.where(condition, 1/(1+z), z/(1+z))

    return x * sigmoid
