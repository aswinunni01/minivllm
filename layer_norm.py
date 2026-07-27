import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x_fp32 = x.astype(mx.float32)
        mean = mx.mean(mx.power(x_fp32, 2), axis=-1, keepdims=True)
        mean = mean + self.eps
        sqrt_mean = mx.sqrt(mean)

        return ((x_fp32 / sqrt_mean) * self.weight.astype(mx.float32)).astype(orig_dtype)
        
        
