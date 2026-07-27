import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        inner = mx.arange(0, dims//2, dtype=mx.bfloat16) / (dims//2)
        freqs = mx.power(base, -inner)
        t = mx.arange(seq_len)
        freqs = mx.outer(t, freqs)

        self.sin_freqs = mx.sin(freqs)
        self.cos_freqs = mx.cos(freqs)

        self.base = base
        self.dims = dims
        self.traditional = traditional
        self.seq_len = seq_len

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:

        x1 = x[..., 0:self.dims//2:1] # 0 to half
        x2 = x[..., self.dims//2::1] # half to end
        N = x.shape[0]
        S = x.shape[1]
        H = x.shape[2]
        dims = x.shape[3]
        indices = slice(0, S) if offset is None else offset
        cos_freqs = self.cos_freqs[indices, :].reshape(1, S, 1, self.dims//2)
        sin_freqs = self.sin_freqs[indices, :].reshape(1, S, 1, self.dims//2)

        output1 = x1 * cos_freqs - x2 * sin_freqs
        output2 = x1 * sin_freqs + x2 * cos_freqs

        out = mx.concat([output1, output2], axis=-1)

        return out.reshape(N, S, H, dims).astype(mx.bfloat16)