import mlx.core as mx

from extensions import tiny_llm_ext

from .basics import softmax

# Placeholder handed to the extension when no mask is supplied: the primitive
# always binds four input buffers, so it needs a real (empty) array.
_NO_ATTENTION_MASK = mx.zeros((1,), dtype=mx.float32)


class FastRMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # One fused dispatch: reduction, normalization, and weight multiply
        # happen inside the kernel; accumulation runs in float.
        return tiny_llm_ext.rms_norm(
            mx.contiguous(x), mx.contiguous(self.weight.astype(x.dtype)), self.eps
        )


class FastRoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int | list[int] | mx.array = 0) -> mx.array:
        # Accept one scalar offset or one per batch row; normalize both cases
        # to an int32 array before the dispatch so the kernel reads offsets[b].
        batch_size = x.shape[0]
        if isinstance(offset, int):
            offset = mx.full((batch_size,), offset, dtype=mx.int32)
        elif isinstance(offset, list):
            if len(offset) != batch_size:
                raise ValueError("FastRoPE needs one offset per batch row")
            offset = mx.array(offset, dtype=mx.int32)
        elif offset.ndim == 0:
            offset = mx.broadcast_to(offset.astype(mx.int32), (batch_size,))
        elif offset.shape != (batch_size,):
            raise ValueError("FastRoPE needs one offset per batch row")
        return tiny_llm_ext.rope(
            mx.contiguous(x),
            mx.contiguous(offset.astype(mx.int32)),
            self.dims,
            self.base,
            self.traditional,
        )


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    # gate / (1 + exp(-gate)) * up in a single element-wise kernel pass.
    return tiny_llm_ext.swiglu(mx.contiguous(gate), mx.contiguous(up))


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    # Readable grouped-query attention used as the correctness twin of the
    # custom decode kernel; reshapes fold GQA repeats into broadcast dims.
    expected_shape = query.shape
    batch_shape = query.shape[:-3]
    num_heads, query_length, head_dim = query.shape[-3:]
    num_kv_heads, context_length, _ = key.shape[-3:]
    if key.shape != value.shape or num_heads % num_kv_heads != 0:
        raise ValueError("incompatible grouped-query attention shapes")

    repeats = num_heads // num_kv_heads
    query = query.reshape(
        *batch_shape, -1, num_kv_heads, repeats, query_length, head_dim
    )
    key = key.reshape(*batch_shape, -1, num_kv_heads, 1, context_length, head_dim)
    value = value.reshape(*batch_shape, -1, num_kv_heads, 1, context_length, head_dim)
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * mx.array(scale, dtype=query.dtype)
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unsupported attention mask: {mask}")
        causal = mx.tril(
            mx.ones((query_length, context_length)), k=context_length - query_length
        )
        scores = scores + mx.where(causal, 0, -mx.inf).astype(scores.dtype)
    elif mask is not None:
        mask = mx.broadcast_to(
            mask, (*batch_shape, num_heads, query_length, context_length)
        )
        scores = scores + mask.reshape(
            *batch_shape, -1, num_kv_heads, repeats, query_length, context_length
        )
    return mx.matmul(softmax(scores, axis=-1), value).reshape(expected_shape)


def decode_attention_custom(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    mask: mx.array | str | None = None,
) -> mx.array:
    # Online-softmax decode kernel: flattens to [B*Hq, L, D] rows, groups KV
    # heads by the GQA ratio, and lets the extension handle masking.
    batch_size, num_heads, query_length, head_dim = query.shape
    key_batch_size, num_kv_heads, context_length, key_head_dim = key.shape
    if batch_size != key_batch_size or key.shape != value.shape:
        raise ValueError("query, key, and value batch dimensions must match")
    if head_dim != key_head_dim or num_heads % num_kv_heads != 0:
        raise ValueError("incompatible grouped-query attention shapes")
    if isinstance(mask, str) and mask != "causal":
        raise ValueError(f"unsupported attention mask: {mask}")

    query = mx.contiguous(query.reshape(batch_size * num_heads, query_length, head_dim))
    key = mx.contiguous(
        key.reshape(batch_size * num_kv_heads, context_length, head_dim)
    )
    value = mx.contiguous(
        value.reshape(batch_size * num_kv_heads, context_length, head_dim)
    )

    is_causal = isinstance(mask, str) and mask == "causal"
    has_mask = isinstance(mask, mx.array)
    if has_mask:
        mask = mx.broadcast_to(
            mask, (batch_size, num_heads, query_length, context_length)
        )
        mask = mx.contiguous(
            mask.astype(mx.float32).reshape(
                batch_size * num_heads, query_length, context_length
            )
        )
    else:
        mask = _NO_ATTENTION_MASK

    result = tiny_llm_ext.decode_attention(
        query,
        key,
        value,
        mask,
        scale,
        is_causal,
        has_mask,
        num_heads,
        num_kv_heads,
    )
    return result.reshape(batch_size, num_heads, query_length, head_dim)
