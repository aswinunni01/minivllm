import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:

    if(scale is None):
        scale = 1/mx.sqrt(query.shape[-1])

    scores = mx.matmul(query, key.swapaxes(-2, -1))
    scores = scores * scale
    if(mask is not None):
        scores = scores + mask

    attn = mx.softmax(scores, axis=-1)

    weighted_values = mx.matmul(attn, value)

    return weighted_values

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.head_dim = hidden_size // num_heads

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.num_heads = num_heads
        self.hidden_size = hidden_size

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:

        Q = linear(query, self.wq)
        K = linear(key, self.wk)
        V = linear(value, self.wv)

        Q = Q.reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)
        K = K.reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        V = V.reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        Q = Q.swapaxes(-3, -2)
        K = K.swapaxes(-3, -2)
        V = V.swapaxes(-3, -2)

        weighted_v = scaled_dot_product_attention_simple(Q, K, V, mask=mask)

        weighted_v = weighted_v.swapaxes(-3, -2)
        weighted_v = weighted_v.reshape(query.shape[0], query.shape[1], self.num_heads * self.head_dim)

        output = linear(weighted_v, self.wo)
        return output


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    i = mx.arange(L, dtype=dtype).reshape(L, 1)
    j = mx.arange(S, dtype=dtype).reshape(1, S)

    mask_condition = j > i + (S-L)

    return mx.where(mask_condition, -mx.inf, 0.0)

# query -> (B, Hq, L, D)
# key -> (B, H, S, D)
# value -> (B,H, S, D)
def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    n_repeats = query.shape[-3] // key.shape[-3]
    
    # q_reshaped -> (B, H, n_repeats, L, D)
    q_reshaped = query.reshape(*query.shape[:-3], key.shape[-3], n_repeats, query.shape[-2], query.shape[-1])
    # key_reshaped -> (B, H, 1, S, D)
    key_reshaped = key.reshape(*key.shape[:-2], 1, key.shape[-2], key.shape[-1])
    # value_reshaped -> (B, H, 1, S, D)
    v_reshaped = value.reshape(*value.shape[:-2], 1, value.shape[-2], value.shape[-1])

    if mask is not None and not isinstance(mask, str):  
        mask = mask.reshape(*query.shape[:-3], key.shape[-3], n_repeats, query.shape[-2], key.shape[-2])
    if(mask is not None and mask == "causal"):
        mask = causal_mask(query.shape[-2], key.shape[-2], dtype=query.dtype)

    out =  scaled_dot_product_attention_simple(q_reshaped, key_reshaped, v_reshaped, scale = scale, mask=mask)

    return out.reshape(*query.shape).astype(mx.bfloat16)


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass
