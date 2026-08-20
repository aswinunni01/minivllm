from typing import Any

import mlx.core as mx

from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .positional_encoding import RoPE
from .layer_norm import RMSNorm
from .embedding import Embedding
from .kv_cache import TinyKvCache, TinyKvFullCache
from .quantize import QuantizedWeights, dequantize_linear
from .week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    scaled_dot_product_attention,
    swiglu,
)

WEEK2_CHECKPOINTS = (
    "kv-cache",
    "quantized-matvec",
    "rmsnorm",
    "rope",
    "swiglu",
    "decode-attention",
    "simd-matmul",
    "split-k",
)

DECODE_ATTENTION_MAX_CONTEXT = 256
DECODE_ATTENTION_MAX_QUERY = 2


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_decode_attention: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_decode_attention = use_decode_attention
        self.use_fast_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        
        # wq -> (B, Hq x D, E)
        self.wq = wq
        # wq -> (B, H x D, E)
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.q_norm = q_norm
        self.k_norm = k_norm
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms_norm_eps = rms_norm_eps

        self.rope = RoPE(head_dim, max_seq_len, theta)


    # x -> (B, L, E)
    # L -> length of current sequence
    # S -> length of sequence until now
    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        
        Q = linear(x, self.wq) # (B, L, Hq x D)
        K = linear(x, self.wk) # (B, L, H x D). Note this time it is L, not S because (S-L) will be retrieved from kv cache
        V = linear(x, self.wv) # (B, L, H x D)

        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)
        K = K.reshape(K.shape[0], K.shape[1], self.num_kv_heads, self.head_dim)
        V = V.reshape(V.shape[0], V.shape[1], self.num_kv_heads, self.head_dim)
        Q_rmsnorm = RMSNorm(self.head_dim, self.q_norm, self.rms_norm_eps)
        Q = Q_rmsnorm(Q)

        K_rmsnorm = RMSNorm(self.head_dim, self.k_norm, self.rms_norm_eps)
        K = K_rmsnorm(K)

        Q = self.rope(Q, slice(offsets, offsets+Q.shape[1])) # (B, L, Hq x D)
        K = self.rope(K, slice(offsets, offsets+K.shape[1]))

        

        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        K, V, _, _ = cache.update_and_fetch(K, V) # (B, S, H x D)

        weighted_values = scaled_dot_product_attention_grouped(Q, K, V, mask=mask) # (B, Hq, L, D)
        weighted_values = weighted_values.swapaxes(2, 1)
        weighted_values = weighted_values.reshape(Q.shape[0], Q.shape[2], self.num_heads*self.head_dim) #(B, L, Hq X D)

        output = linear(weighted_values, self.wo) # (B, L, E)

        return output


        
        


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        use_fast_swiglu: bool = True,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu
        

    def __call__(self, x: mx.array) -> mx.array:
        
        up = linear(x, self.w_up)
        gate = silu(linear(x, self.w_gate))
        intermediate = up * gate


        out = linear(intermediate, self.w_down)

        return out

        


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_decode_attention: bool = True,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_fast_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        self.use_fast_swiglu = use_fast_swiglu
        self.use_decode_attention = use_decode_attention

        self.rms_norm_eps = rms_norm_eps
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.q_norm = q_norm
        self.k_norm = k_norm

        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

        self.w_input_layernorm = w_input_layernorm
        self.w_post_attention_layernorm = w_post_attention_layernorm

        self.input_layernorm = RMSNorm(self.head_dim, self.w_input_layernorm, self.rms_norm_eps)
        self.self_attn = Qwen3MultiHeadAttention(self.hidden_size, self.num_attention_heads, self.num_kv_heads, self.head_dim, self.wq, self.wk, self.wv, self.wo, self.q_norm, self.k_norm, self.max_seq_len, self.theta, self.rms_norm_eps, self.use_fast_rms_norm, self.use_fast_rope, self.use_decode_attention)

        self.mlp = Qwen3MLP(self.head_dim, self.hidden_size, self.w_gate, self.w_up, self.w_down, self.use_fast_swiglu)

        self.post_attention_layernorm = RMSNorm(self.head_dim, self.w_post_attention_layernorm, self.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        
        input_norm = self.input_layernorm(x)

        attention_out = self.self_attn(input_norm, offset, cache, mask)

        x = x + attention_out

        post_attn_norm = self.post_attention_layernorm(x)

        mlp_out = self.mlp(post_attn_norm)

        out = mlp_out + x

        return out


class Qwen3ModelWeek2:
    def __init__(self, mlx_model: Any, checkpoint: str = "split-k"):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers

        checkpoint_index = WEEK2_CHECKPOINTS.index(checkpoint)
        self.use_fast_rms_norm = checkpoint_index >= WEEK2_CHECKPOINTS.index("rmsnorm")
        self.use_fast_rope = checkpoint_index >= WEEK2_CHECKPOINTS.index("rope")
        self.use_fast_swiglu = checkpoint_index >= WEEK2_CHECKPOINTS.index("swiglu")
        self.use_decode_attention = checkpoint_index >= WEEK2_CHECKPOINTS.index("decode-attention")


        self.vocab_size = mlx_model.args.vocab_size
        self.embedding_dim = mlx_model.args.hidden_size

        self.num_attention_heads = mlx_model.args.num_attention_heads
        self.num_kv_heads = mlx_model.args.num_key_value_heads
        self.head_dim = mlx_model.args.head_dim
        self.intermediate_size = mlx_model.args.intermediate_size

        self.rms_norm_eps = mlx_model.args.rms_norm_eps
        self.theta = mlx_model.args.rope_theta

        self.max_seq_len = mlx_model.args.max_position_embeddings

        self.layers_inner = []
        
        self.embedding = Embedding(self.vocab_size, self.embedding_dim, dequantize_linear(mlx_model.model.embed_tokens))


        for i in range(self.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj)

            q_norm = mlx_model.model.layers[i].self_attn.q_norm.weight
            k_norm = mlx_model.model.layers[i].self_attn.k_norm.weight

            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj)
            w_up =   dequantize_linear(mlx_model.model.layers[i].mlp.up_proj)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj)

            w_input_layernorm = mlx_model.model.layers[i].input_layernorm.weight
            w_post_attention_layernorm = mlx_model.model.layers[i].post_attention_layernorm.weight

            layer = Qwen3TransformerBlock(self.num_attention_heads, self.num_kv_heads, self.embedding_dim, self.head_dim, self.intermediate_size, self.rms_norm_eps, wq, wk, wv, wo, q_norm, k_norm, w_gate, w_up, w_down, w_input_layernorm, w_post_attention_layernorm, self.max_seq_len, self.theta, self.use_fast_rms_norm, self.use_fast_rope, self.use_fast_swiglu, self.use_decode_attention)
            self.layers_inner.append(layer)
        
        self.output_layernorm = RMSNorm(self.embedding_dim, mlx_model.model.norm.weight, self.rms_norm_eps)
        
        if(not mlx_model.args.tie_word_embeddings ):
            self.out_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.out_lm_head = None


    def create_kv_cache(self) -> list[TinyKvCache]:
        kv_caches = []
        for i in range(self.num_hidden_layers):
            kv_cache = TinyKvFullCache()
            kv_caches.append(kv_cache)

        return kv_caches


    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        
        x = self.embedding(inputs)

        for i in range(self.num_hidden_layers):
            layer = self.layers_inner[i]
            cache_i = cache[i]
            cache_i_offset = getattr(cache_i, "offset", None)
            if(cache_i_offset is not None and cache_i_offset != offset):
                raise ValueError(f"layer {i} cache offset {cache_i_offset} does not match model offset {offset}")
            x = layer(x, offset, cache_i, mask="causal")

        x = self.output_layernorm(x)

        if(self.out_lm_head is None):
            x = self.embedding.as_linear(x)
        else:
            x = linear(x, self.out_lm_head)

        return x
