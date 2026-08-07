import mlx.core as mx
from mlx_lm.models import qwen3

from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # wq -> (B, Hq x D, E)
        self.wq = wq
        # wq -> (B, H x D, E)
        self.wk = wk
        self.wv = wv
        self.wo = wo

        # Hold the readable norm objects (weights stay on .weight); Week 1
        # keeps the readable kernels while Week 2 adds the fast ones.
        self.q_norm = RMSNorm(head_dim, q_norm, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, k_norm, rms_norm_eps)
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms_norm_eps = rms_norm_eps

        self.rope = RoPE(head_dim, max_seq_len, theta)

    # x -> (B, L, E)
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        
        q = linear(x, self.wq) # -> (B, L, HqxD)
        k = linear(x, self.wk) # -> (B, L, HxD)
        v = linear(x, self.wv) # -> (B, L, HxD)

        # q -> (B, L, Hq, D)
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.head_dim)
        # k -> (B, L, H, D)
        k = k.reshape(k.shape[0], k.shape[1], self.num_kv_heads, self.head_dim)
        v = v.reshape(v.shape[0], v.shape[1], self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = self.rope(q)
        k = self.rope(k)

        q = q.swapaxes(-3, -2)
        k = k.swapaxes(-3, -2)
        v = v.swapaxes(-3, -2)
        x = scaled_dot_product_attention_grouped(q, k, v, mask=mask) # -> (B, Hq, L, D)
        
        x = x.swapaxes(-3, -2).reshape(x.shape[0], x.shape[2], self.num_heads * self.head_dim)

        return linear(x, self.wo) # (B, L, E)



class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hiddem_dim = hidden_dim 
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        
        up = linear(x, self.w_up)
        gate = silu(linear(x, self.w_gate))

        intermediate = up * gate

        return linear(intermediate, self.w_down)
        


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):

        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
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
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.multi_head_attn = Qwen3MultiHeadAttention(hidden_size, num_attention_heads, num_kv_heads, head_dim, wq, wk, wv, wo, q_norm, k_norm, max_seq_len, theta, rms_norm_eps)
        self.mlp = Qwen3MLP(head_dim, hidden_size, w_gate, w_up, w_down)
        
        self.input_layer_norm = RMSNorm(head_dim, w_input_layernorm, rms_norm_eps)
        self.post_attention_layer_norm = RMSNorm(head_dim, w_post_attention_layernorm, rms_norm_eps)


    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        input_layernorm = self.input_layer_norm(x)

        attn_output = self.multi_head_attn(input_layernorm, mask)
        
        x = x + attn_output # Residual connection

        x_layernorm = self.post_attention_layer_norm(x)

        mlp_output = self.mlp(x_layernorm)

        x = x + mlp_output

        return x


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.num_layers = mlx_model.args.num_hidden_layers
        
        self.vocab_size = mlx_model.args.vocab_size
        self.embedding_dim = mlx_model.args.hidden_size

        self.num_attention_heads = mlx_model.args.num_attention_heads
        self.num_kv_heads = mlx_model.args.num_key_value_heads
        self.head_dim = mlx_model.args.head_dim
        self.intermediate_size = mlx_model.args.intermediate_size

        self.rms_norm_eps = mlx_model.args.rms_norm_eps
        self.theta = mlx_model.args.rope_theta

        self.max_seq_len = mlx_model.args.max_position_embeddings

        self.layers = []
        
        self.embedding_layer = Embedding(self.vocab_size, self.embedding_dim, dequantize_linear(mlx_model.model.embed_tokens))


        for i in range(self.num_layers):
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

            layer = Qwen3TransformerBlock(self.num_attention_heads, self.num_kv_heads, self.embedding_dim, self.head_dim, self.intermediate_size, self.rms_norm_eps, wq, wk, wv, wo, q_norm, k_norm, w_gate, w_up, w_down, w_input_layernorm, w_post_attention_layernorm, self.max_seq_len, self.theta)
            self.layers.append(layer)
        
        self.output_layernorm = RMSNorm(self.embedding_dim, mlx_model.model.norm.weight, self.rms_norm_eps)
        
        if(not mlx_model.args.tie_word_embeddings ):
            self.out_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.out_lm_head = None


    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        out = self.embedding_layer(inputs)
        for layer in self.layers:
            out = layer(out, mask="causal")
        out = self.output_layernorm(out)
        if(self.out_lm_head is not None):
            out = linear(out, self.out_lm_head)
        else:
            out = self.embedding_layer.as_linear(out)
        return out