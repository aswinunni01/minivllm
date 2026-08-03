import mlx.core as mx
from .quantize import QuantizedWeights, dequantize_weights


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x]

    def as_linear(self, x: mx.array) -> mx.array:
        return mx.matmul(x, self.weight.T)


class QuantizedEmbedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: QuantizedWeights,
        use_custom_kernel: bool = False,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        embeddings_quantized = self.weight.weight[x]
        scales = self.weight.scales[x]
        biases = None if self.weight.biases is None else self.weight.biases[x]
        embeddings_dequantized = dequantize_weights(embeddings_quantized, scales, biases, self.weight.group_size, self.weight.bits)
        return embeddings_dequantized


    def as_linear(self, x: mx.array) -> mx.array:
        pass
