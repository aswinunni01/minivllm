# minivllm

A high-performance LLM inference and serving engine built from scratch in Python and MLX, optimized for Apple Silicon. 

This repository contains my personal, from-scratch implementation of the `tiny-llm` course, working step-by-step from fundamental matrix operations to a fully-functioning serving engine with continuous batching, custom kernels, and paged attention.

---

## 🚀 Features & Progress

### 🧱 Core Architecture & Mathematical Operators (`basics.py`, `layer_norm.py`, `positional_encoding.py`)
- **Rotary Position Embeddings (RoPE)**: High-performance frequency-based token positional encoding.
- **RMSNorm**: Efficient Root Mean Square normalization.
- **SwiGLU**: Fused gated linear unit activation function.

### 🧠 Attention Mechanisms (`attention.py`)
- **Grouped Query Attention (GQA)**: Custom implementation mapping Qwen3-0.6B/4B structures.
- **Causal Masking**: Scaled dot-product attention with causal masking.

### 💾 Memory & Serving Infrastructure (`kv_cache.py`, `paged_kv_cache.py`, `batch.py`)
- **Paged KV Cache**: Dynamic allocation of key-value cache blocks (similar to vLLM) to prevent memory fragmentation.
- **Continuous Batching**: Request scheduler supporting continuous batching, chunked prefill, and dynamic scheduling.
- **Top-P/Top-K Sampler**: Sampling algorithms with temperature scaling.

---

## 📊 Benchmarks (Apple M1 Air, 16GB)

The following benchmark values are evaluated on **Qwen3-0.6B** (Input: 128 tokens, Output: 129 tokens, `prefill-logits=last`):

| Checkpoint | Prefill Throughput (tok/s) | Decode Throughput (tok/s) | Notes |
| :--- | :---: | :---: | :--- |
| **Week 1 (No Cache)** | 1007.65* | 4.69* | *Note: Evaluated with `prefill-logits=all` |
| **2.1 KV Cache** | 1028.09 | 32.38 | Hand-written KV cache implementation |
| **MLX Baseline** | 1069.72 | 108.84 | MLX reference baseline |

---

## 🛠 Setup & Usage

### Linking to a parent workspace
This project is typically linked inside the `tiny-llm` training directory as a symlink:
```bash
ln -s /path/to/minivllm /path/to/tiny-llm/src/tiny_llm
```

### Metal / C++ kernels
The MLX extension kernels (quantized matvec/matmul, fused RMSNorm/RoPE/SwiGLU,
decode attention, paged attention, FlashAttention prefill) live under
[`extensions/`](./extensions) - a synced mirror of `tiny-llm/src/extensions`,
where the course builds them.

### Running Tests
To verify implementation correctness:
```bash
pytest tests/
```
