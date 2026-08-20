# minivllm

A high-performance LLM inference and serving engine built from scratch in Python and MLX, optimized for Apple Silicon (M4 Pro). 

This repository contains my personal, from-scratch implementation of the `tiny-llm` course, working step-by-step from fundamental matrix operations to a fully-functioning serving engine with continuous batching, custom kernels, and paged attention.

---

## 🚀 Features & Progress

### 🧱 Core Architecture & Mathematical Operators (`basics.py`, `layer_norm.py`, `positional_encoding.py`)
- **Rotary Position Embeddings (RoPE)**: High-performance frequency-based token positional encoding.
- **RMSNorm**: Efficient Root Mean Square normalization.
- **SwiGLU**: Fused gated linear unit activation function.

### 🧠 Attention Mechanisms (`attention.py`)
- **Grouped Query Attention (GQA)**: Custom implementation mapping Qwen3-4B structures.
- **Causal Masking**: Scaled dot-product attention with causal masking.

### ⚡ Custom Kernels & Quantization (`quantize.py`, `week2_kernels.py`)
- **Quantization**: 4-bit and 8-bit model weight quantization.
- **Fused Operators**: Model kernels fusing RMSNorm, RoPE, SwiGLU, and Decode Attention to improve memory bandwidth utilisation.

### 💾 Memory & Serving Infrastructure (`kv_cache.py`, `paged_kv_cache.py`, `batch.py`)
- **Paged KV Cache**: Dynamic allocation of key-value cache blocks (similar to vLLM) to prevent memory fragmentation.
- **Continuous Batching**: Request scheduler supporting continuous batching, chunked prefill, and dynamic scheduling.
- **Top-P/Top-K Sampler**: Sampling algorithms with temperature scaling.

---

## 📊 Benchmarks (Apple M4 Pro)

The following benchmark values are evaluated on **Qwen3-4B** (Input: 128 tokens, Output: 129 tokens) showing the performance progression of optimization stages compared to the MLX reference baseline:

| Optimization Step | Prefill Latency (ms) | Decode Latency (ms) | Output Rate (tokens/s) |
| :--- | :---: | :---: | :---: |
| **2.1 KV Cache** | 730.43 | 24.62 | 24.00 |
| **2.3 Quantized Matvec** | 105.00 | 58.70 | 37.94 |
| **2.4 Fast RMSNorm** | 104.98 | 65.94 | 40.81 |
| **2.4 + Fast RoPE** | 105.39 | 71.16 | 42.81 |
| **2.4 + Fused SwiGLU** | 105.96 | 75.21 | 44.32 |
| **2.5 Decode Attention** | 105.98 | 75.74 | 44.50 |
| **2.6 SIMD Matrix Prefill** | 797.45 | 75.12 | 69.17 |
| **2.7 Split-K Prefill** | 792.54 | 75.40 | 69.37 |
| **MLX Baseline** | 830.49 | 89.37 | 81.30 |

---

## 🛠 Setup & Usage

### Linking to a parent workspace
This project is typically linked inside the `tiny-llm` training directory as a symlink:
```bash
ln -s /path/to/minivllm /path/to/tiny-llm/src/tiny_llm
```

### Running Tests
To verify implementation correctness:
```bash
pytest tests/
```
