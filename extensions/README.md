# Metal / C++ extension sources

This directory is a **mirror** of `tiny-llm/src/extensions`, where the
course's extension actually lives and builds (`pdm run build-ext`). The
Python package at the repository root imports the compiled `_ext` module
from there, so keep the two in sync after editing either side:

    ./sync_from_tiny_llm.sh

Contents:
- `src/quantized_matmul.{cpp,metal}` - W4A16 g128 matvec/matmul schedules,
  SIMD-matrix prefill tile, Split-K accumulation + reduction, quantized
  embedding lookup
- `src/week2_kernels.{cpp,metal}`    - fused RMSNorm, RoPE, SwiGLU, and
  online-softmax decode attention
- `src/paged_attention.{cpp,metal}`  - paged cache slice writes, direct
  paged decode, scalar FP32 prefill, FlashAttention-style MMA BF16 prefill
- `bindings.cpp`, `tiny_llm_ext.h`   - nanobind bindings and MLX primitives
- `build.py`, `test.py`, `CMakeLists.txt` - mlx extension build system
