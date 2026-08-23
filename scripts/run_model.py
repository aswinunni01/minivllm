"""Load a Qwen3 model and chat with it using minivllm.

Requires the Metal extension to be built first (see README):
    pip install -e ".[build]"
    cd extensions && python build.py && cd ..

Examples:
    python scripts/run_model.py --prompt "Explain RoPE in one paragraph."
    python scripts/run_model.py --model Qwen/Qwen3-0.6B-MLX-4bit \
        --checkpoint split-k --max-tokens 128
    python scripts/run_model.py --week3 --prompt "Write a haiku about KV caches."
"""

import argparse

import mlx.core as mx
from mlx_lm import load


def greedy_generate(model, tokenizer, prompt: str) -> None:
    """Stream greedy decode through any Week 2 / Week 3 model wrapper."""
    kv_cache = model.create_kv_cache()
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    offset = 0
    try:
        while True:
            logits = model(tokens[None], offset, kv_cache, logits_to_keep=1)[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            token = int(mx.argmax(logprobs, axis=-1).item())
            if token == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token)
            print(detokenizer.last_segment, end="", flush=True)
            # First iteration is prefill: advance by the whole prompt length.
            offset += tokens.size
            tokens = mx.array([token])
        print()
    finally:
        for layer_cache in kv_cache:
            layer_cache.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B-MLX-4bit",
        help="Hugging Face model id or local path (4-bit MLX checkpoints work best)",
    )
    parser.add_argument(
        "--checkpoint",
        default="split-k",
        choices=("kv-cache", "quantized-matvec", "rmsnorm", "rope", "swiglu",
                 "decode-attention", "simd-matmul", "split-k"),
        help="Week 2 kernel checkpoint to enable (default: all of them)",
    )
    parser.add_argument("--week3", action="store_true",
                        help="use the Week 3 serving model (paged KV + paged attention)")
    parser.add_argument("--page-size", type=int, default=128,
                        help="page size for the Week 3 paged cache")
    parser.add_argument("--no-paged-attention", action="store_true",
                        help="Week 3: gather pages densely instead of direct paged attention")
    parser.add_argument("--prompt", required=True, help="the prompt to generate from")
    args = parser.parse_args()

    mlx_model, tokenizer = load(args.model)

    if args.week3:
        from tiny_llm.qwen3_week3 import Qwen3ModelWeek3

        model = Qwen3ModelWeek3(
            mlx_model,
            page_size=args.page_size,
            enable_paged_attention=not args.no_paged_attention,
        )
    else:
        from tiny_llm.qwen3_week2 import Qwen3ModelWeek2

        model = Qwen3ModelWeek2(mlx_model, checkpoint=args.checkpoint)

    greedy_generate(model, tokenizer, args.prompt)


if __name__ == "__main__":
    main()
