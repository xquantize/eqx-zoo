"""Benchmark load time, compile time and throughput of an eqx-zoo model.

Usage:
    uv run python benchmarks/generation.py Qwen/Qwen3-0.6B --dtype bfloat16
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from eqx_zoo import CausalLM, generate


def timed(fn, repeats: int = 1) -> float:
    """Best wall-clock time of `fn()` over `repeats` runs, waiting for all results."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo_id")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=64)
    args = parser.parse_args()

    dtype = getattr(jnp, args.dtype)
    model = None

    def load():
        nonlocal model
        model = CausalLM.from_pretrained(args.repo_id, dtype=dtype)
        return eqx.filter(model, eqx.is_array)

    load_s = timed(load)
    ids = jnp.arange(args.prompt_len) % model.config.vocab_size
    forward = eqx.filter_jit(lambda m, x: m(x)[0])

    prefill_first = timed(lambda: forward(model, ids))
    prefill_warm = timed(lambda: forward(model, ids), repeats=3)
    generate_first = timed(lambda: generate(model, ids, args.new_tokens))
    generate_warm = timed(lambda: generate(model, ids, args.new_tokens), repeats=3)
    decode_s = max(generate_warm - prefill_warm, 1e-9)

    print(f"| {args.repo_id} | {args.dtype} | prompt {args.prompt_len}, new {args.new_tokens} |")
    print("|---|---|---|")
    print(f"| Load | {load_s:.1f} s | |")
    print(f"| Prefill compile | {prefill_first - prefill_warm:.1f} s | |")
    print(f"| Prefill throughput | {args.prompt_len / prefill_warm:.0f} tok/s | |")
    print(f"| Generate compile | {generate_first - generate_warm:.1f} s | |")
    print(f"| Decode throughput | {args.new_tokens / decode_s:.1f} tok/s | |")


if __name__ == "__main__":
    main()
