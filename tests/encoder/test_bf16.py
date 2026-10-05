"""bfloat16 tests for encoders: about as accurate as the reference libraries' own bf16.

As for causal LMs, our bf16 output is compared with float32 using the root-mean-square
error, relative to the reference's own bf16 error on the same input: Hugging Face for hidden
states, sentence-transformers for embeddings. Encoders are well conditioned in bf16: measured
ratios are 0.84-1.51 in both eager and JIT-compiled execution, depending on the platform,
so one limit serves both.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from support.bf16 import MODES, check, rms, run

# Our bf16 RMS error may be at most this multiple of the reference's own bf16 RMS error.
# These model-level tests catch gross bf16 errors (wrong dtypes, missing casts) on any
# platform. Measured worst case for correct code: multilingual-e5-base embeddings, eager,
# 1.25x on ARM and 1.51x on x86, so the limit leaves room for platform rounding differences.
# Subtler precision bugs are too small to separate from those differences at this level,
# so LayerNorm's float32 statistics are tested directly in tests/test_pooling.py.
FACTOR = 2.0


def test_bf16_parameters(bf16_model):
    params = jax.tree.leaves(eqx.filter(bf16_model, eqx.is_inexact_array))
    assert all(p.dtype == jnp.bfloat16 for p in params)


@pytest.mark.parametrize("jit", MODES)
def test_bf16_hidden_accuracy(case, bf16_model, jit):
    ref = case.reference
    ours = run(bf16_model, lambda m, x: m(x), jnp.asarray(ref["input_ids"]), jit=jit)
    yardstick = rms(ref["bf16_hidden"], ref["hidden"])
    check(case.name, "hidden", rms(ours, ref["hidden"]), yardstick, FACTOR)


@pytest.mark.parametrize("jit", MODES)
def test_bf16_embedding_accuracy(case, bf16_model, jit):
    ref = case.reference
    if "st_bf16_embeddings" not in ref:
        pytest.skip("no sentence-transformers reference for this model")
    args = [jnp.asarray(ref[k]) for k in ("st_input_ids", "st_attention_mask", "st_token_type_ids")]
    ours = run(bf16_model, lambda m, i, a, t: jax.vmap(m.embed)(i, a, t), *args, jit=jit)
    yardstick = rms(ref["st_bf16_embeddings"], ref["st_embeddings"])
    check(case.name, "embedding", rms(ours, ref["st_embeddings"]), yardstick, FACTOR)
