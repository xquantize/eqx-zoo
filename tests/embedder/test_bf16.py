"""bfloat16 tests for decoder embedders, with the same method and limit as for encoders.

Our bf16 output is compared with float32 using the root-mean-square error, relative to the
reference's own bf16 error: Hugging Face for hidden states, sentence-transformers for
embeddings. Measured ratios (eager / JIT, on ARM): Qwen3-Embedding-0.6B hidden states
0.97x / 1.29x and embeddings 1.38x / 1.36x; the tiny embedder 1.05x / 1.16x.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from support.bf16 import MODES, check, rms, run

# As for encoders: catches gross bf16 errors, with room for platform rounding differences.
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
    args = (jnp.asarray(ref["st_input_ids"]), jnp.asarray(ref["st_attention_mask"]))
    ours = run(bf16_model, lambda m, i, a: jax.vmap(m.embed)(i, a), *args, jit=jit)
    yardstick = rms(ref["st_bf16_embeddings"], ref["st_embeddings"])
    check(case.name, "embedding", rms(ours, ref["st_embeddings"]), yardstick, FACTOR)
