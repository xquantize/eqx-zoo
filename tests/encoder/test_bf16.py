"""bfloat16 tests for encoders: about as accurate as the reference libraries' own bf16.

As for causal LMs, our bf16 output is compared with float32 using the root-mean-square
error, relative to the reference's own bf16 error on the same input: Hugging Face for hidden
states, sentence-transformers for embeddings. Encoders are well conditioned in bf16: measured
ratios are 0.84-1.25 in both eager and JIT-compiled execution, so one limit serves both.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Our bf16 RMS error may be at most this multiple of the reference's own bf16 RMS error.
# Measured worst case for correct code: 1.25 (multilingual-e5-base embeddings, eager).
# Computing LayerNorm statistics in bf16 instead of float32 raises multilingual-e5-base to
# about 1.6x and fails these tests; smaller models are barely affected by that bug.
FACTOR = 1.5
MODES = [pytest.param(False, id="eager"), pytest.param(True, id="jit")]


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def check(name: str, what: str, error: float, yardstick: float) -> None:
    ratio = error / yardstick
    assert ratio <= FACTOR, (
        f"{name}: bf16 {what} RMS error {error:.3g} is {ratio:.2f}x the reference's own bf16 "
        f"error ({yardstick:.3g}); the limit is {FACTOR}x"
    )


def run(model, fn, *args, jit: bool) -> np.ndarray:
    """Apply `fn(model, *args)` JIT-compiled, or genuinely op by op, in float32 output."""
    if jit:
        out = eqx.filter_jit(fn)(model, *args)
    else:
        with jax.disable_jit():
            out = fn(model, *args)
    return np.asarray(out.astype(jnp.float32))


def test_bf16_parameters(bf16_model):
    params = jax.tree.leaves(eqx.filter(bf16_model, eqx.is_inexact_array))
    assert all(p.dtype == jnp.bfloat16 for p in params)


@pytest.mark.parametrize("jit", MODES)
def test_bf16_hidden_accuracy(case, bf16_model, jit):
    ref = case.reference
    ours = run(bf16_model, lambda m, x: m(x), jnp.asarray(ref["input_ids"]), jit=jit)
    check(case.name, "hidden", rms(ours, ref["hidden"]), rms(ref["bf16_hidden"], ref["hidden"]))


@pytest.mark.parametrize("jit", MODES)
def test_bf16_embedding_accuracy(case, bf16_model, jit):
    ref = case.reference
    if "st_bf16_embeddings" not in ref:
        pytest.skip("no sentence-transformers reference for this model")
    args = [jnp.asarray(ref[k]) for k in ("st_input_ids", "st_attention_mask", "st_token_type_ids")]
    ours = run(bf16_model, lambda m, i, a, t: jax.vmap(m.embed)(i, a, t), *args, jit=jit)
    check(
        case.name,
        "embedding",
        rms(ours, ref["st_embeddings"]),
        rms(ref["st_bf16_embeddings"], ref["st_embeddings"]),
    )
