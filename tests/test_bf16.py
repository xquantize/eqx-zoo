"""bfloat16 tests: our bf16 must be about as accurate as Hugging Face's own bf16.

Two bf16 implementations round differently, so they are not compared with each other.
Both are compared with Hugging Face's float32 output using the root-mean-square error over
all logits, which is robust to individual unlucky roundings. Exact greedy agreement is not
asserted: in bf16 it legitimately drifts, even between Hugging Face's own bf16 and float32
runs.

Accuracy is checked in two modes. Eager execution has a low noise floor, so it is the sharp
check of our own numerics. JIT-compiled execution, the path `generate` runs, is currently
less accurate than eager (issue #11), so it gets a looser sanity bound.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Limits on our bf16 RMS error, as multiples of Hugging Face's own bf16 RMS error.
# Eager: measured worst case 1.30 (Llama 3.2 1B); computing RMSNorm without its float32
# upcast raises this to 2.36, so 1.5 separates correct code from that bug.
EAGER_FACTOR = 1.5
# JIT: measured worst case 2.43 (Llama 3.2 1B); see issue #11.
JIT_FACTOR = 3.0
# Hugging Face's float32 top-1 token must be among our bf16 top-k at every position.
TOP_K = 5


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def forward(model, ids, *, jit: bool = True) -> np.ndarray:
    fn = lambda m, x: m(x)[0]  # noqa: E731
    logits = (eqx.filter_jit(fn) if jit else fn)(model, ids)
    return np.asarray(logits.astype(jnp.float32))


def yardstick(reference) -> float:
    """Hugging Face's own bf16 RMS error against its float32 output."""
    return rms(reference["bf16_logits"], reference["logits"])


def test_bf16_parameters(bf16_model):
    params = jax.tree.leaves(eqx.filter(bf16_model, eqx.is_inexact_array))
    assert all(p.dtype == jnp.bfloat16 for p in params)


def test_bf16_accuracy_eager(case, bf16_model):
    ref = case.reference
    error = rms(forward(bf16_model, jnp.asarray(ref["input_ids"]), jit=False), ref["logits"])
    limit = EAGER_FACTOR * yardstick(ref)
    assert error <= limit, f"{case.name}: eager bf16 RMS error {error:.4g} exceeds {limit:.4g}"


def test_bf16_accuracy_jit(case, bf16_model):
    ref = case.reference
    error = rms(forward(bf16_model, jnp.asarray(ref["input_ids"])), ref["logits"])
    limit = JIT_FACTOR * yardstick(ref)
    assert error <= limit, f"{case.name}: JIT bf16 RMS error {error:.4g} exceeds {limit:.4g}"


def test_bf16_top1_in_top5(case, bf16_model):
    ref = case.reference
    ours = forward(bf16_model, jnp.asarray(ref["input_ids"]))
    top_k = np.argsort(ours, axis=-1)[:, -TOP_K:]
    expected = ref["logits"].argmax(-1)
    missing = [i for i, (e, row) in enumerate(zip(expected, top_k, strict=True)) if e not in row]
    assert not missing, f"{case.name}: fp32 top-1 not in bf16 top-{TOP_K} at positions {missing}"


def test_bf16_cache_matches_full_forward(case, bf16_model):
    ids = jnp.asarray(case.reference["input_ids"])
    step = eqx.filter_jit(lambda m, x, c: m(x, c))
    cache = bf16_model.init_cache(ids.shape[0])

    logits, cache = step(bf16_model, ids[:2], cache)  # prefill
    rows = [logits]
    for t in range(2, ids.shape[0]):  # then decode one token at a time
        logits, cache = step(bf16_model, ids[t : t + 1], cache)
        rows.append(logits)

    cached = np.asarray(jnp.concatenate(rows).astype(jnp.float32))
    error = rms(cached, forward(bf16_model, ids))
    limit = JIT_FACTOR * yardstick(case.reference)
    assert error <= limit, f"{case.name}: cached bf16 RMS error {error:.4g} exceeds {limit:.4g}"
