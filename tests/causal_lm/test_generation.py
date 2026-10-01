"""Tests for cached decoding and generation, for every registered model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eqx_zoo import generate


def test_cache_matches_full_forward(case):
    model, ref = case.model, case.reference
    ids = jnp.asarray(ref["input_ids"])
    cache = model.init_cache(ids.shape[0])

    logits, cache = model(ids[:2], cache)  # prefill
    rows = [logits]
    for t in range(2, ids.shape[0]):  # then decode one token at a time
        logits, cache = model(ids[t : t + 1], cache)
        rows.append(logits)

    assert int(cache.length) == ids.shape[0]
    np.testing.assert_allclose(jnp.concatenate(rows), ref["logits"], rtol=1e-3, atol=1e-2)


def test_greedy_generation_matches_hf(case):
    expected = case.reference["generated"]
    tokens = generate(case.model, jnp.asarray(case.reference["input_ids"]), len(expected))
    np.testing.assert_array_equal(tokens, expected)


def test_sampling(case):
    ids = jnp.asarray(case.reference["input_ids"])
    tokens = generate(case.model, ids, 5, temperature=0.7, key=jax.random.key(0))
    assert tokens.shape == (5,)
    with pytest.raises(ValueError, match="PRNG key"):
        generate(case.model, ids, 5, temperature=0.7)
