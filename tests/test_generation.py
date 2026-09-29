"""Tests for cached decoding and generation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eqx_zoo import generate


def test_cache_matches_full_forward(model, reference):
    ids = jnp.asarray(reference["input_ids"])
    cache = model.init_cache(ids.shape[0])

    logits, cache = model(ids[:2], cache)  # prefill
    rows = [logits]
    for t in range(2, ids.shape[0]):  # then decode one token at a time
        logits, cache = model(ids[t : t + 1], cache)
        rows.append(logits)

    assert int(cache.length) == ids.shape[0]
    np.testing.assert_allclose(jnp.concatenate(rows), reference["logits"], rtol=1e-3, atol=1e-2)


def test_greedy_generation_matches_hf(model, reference):
    expected = reference["generated"]
    tokens = generate(model, jnp.asarray(reference["input_ids"]), len(expected))
    np.testing.assert_array_equal(tokens, expected)


def test_sampling(model, reference):
    ids = jnp.asarray(reference["input_ids"])
    tokens = generate(model, ids, 5, temperature=0.7, key=jax.random.key(0))
    assert tokens.shape == (5,)
    with pytest.raises(ValueError, match="PRNG key"):
        generate(model, ids, 5, temperature=0.7)
