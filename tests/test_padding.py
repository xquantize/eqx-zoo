"""Padding tests: left-padded inputs must behave exactly like unpadded ones."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

PAD = 3  # number of padding tokens prepended
TOL = dict(rtol=1e-3, atol=1e-2)  # as in the logits parity tests


def left_pad(ids):
    """Prepend `PAD` padding tokens, returning the ids and their attention mask."""
    padded = jnp.concatenate([jnp.zeros(PAD, dtype=ids.dtype), ids])
    return padded, jnp.arange(padded.shape[0]) >= PAD


def test_left_padding_matches_unpadded(case):
    padded, mask = left_pad(jnp.asarray(case.reference["input_ids"]))
    forward = eqx.filter_jit(lambda m, x, a: m(x, attention_mask=a)[0])
    logits = forward(case.model, padded, mask)
    np.testing.assert_allclose(logits[PAD:], case.reference["logits"], **TOL)


def test_left_padded_prefill_then_decode(case):
    ids = jnp.asarray(case.reference["input_ids"])
    padded, mask = left_pad(ids[:2])
    cache = case.model.init_cache(padded.shape[0] + ids.shape[0] - 2)

    logits, cache = case.model(padded, cache, attention_mask=mask)  # padded prefill
    rows = [logits[PAD:]]
    for t in range(2, ids.shape[0]):  # then decode real tokens one at a time
        logits, cache = case.model(ids[t : t + 1], cache)
        rows.append(logits)

    np.testing.assert_allclose(jnp.concatenate(rows), case.reference["logits"], **TOL)
