"""The sliding-window attention mask, pinned to Transformers' own mask function."""

import jax.numpy as jnp
import numpy as np
import pytest

from eqx_zoo.layers import causal_mask


@pytest.mark.parametrize("window", [1, 2, 3, 5, 20])
@pytest.mark.parametrize("first_query", [0, 7])
def test_sliding_window_mask_matches_transformers(window, first_query):
    # `masking_utils` arrived in transformers 4.53; older versions can't serve as reference.
    masking_utils = pytest.importorskip("transformers.masking_utils")
    torch = pytest.importorskip("torch")
    reference = masking_utils.sliding_window_causal_mask_function(window)

    # `first_query > 0` is a cached continuation: queries at later slots, keys from slot 0.
    query_slots = jnp.arange(first_query, first_query + 9)
    key_slots = jnp.arange(16)
    ours = np.asarray(causal_mask(query_slots, key_slots, window=window))
    expected = np.array(
        [
            [bool(reference(0, 0, torch.tensor(int(q)), torch.tensor(int(k)))) for k in key_slots]
            for q in query_slots
        ]
    )
    np.testing.assert_array_equal(ours, expected)
    assert ours.any() and not ours.all()


def test_no_window_is_plain_causal():
    slots = jnp.arange(6)
    np.testing.assert_array_equal(
        np.asarray(causal_mask(slots, slots, window=None)), np.tril(np.ones((6, 6), dtype=bool))
    )
