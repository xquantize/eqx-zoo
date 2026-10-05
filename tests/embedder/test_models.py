"""Parity tests for decoder embedders against Hugging Face and sentence-transformers."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eqx_zoo.layers import causal_mask

TOL = dict(rtol=1e-3, atol=1e-3)  # the residual stream across many layers, as for causal LMs


def test_layer_by_layer(case):
    model, ref = case.model, case.reference
    ids = jnp.asarray(ref["input_ids"])
    positions = jnp.arange(ids.shape[0])
    mask = causal_mask(positions, positions)

    x = model.model.embed_tokens.weight[ids]
    np.testing.assert_array_equal(x, ref["embed"])
    for i, layer in enumerate(model.model.layers):
        x, _ = layer(x, positions, mask)
        np.testing.assert_allclose(x, ref[f"layer{i}"], **TOL, err_msg=f"{case.name}: layer {i}")


def test_hidden_states(case):
    ref = case.reference
    hidden = eqx.filter_jit(lambda m, x: m(x))(case.model, jnp.asarray(ref["input_ids"]))
    np.testing.assert_allclose(hidden, ref["hidden"], **TOL)


def test_embeddings_match_sentence_transformers(case):
    ref = case.reference
    if "st_embeddings" not in ref:
        with pytest.raises(ValueError, match="sentence-transformers"):
            case.model.embed(jnp.asarray(ref["input_ids"]))
        return

    embed = eqx.filter_jit(lambda m, i, a: jax.vmap(m.embed)(i, a))
    ours = embed(
        case.model, jnp.asarray(ref["st_input_ids"]), jnp.asarray(ref["st_attention_mask"])
    )
    np.testing.assert_allclose(ours, ref["st_embeddings"], rtol=1e-4, atol=1e-5)
