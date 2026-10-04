"""Parity tests for encoders against Hugging Face."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

TOL = dict(rtol=1e-4, atol=1e-4)


def test_layer_by_layer(case):
    model, ref = case.model, case.reference
    ids = jnp.asarray(ref["input_ids"])
    mask = jnp.ones((ids.shape[0], ids.shape[0]), dtype=bool)

    x = model.embeddings(ids, jnp.zeros_like(ids))
    np.testing.assert_allclose(x, ref["embed"], **TOL, err_msg=f"{case.name}: embeddings")
    for i, layer in enumerate(model.encoder.layer):
        x = layer(x, mask)
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

    embed = eqx.filter_jit(lambda m, i, a, t: jax.vmap(m.embed)(i, a, t))
    ours = embed(
        case.model,
        jnp.asarray(ref["st_input_ids"]),
        jnp.asarray(ref["st_attention_mask"]),
        jnp.asarray(ref["st_token_type_ids"]),
    )
    np.testing.assert_allclose(ours, ref["st_embeddings"], rtol=1e-4, atol=1e-5)
