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
        if case.model.pooling is not None:
            pytest.skip("pooling supplied to from_pretrained; see test_pooled_embeddings")
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


def test_pooled_embeddings(case):
    """Embeddings for checkpoints whose pooling comes from their model card."""
    ref = case.reference
    if "pooled_embeddings" not in ref:
        pytest.skip("pooling comes from the checkpoint's sentence-transformers pipeline")
    embed = eqx.filter_jit(lambda m, i, a, t: jax.vmap(m.embed)(i, a, t))
    ours = embed(
        case.model,
        jnp.asarray(ref["pooled_input_ids"]),
        jnp.asarray(ref["pooled_attention_mask"]),
        jnp.asarray(ref["pooled_token_type_ids"]),
    )
    np.testing.assert_allclose(ours, ref["pooled_embeddings"], rtol=1e-4, atol=1e-5)


def test_sentence_pair(case):
    """A (title, abstract) pair: the second segment's token type must be used."""
    ref = case.reference
    if "pair_hidden" not in ref:
        pytest.skip("no sentence-pair reference")
    ids, types = jnp.asarray(ref["pair_input_ids"]), jnp.asarray(ref["pair_token_type_ids"])
    np.testing.assert_allclose(
        case.model(ids, None, types), ref["pair_hidden"], rtol=1e-4, atol=1e-5
    )
    if types.any():
        # The check is sensitive to token types: ignoring them gives a clearly different result.
        ignored = case.model(ids, None, jnp.zeros_like(types))
        assert not np.allclose(ignored, ref["pair_hidden"], rtol=1e-4, atol=1e-5)
