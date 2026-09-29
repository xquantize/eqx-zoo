"""End-to-end parity tests for Qwen3-0.6B against Hugging Face."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np


def test_layer_by_layer(model, reference):
    ids = jnp.asarray(reference["input_ids"])
    positions = jnp.arange(ids.shape[0])

    x = model.model.embed_tokens.weight[ids]
    np.testing.assert_array_equal(x, reference["embed"])

    for i, layer in enumerate(model.model.layers):
        x, _ = layer(x, positions)
        np.testing.assert_allclose(
            x, reference[f"layer{i}"], rtol=1e-3, atol=1e-3, err_msg=f"layer {i}"
        )

    np.testing.assert_allclose(model.model.norm(x), reference["final_norm"], rtol=1e-3, atol=1e-3)


def test_logits(model, reference):
    logits = eqx.filter_jit(lambda m, ids: m(ids)[0])(model, jnp.asarray(reference["input_ids"]))
    np.testing.assert_allclose(logits, reference["logits"], rtol=1e-3, atol=1e-2)
    np.testing.assert_array_equal(logits.argmax(-1), reference["logits"].argmax(-1))
