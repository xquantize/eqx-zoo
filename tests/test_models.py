"""End-to-end parity tests against Hugging Face, for every registered model."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from eqx_zoo.layers import causal_mask

TOL = dict(rtol=1e-3, atol=1e-3)


def test_layer_by_layer(case):
    model, ref = case.model, case.reference
    ids = jnp.asarray(ref["input_ids"])
    positions = jnp.arange(ids.shape[0])
    mask = causal_mask(positions, positions)

    x = model.model.embed_tokens.weight[ids]
    np.testing.assert_array_equal(x, ref["embed"])

    for i in range(model.config.num_hidden_layers):
        x, _ = model.model.layer(i)(x, positions, mask)
        np.testing.assert_allclose(x, ref[f"layer{i}"], **TOL, err_msg=f"{case.name}: layer {i}")

    np.testing.assert_allclose(model.model.norm(x), ref["final_norm"], **TOL)


def test_logits(case):
    ref = case.reference
    logits = eqx.filter_jit(lambda m, ids: m(ids)[0])(case.model, jnp.asarray(ref["input_ids"]))
    np.testing.assert_allclose(logits, ref["logits"], rtol=1e-3, atol=1e-2)
    np.testing.assert_array_equal(logits.argmax(-1), ref["logits"].argmax(-1))
