"""End-to-end parity tests against Hugging Face, for every registered model."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from eqx_zoo.layers import causal_mask
from eqx_zoo.models.gpt2 import GPT2
from eqx_zoo.models.gpt_neox import GPTNeoX

TOL = dict(rtol=1e-3, atol=1e-3)


def test_layer_by_layer(case):
    model, ref = case.model, case.reference
    ids = jnp.asarray(ref["input_ids"])
    positions = jnp.arange(ids.shape[0])
    # The mask the model builds itself, window included; only some `CausalLM`s have one.
    window = getattr(model.config, "sliding_window", None)
    mask = causal_mask(positions, positions, window=window)

    if isinstance(model, GPT2):
        # Hugging Face's `wte` gives the token embedding alone; positions are added after it.
        x = model.wte.weight[ids]
        np.testing.assert_array_equal(x, ref["embed"])
        x = x + model.wpe.weight[positions]
        for i, block in enumerate(model.h):
            x, _ = block(x, mask)
            np.testing.assert_allclose(
                x, ref[f"layer{i}"], **TOL, err_msg=f"{case.name}: layer {i}"
            )
        np.testing.assert_allclose(model.ln_f(x), ref["final_norm"], **TOL)
        return

    if isinstance(model, GPTNeoX):
        # Same wiring as the Llama-style decoders (positions, mask), under GPT-NeoX's names.
        embedding, layers, norm = (
            model.gpt_neox.embed_in,
            model.gpt_neox.layers,
            model.gpt_neox.final_layer_norm,
        )
    else:
        embedding, layers, norm = model.model.embed_tokens, model.model.layers, model.model.norm

    x = embedding.weight[ids]
    np.testing.assert_array_equal(x, ref["embed"])

    for i, layer in enumerate(layers):
        x, _ = layer(x, positions, mask)
        np.testing.assert_allclose(x, ref[f"layer{i}"], **TOL, err_msg=f"{case.name}: layer {i}")

    np.testing.assert_allclose(norm(x), ref["final_norm"], **TOL)


def test_logits(case):
    ref = case.reference
    logits = eqx.filter_jit(lambda m, ids: m(ids)[0])(case.model, jnp.asarray(ref["input_ids"]))
    np.testing.assert_allclose(logits, ref["logits"], rtol=1e-3, atol=1e-2)
    np.testing.assert_array_equal(logits.argmax(-1), ref["logits"].argmax(-1))


def test_last_only_logits(case):
    """Logits for the last position only match the last row of the full logits.

    Not bit-identical: multiplying one row takes a different path through the matrix library
    than multiplying every row. Measured on CI (x86): at most 3.2e-5 on logits reaching the
    tens (Llama 3.2 1B), under 2e-6 of the largest logit. Computing the wrong row is off by
    order one.
    """
    ids = jnp.asarray(case.reference["input_ids"])
    full, _ = case.model(ids)
    last, _ = case.model(ids, last_only=True)
    assert last.shape == (1, full.shape[-1])
    error = float(jnp.abs(last - full[-1:]).max() / jnp.abs(full[-1:]).max())
    assert error <= 1e-5, f"last-only logits differ by {error:.1e} of the largest logit"
