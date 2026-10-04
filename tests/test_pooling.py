"""Tests for LayerNorm and the pooling functions used by encoders."""

import jax.numpy as jnp
import numpy as np
import torch

from eqx_zoo.layers import LayerNorm, cls_pool, l2_normalize, mean_pool

rng = np.random.default_rng(0)


def test_layernorm_matches_torch():
    x = rng.normal(size=(5, 16)).astype(np.float32)
    weight = rng.normal(size=16).astype(np.float32)
    bias = rng.normal(size=16).astype(np.float32)

    norm = LayerNorm(16, eps=1e-12)
    norm = norm.__class__.__new__(norm.__class__)
    object.__setattr__(norm, "weight", jnp.asarray(weight))
    object.__setattr__(norm, "bias", jnp.asarray(bias))
    object.__setattr__(norm, "eps", 1e-12)

    expected = torch.nn.functional.layer_norm(
        torch.tensor(x), (16,), torch.tensor(weight), torch.tensor(bias), eps=1e-12
    ).numpy()
    np.testing.assert_allclose(norm(jnp.asarray(x)), expected, rtol=1e-5, atol=1e-5)


def test_mean_pool_ignores_padding():
    hidden = rng.normal(size=(6, 4)).astype(np.float32)
    mask = np.array([True, True, True, False, False, False])
    expected = hidden[:3].mean(axis=0)
    np.testing.assert_allclose(
        mean_pool(jnp.asarray(hidden), jnp.asarray(mask)), expected, rtol=1e-6
    )


def test_cls_pool_takes_first_real_token():
    hidden = rng.normal(size=(6, 4)).astype(np.float32)
    right_padded = np.array([True, True, True, True, False, False])
    left_padded = np.array([False, False, True, True, True, True])
    np.testing.assert_array_equal(
        cls_pool(jnp.asarray(hidden), jnp.asarray(right_padded)), hidden[0]
    )
    np.testing.assert_array_equal(
        cls_pool(jnp.asarray(hidden), jnp.asarray(left_padded)), hidden[2]
    )


def test_l2_normalize():
    x = rng.normal(size=(3, 8)).astype(np.float32)
    out = np.asarray(l2_normalize(jnp.asarray(x)))
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(out, x / np.linalg.norm(x, axis=-1, keepdims=True), rtol=1e-6)
