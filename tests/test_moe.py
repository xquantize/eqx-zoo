"""Tests for the sparse mixture-of-experts block, against a direct per-token reference."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eqx_zoo.layers import SparseMoE

DIM, HIDDEN, EXPERTS = 16, 8, 6


def reference(moe: SparseMoE, x):
    """For each token, run its top-k experts one by one and sum their weighted outputs."""
    probs = jax.nn.softmax(jax.vmap(moe.gate)(x).astype(jnp.float32), axis=-1)
    outputs = []
    for t in range(x.shape[0]):
        weights, experts = jax.lax.top_k(probs[t], moe.num_experts_per_tok)
        if moe.norm_topk_prob:
            weights = weights / weights.sum()
        total = jnp.zeros(DIM)
        for weight, e in zip(weights, experts, strict=True):
            expert = jax.tree.map(lambda a, e=int(e): a[e], moe.experts)
            total = total + weight * expert(x[t])
        outputs.append(total)
    return jnp.stack(outputs)


@pytest.mark.parametrize("norm_topk_prob", [True, False])
@pytest.mark.parametrize("k", [1, 2, EXPERTS])
def test_matches_reference(k, norm_topk_prob):
    moe = SparseMoE(
        DIM,
        HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=k,
        norm_topk_prob=norm_topk_prob,
        key=jax.random.key(0),
    )
    x = jax.random.normal(jax.random.key(1), (7, DIM))
    np.testing.assert_allclose(moe(x), reference(moe, x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(eqx.filter_jit(moe)(x), reference(moe, x), rtol=1e-5, atol=1e-5)


def test_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        SparseMoE(
            DIM,
            HIDDEN,
            num_experts=4,
            num_experts_per_tok=5,
            norm_topk_prob=True,
            key=jax.random.key(0),
        )
