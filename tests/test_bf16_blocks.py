"""Block-level bf16 precision tests.

Subtle precision bugs, such as dropping a float32 upcast, barely move model-level outputs
and can be hidden by compiler fusion inside a full model, but are unmistakable at the block
level. Each test compares a block in bf16 with the reference arithmetic, in units of bf16
rounding steps. LayerNorm's test lives in tests/test_pooling.py.
"""

import jax.numpy as jnp
import numpy as np
import torch

from eqx_zoo.layers import dot_product_attention

rng = np.random.default_rng(0)


def bf16_steps(out: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Absolute error in units of one bf16 rounding step at each expected value."""
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(expected), 1e-30))) - 7)
    return np.abs(out - expected) / ulp


def test_attention_bf16_matches_reference_arithmetic():
    """In bf16, attention must match Hugging Face's: float32 softmax, then a bf16 cast.

    Correct code is within 2 bf16 steps (almost every value is bit-identical); computing
    the softmax in bf16 instead is off by thousands of steps over 1,024 keys.
    """
    q_len, k_len, heads, dim = 64, 1024, 4, 64
    q = rng.normal(size=(q_len, heads, dim)).astype(np.float32)
    k = rng.normal(size=(k_len, heads, dim)).astype(np.float32)
    v = rng.normal(size=(k_len, heads, dim)).astype(np.float32)

    tq, tk, tv = (torch.tensor(a).bfloat16().transpose(0, 1) for a in (q, k, v))
    scores = (tq @ tk.transpose(1, 2)) * dim**-0.5
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    expected = (probs @ tv).transpose(0, 1).float().numpy()

    bf = lambda a: jnp.asarray(a).astype(jnp.bfloat16)  # noqa: E731
    mask = jnp.ones((q_len, k_len), dtype=bool)
    out = np.asarray(dot_product_attention(bf(q), bf(k), bf(v), mask).astype(jnp.float32))
    
    np.testing.assert_allclose(out, expected, rtol=2**-7, atol=2**-11)
