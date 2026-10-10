"""Compare candidate tolerance metrics for the bf16 attention test, on this machine.

Prints each metric for correct attention and for a version that computes the softmax in
bf16 (the bug the test exists to catch), so a metric can be chosen that separates them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from eqx_zoo.layers import dot_product_attention

rng = np.random.default_rng(0)
q_len, k_len, heads, dim = 64, 1024, 4, 64
q = rng.normal(size=(q_len, heads, dim)).astype(np.float32)
k = rng.normal(size=(k_len, heads, dim)).astype(np.float32)
v = rng.normal(size=(k_len, heads, dim)).astype(np.float32)

tq, tk, tv = (torch.tensor(a).bfloat16().transpose(0, 1) for a in (q, k, v))
scores = (tq @ tk.transpose(1, 2)) * dim**-0.5
probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
expected = (probs @ tv).transpose(0, 1).float().numpy()


def broken(q, k, v, mask):
    """Attention with the softmax computed in bf16: the bug the test must catch."""
    scores = jnp.einsum("qhd,khd->hqk", q, k) * q.shape[-1] ** -0.5
    scores = jnp.where(mask[None], scores, jnp.finfo(scores.dtype).min)
    return jnp.einsum("hqk,khd->qhd", jax.nn.softmax(scores, axis=-1).astype(v.dtype), v)


bf = lambda a: jnp.asarray(a).astype(jnp.bfloat16)  # noqa: E731
mask = jnp.ones((q_len, k_len), dtype=bool)
ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(expected), 1e-30))) - 7)
print("device:", jax.devices()[0])
for name, fn in [("correct", dot_product_attention), ("broken", broken)]:
    out = np.asarray(fn(bf(q), bf(k), bf(v), mask).astype(jnp.float32))
    steps = np.abs(out - expected) / ulp
    rel_rms = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    violations = int((~np.isclose(out, expected, rtol=2**-7, atol=2**-11)).sum())
    print(
        f"{name:8s} max steps {steps.max():9.1f}  mean steps {steps.mean():7.3f}  "
        f"relative RMS {rel_rms:.2e}  allclose violations {violations}"
    )
