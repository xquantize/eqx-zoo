"""The gated delta rule: linear attention with a decaying, error-correcting recurrent state."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _l2norm(x: Float[Array, "... dim"], eps: float = 1e-6) -> Float[Array, "... dim"]:
    """Scale vectors to unit length, as the reference does: x / sqrt(sum(x^2) + eps)."""
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def gated_delta_rule(
    q: Float[Array, "seq heads k_dim"],
    k: Float[Array, "seq heads k_dim"],
    v: Float[Array, "seq heads v_dim"],
    g: Float[Array, "seq heads"],
    beta: Float[Array, "seq heads"],
    state: Float[Array, "heads k_dim v_dim"] | None = None,
) -> tuple[Float[Array, "seq heads v_dim"], Float[Array, "heads k_dim v_dim"]]:
    """Apply the gated delta rule token by token (the recurrent form).

    Each head keeps a memory matrix S. For each token, S decays by exp(g), the value it
    predicts for the key is corrected towards the token's value (scaled by beta), and the
    query reads the result. Queries and keys are L2-normalised and queries scaled by
    1/sqrt(k_dim), as in the reference. Everything is computed in float32.

    Args:
        q: Queries.
        k: Keys.
        v: Values.
        g: Log decay of each head's memory at each token; at most zero.
        beta: Write strength of each token, between zero and one.
        state: Each head's memory before the first token, or `None` for zeros.

    Returns:
        The output for each token, in the queries' dtype, and the final memory in float32.
    """
    dtype = q.dtype
    q, k, v, g, beta = (x.astype(jnp.float32) for x in (q, k, v, g, beta))
    q = _l2norm(q) * q.shape[-1] ** -0.5
    k = _l2norm(k)
    if state is None:
        state = jnp.zeros((q.shape[1], q.shape[2], v.shape[2]), dtype=jnp.float32)

    def step(s, inputs):
        q_t, k_t, v_t, g_t, beta_t = inputs
        s = s * jnp.exp(g_t)[:, None, None]
        predicted = jnp.einsum("hkv,hk->hv", s, k_t)
        delta = (v_t - predicted) * beta_t[:, None]
        s = s + k_t[:, :, None] * delta[:, None, :]
        return s, jnp.einsum("hkv,hk->hv", s, q_t)

    state, out = jax.lax.scan(step, state.astype(jnp.float32), (q, k, v, g, beta))
    return out.astype(dtype), state
