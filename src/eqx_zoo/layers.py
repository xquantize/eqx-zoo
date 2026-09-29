"""Building blocks shared across model families."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class RMSNorm(eqx.Module):
    """Root-mean-square layer norm over the last axis.

    Normalisation is computed in float32 and cast back to the input dtype before
    scaling, matching the Hugging Face reference implementation.
    """

    weight: Float[Array, " dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6, dtype=jnp.float32):
        self.weight = jnp.ones(dim, dtype=dtype)
        self.eps = eps

    def __call__(self, x: Float[Array, "*batch dim"]) -> Float[Array, "*batch dim"]:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x.astype(dtype)


class SwiGLU(eqx.Module):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(self, dim: int, hidden_dim: int, *, key: PRNGKeyArray, dtype=jnp.float32):
        gate_key, up_key, down_key = jax.random.split(key, 3)
        self.gate_proj = eqx.nn.Linear(dim, hidden_dim, use_bias=False, key=gate_key, dtype=dtype)
        self.up_proj = eqx.nn.Linear(dim, hidden_dim, use_bias=False, key=up_key, dtype=dtype)
        self.down_proj = eqx.nn.Linear(hidden_dim, dim, use_bias=False, key=down_key, dtype=dtype)

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
