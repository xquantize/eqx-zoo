"""Building blocks shared across model families."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


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


def rope_cos_sin(
    positions: Int[Array, " seq"], head_dim: int, theta: float
) -> tuple[Float[Array, "seq head_dim"], Float[Array, "seq head_dim"]]:
    """Rotary embedding tables in the rotate-half layout, computed in float32."""
    inv_freq = 1.0 / theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    freqs = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    angles = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(
    x: Float[Array, "seq heads head_dim"],
    cos: Float[Array, "seq head_dim"],
    sin: Float[Array, "seq head_dim"],
) -> Float[Array, "seq heads head_dim"]:
    """Rotate ``x`` by position, pairing dimension ``i`` with ``i + head_dim // 2``."""
    cos = cos.astype(x.dtype)[:, None, :]
    sin = sin.astype(x.dtype)[:, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


class Attention(eqx.Module):
    """Causal grouped-query attention with per-head query/key RMSNorm and RoPE."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    q_norm: RMSNorm
    k_norm: RMSNorm
    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    rope_theta: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        eps: float,
        key: PRNGKeyArray,
        dtype=jnp.float32,
    ):
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        q_key, k_key, v_key, o_key = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(dim, num_heads * head_dim, use_bias=False, key=q_key, dtype=dtype)
        self.k_proj = eqx.nn.Linear(dim, num_kv_heads * head_dim, use_bias=False, key=k_key, dtype=dtype)
        self.v_proj = eqx.nn.Linear(dim, num_kv_heads * head_dim, use_bias=False, key=v_key, dtype=dtype)
        self.o_proj = eqx.nn.Linear(num_heads * head_dim, dim, use_bias=False, key=o_key, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, eps=eps, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=eps, dtype=dtype)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta

    def __call__(
        self, x: Float[Array, "seq dim"], positions: Int[Array, " seq"]
    ) -> Float[Array, "seq dim"]:
        seq = x.shape[0]
        q = jax.vmap(self.q_proj)(x).reshape(seq, self.num_heads, self.head_dim)
        k = jax.vmap(self.k_proj)(x).reshape(seq, self.num_kv_heads, self.head_dim)
        v = jax.vmap(self.v_proj)(x).reshape(seq, self.num_kv_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_cos_sin(positions, self.head_dim, self.rope_theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        groups = self.num_heads // self.num_kv_heads
        k = jnp.repeat(k, groups, axis=1)
        v = jnp.repeat(v, groups, axis=1)

        scores = jnp.einsum("qhd,khd->hqk", q, k) / jnp.sqrt(self.head_dim).astype(q.dtype)
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", probs, v).reshape(seq, self.num_heads * self.head_dim)
        return jax.vmap(self.o_proj)(out)
