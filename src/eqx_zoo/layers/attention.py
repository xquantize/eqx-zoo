"""Grouped-query attention and its key/value cache."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from eqx_zoo.layers._common import linear
from eqx_zoo.layers.norm import RMSNorm
from eqx_zoo.layers.rope import Llama3RopeScaling, apply_rope, rope_cos_sin


class KVCache(eqx.Module):
    """Preallocated key/value buffers for one attention layer.

    Attributes:
        k: Cached keys, after normalisation and RoPE.
        v: Cached values.
    """

    k: Float[Array, "max_len kv_heads head_dim"]
    v: Float[Array, "max_len kv_heads head_dim"]

    @classmethod
    def empty(cls, max_len: int, num_kv_heads: int, head_dim: int, dtype: DTypeLike) -> "KVCache":
        """Allocate a zero-filled cache.

        Args:
            max_len: Number of token slots.
            num_kv_heads: Number of key/value heads.
            head_dim: Per-head dimension.
            dtype: Buffer dtype.

        Returns:
            An empty cache.
        """
        shape = (max_len, num_kv_heads, head_dim)
        return cls(k=jnp.zeros(shape, dtype=dtype), v=jnp.zeros(shape, dtype=dtype))

    def update(
        self,
        k: Float[Array, "seq kv_heads head_dim"],
        v: Float[Array, "seq kv_heads head_dim"],
        start: Int[Array, ""],
    ) -> "KVCache":
        """Write new keys and values into consecutive slots.

        Writes past the end are not checked: JAX clamps out-of-range indices, so the
        caller must allocate enough slots.

        Args:
            k: New keys.
            v: New values.
            start: Slot at which to write the first new token.

        Returns:
            The updated cache.
        """
        return KVCache(
            k=jax.lax.dynamic_update_slice_in_dim(self.k, k.astype(self.k.dtype), start, axis=0),
            v=jax.lax.dynamic_update_slice_in_dim(self.v, v.astype(self.v.dtype), start, axis=0),
        )


class Cache(eqx.Module):
    """Key/value caches for every layer of a decoder, plus the number of tokens seen.

    Attributes:
        layers: One `KVCache` per decoder layer.
        length: Number of tokens processed so far, i.e. the position of the next token.
    """

    layers: list[KVCache]
    length: Int[Array, ""]


class Attention(eqx.Module):
    """Causal grouped-query attention with RoPE, optional per-head q/k norm and q/k/v bias.

    Each key/value head is shared by `num_heads // num_kv_heads` consecutive query heads.
    When enabled, queries and keys are normalised per head before the rotary embedding.

    Attributes:
        q_proj: Query projection to `num_heads * head_dim`.
        k_proj: Key projection to `num_kv_heads * head_dim`.
        v_proj: Value projection to `num_kv_heads * head_dim`.
        o_proj: Output projection back to the model dimension.
        q_norm: RMSNorm applied to each query head, or `None` if disabled.
        k_norm: RMSNorm applied to each key head, or `None` if disabled.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimension.
        rope_theta: RoPE base frequency.
        rope_scaling: RoPE frequency scaling, or `None`.
    """

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    q_norm: RMSNorm | None
    k_norm: RMSNorm | None
    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    rope_theta: float = eqx.field(static=True)
    rope_scaling: Llama3RopeScaling | None = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        rope_scaling: Llama3RopeScaling | None = None,
        eps: float = 1e-6,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        key: PRNGKeyArray,
        dtype: DTypeLike = jnp.float32,
    ):
        """Create a randomly initialised attention block.

        Args:
            dim: Model (input and output) dimension.
            num_heads: Number of query heads.
            num_kv_heads: Number of key/value heads; must divide `num_heads`.
            head_dim: Per-head dimension; independent of `dim // num_heads`.
            rope_theta: RoPE base frequency.
            rope_scaling: Optional Llama 3 RoPE frequency scaling.
            eps: Epsilon for the query/key RMSNorms.
            qkv_bias: Whether the query, key and value projections have a bias.
            qk_norm: Whether to apply RMSNorm to each query and key head.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.

        Raises:
            ValueError: If `num_kv_heads` does not divide `num_heads`.
        """
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        q_key, k_key, v_key, o_key = jax.random.split(key, 4)
        q_dim, kv_dim = num_heads * head_dim, num_kv_heads * head_dim
        self.q_proj = linear(dim, q_dim, key=q_key, dtype=dtype, bias=qkv_bias)
        self.k_proj = linear(dim, kv_dim, key=k_key, dtype=dtype, bias=qkv_bias)
        self.v_proj = linear(dim, kv_dim, key=v_key, dtype=dtype, bias=qkv_bias)
        self.o_proj = linear(q_dim, dim, key=o_key, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, eps=eps, dtype=dtype) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps=eps, dtype=dtype) if qk_norm else None
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

    def __call__(
        self,
        x: Float[Array, "seq dim"],
        positions: Int[Array, " seq"],
        cache: KVCache | None = None,
    ) -> tuple[Float[Array, "seq dim"], KVCache | None]:
        """Attend causally over a sequence, optionally continuing from a cache.

        Args:
            x: Hidden states for each new token.
            positions: Absolute position of each new token. Must be consecutive; with a
                cache, the new keys and values are written starting at slot `positions[0]`.
            cache: Keys and values of earlier tokens, or `None` to attend only within `x`.

        Returns:
            The attention output for each new token (before the residual connection), and
            the updated cache, or `None` if no cache was given.
        """
        seq = x.shape[0]
        q = jax.vmap(self.q_proj)(x).reshape(seq, self.num_heads, self.head_dim)
        k = jax.vmap(self.k_proj)(x).reshape(seq, self.num_kv_heads, self.head_dim)
        v = jax.vmap(self.v_proj)(x).reshape(seq, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_cos_sin(positions, self.head_dim, self.rope_theta, self.rope_scaling)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is None:
            key_positions = positions
        else:
            cache = cache.update(k, v, positions[0])
            k, v = cache.k, cache.v
            key_positions = jnp.arange(k.shape[0])

        groups = self.num_heads // self.num_kv_heads
        k = jnp.repeat(k, groups, axis=1)
        v = jnp.repeat(v, groups, axis=1)

        scores = jnp.einsum("qhd,khd->hqk", q, k) * self.head_dim**-0.5
        causal = key_positions[None, :] <= positions[:, None]
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", probs, v).reshape(seq, self.num_heads * self.head_dim)
        return jax.vmap(self.o_proj)(out), cache
