"""Building blocks shared across model families.

All modules operate on a single, unbatched sequence; use `jax.vmap` to batch them.
Attribute names mirror the corresponding Hugging Face checkpoint parameters so that
pretrained weights can be loaded by name.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray


def _linear(
    in_dim: int, out_dim: int, *, key: PRNGKeyArray, dtype: DTypeLike, bias: bool = False
) -> eqx.nn.Linear:
    return eqx.nn.Linear(in_dim, out_dim, use_bias=bias, key=key, dtype=dtype)


class RMSNorm(eqx.Module):
    """Root-mean-square layer normalisation over the last axis.

    The normalisation is computed in float32 and cast back to the input dtype before
    scaling, matching the Hugging Face reference implementation.

    Attributes:
        weight: Learnable per-feature scale.
        eps: Constant added to the mean square for numerical stability.
    """

    weight: Float[Array, " dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6, dtype: DTypeLike = jnp.float32):
        """Create a norm with its scale initialised to ones.

        Args:
            dim: Size of the normalised (last) axis.
            eps: Constant added to the mean square for numerical stability.
            dtype: Parameter dtype.
        """
        self.weight = jnp.ones(dim, dtype=dtype)
        self.eps = eps

    def __call__(self, x: Float[Array, "*batch dim"]) -> Float[Array, "*batch dim"]:
        """Normalise `x` over its last axis.

        Args:
            x: Input with any number of leading axes.

        Returns:
            The normalised and scaled input, in the input dtype.
        """
        dtype = x.dtype
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x.astype(dtype)


class SwiGLU(eqx.Module):
    """Gated feed-forward block computing `down(silu(gate(x)) * up(x))`.

    Attributes:
        gate_proj: Projection to the hidden dimension, passed through SiLU.
        up_proj: Projection to the hidden dimension, multiplied with the gate.
        down_proj: Projection back to the model dimension.
    """

    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(
        self, dim: int, hidden_dim: int, *, key: PRNGKeyArray, dtype: DTypeLike = jnp.float32
    ):
        """Create a randomly initialised block.

        Args:
            dim: Model (input and output) dimension.
            hidden_dim: Inner dimension of the gated projection.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        gate_key, up_key, down_key = jax.random.split(key, 3)
        self.gate_proj = _linear(dim, hidden_dim, key=gate_key, dtype=dtype)
        self.up_proj = _linear(dim, hidden_dim, key=up_key, dtype=dtype)
        self.down_proj = _linear(hidden_dim, dim, key=down_key, dtype=dtype)

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        """Apply the block to a single token.

        Args:
            x: One token's hidden state.

        Returns:
            The transformed hidden state.
        """
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclasses.dataclass(frozen=True)
class Llama3RopeScaling:
    """Llama 3 RoPE frequency scaling for extended context lengths.

    Low frequencies are divided by `factor`, high frequencies are unchanged, and the band
    in between is smoothly interpolated.

    Attributes:
        factor: Divisor applied to the lowest frequencies.
        low_freq_factor: Sets the wavelength above which frequencies are fully scaled.
        high_freq_factor: Sets the wavelength below which frequencies are unchanged.
        original_max_position_embeddings: Context length the model was pretrained with.
    """

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


def rope_inv_freq(
    head_dim: int, theta: float, scaling: Llama3RopeScaling | None = None
) -> Float[Array, " half_head_dim"]:
    """Compute RoPE inverse frequencies in float32, optionally with Llama 3 scaling.

    Args:
        head_dim: Per-head dimension; must be even.
        theta: RoPE base frequency.
        scaling: Optional Llama 3 frequency scaling.

    Returns:
        One inverse frequency per pair of rotated dimensions.
    """
    inv_freq = 1.0 / theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    if scaling is None:
        return inv_freq

    context = scaling.original_max_position_embeddings
    low_freq_wavelen = context / scaling.low_freq_factor
    high_freq_wavelen = context / scaling.high_freq_factor
    wavelen = 2 * jnp.pi / inv_freq

    scaled = jnp.where(wavelen > low_freq_wavelen, inv_freq / scaling.factor, inv_freq)
    smooth = (context / wavelen - scaling.low_freq_factor) / (
        scaling.high_freq_factor - scaling.low_freq_factor
    )
    smoothed = (1 - smooth) * scaled / scaling.factor + smooth * scaled
    medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return jnp.where(medium, smoothed, scaled)


def rope_cos_sin(
    positions: Int[Array, " seq"],
    head_dim: int,
    theta: float,
    scaling: Llama3RopeScaling | None = None,
) -> tuple[Float[Array, "seq head_dim"], Float[Array, "seq head_dim"]]:
    """Compute rotary embedding tables in the rotate-half layout, in float32.

    Args:
        positions: Absolute position of each token.
        head_dim: Per-head dimension; must be even.
        theta: RoPE base frequency.
        scaling: Optional Llama 3 frequency scaling.

    Returns:
        The cosine and sine tables, one row per position.
    """
    freqs = positions.astype(jnp.float32)[:, None] * rope_inv_freq(head_dim, theta, scaling)
    angles = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(
    x: Float[Array, "seq heads head_dim"],
    cos: Float[Array, "seq head_dim"],
    sin: Float[Array, "seq head_dim"],
) -> Float[Array, "seq heads head_dim"]:
    """Rotate each head of `x` by its token's position.

    Uses the rotate-half layout, pairing dimension `i` with `i + head_dim // 2`, as in
    Hugging Face transformers (not the interleaved layout of the original RoPE paper).

    Args:
        x: Per-head queries or keys.
        cos: Cosine table from `rope_cos_sin`.
        sin: Sine table from `rope_cos_sin`.

    Returns:
        The rotated input, in the input dtype.
    """
    cos = cos.astype(x.dtype)[:, None, :]
    sin = sin.astype(x.dtype)[:, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


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
        rope_scaling: RoPE frequency scaling, or None.
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
        self.q_proj = _linear(dim, q_dim, key=q_key, dtype=dtype, bias=qkv_bias)
        self.k_proj = _linear(dim, kv_dim, key=k_key, dtype=dtype, bias=qkv_bias)
        self.v_proj = _linear(dim, kv_dim, key=v_key, dtype=dtype, bias=qkv_bias)
        self.o_proj = _linear(q_dim, dim, key=o_key, dtype=dtype)
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
