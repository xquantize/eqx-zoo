"""Rotary position embeddings (RoPE)."""

import dataclasses

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


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
