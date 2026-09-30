"""Normalisation layers."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float


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
