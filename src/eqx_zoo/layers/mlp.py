"""Feed-forward blocks."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from eqx_zoo.layers._common import linear


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
        self.gate_proj = linear(dim, hidden_dim, key=gate_key, dtype=dtype)
        self.up_proj = linear(dim, hidden_dim, key=up_key, dtype=dtype)
        self.down_proj = linear(hidden_dim, dim, key=down_key, dtype=dtype)

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        """Apply the block to a single token.

        Args:
            x: One token's hidden state.

        Returns:
            The transformed hidden state.
        """
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
