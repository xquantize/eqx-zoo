"""Private helpers shared by the building blocks."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import DTypeLike, PRNGKeyArray


def linear(
    in_dim: int, out_dim: int, *, key: PRNGKeyArray, dtype: DTypeLike, bias: bool = False
) -> eqx.nn.Linear:
    """Create a linear layer, without a bias unless requested."""
    return eqx.nn.Linear(in_dim, out_dim, use_bias=bias, key=key, dtype=dtype)


def at_least_float32(dtype: DTypeLike) -> jnp.dtype:
    """The dtype PyTorch computes in natively: float32 for narrower inputs, else the input's.

    PyTorch's LayerNorm and softmax compute bfloat16 inputs in float32 internally but keep
    float64 in float64; casting to a fixed float32 would round float64 models.
    """
    return jnp.promote_types(dtype, jnp.float32)
