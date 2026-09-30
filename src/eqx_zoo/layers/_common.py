"""Private helpers shared by the building blocks."""

import equinox as eqx
from jaxtyping import DTypeLike, PRNGKeyArray


def linear(
    in_dim: int, out_dim: int, *, key: PRNGKeyArray, dtype: DTypeLike, bias: bool = False
) -> eqx.nn.Linear:
    """Create a linear layer, without a bias unless requested."""
    return eqx.nn.Linear(in_dim, out_dim, use_bias=bias, key=key, dtype=dtype)
