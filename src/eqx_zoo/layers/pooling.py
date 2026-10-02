"""Pooling per-token hidden states into one vector, as in sentence-transformers."""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def mean_pool(hidden: Float[Array, "seq dim"], mask: Bool[Array, " seq"]) -> Float[Array, " dim"]:
    """Average the hidden states of real tokens.

    Args:
        hidden: Hidden state of each token.
        mask: `True` for real tokens, `False` for padding.

    Returns:
        The mean over real tokens.
    """
    weights = mask.astype(hidden.dtype)[:, None]
    return (hidden * weights).sum(axis=0) / jnp.maximum(weights.sum(axis=0), 1e-9)


def cls_pool(hidden: Float[Array, "seq dim"], mask: Bool[Array, " seq"]) -> Float[Array, " dim"]:
    """Take the hidden state of the first real token, conventionally `[CLS]`.

    Args:
        hidden: Hidden state of each token.
        mask: `True` for real tokens, `False` for padding.

    Returns:
        The first real token's hidden state.
    """
    return hidden[jnp.argmax(mask)]


def l2_normalize(x: Float[Array, "*batch dim"], eps: float = 1e-12) -> Float[Array, "*batch dim"]:
    """Scale vectors to unit length, as `torch.nn.functional.normalize`.

    Args:
        x: Vectors along the last axis.
        eps: Lower bound on the norm, avoiding division by zero.

    Returns:
        `x` divided by its L2 norm.
    """
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)
