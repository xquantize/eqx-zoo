"""Sparse mixture-of-experts feed-forward block."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from eqx_zoo.layers._common import linear
from eqx_zoo.layers.mlp import SwiGLU


@jax.custom_batching.custom_vmap
def batchable_ragged_dot(lhs, rhs, group_sizes):
    """`jax.lax.ragged_dot` that can be vmapped, for inference: not differentiable in reverse mode.

    Vmapped over rows that each have their own tokens and group sizes, it sorts the whole
    batch's tokens by group and computes them with one grouped matrix multiplication.
    """
    return jax.lax.ragged_dot(lhs, rhs, group_sizes)


@batchable_ragged_dot.def_vmap
def _batchable_ragged_dot_vmap(axis_size, in_batched, lhs, rhs, group_sizes):
    lhs_batched, rhs_batched, sizes_batched = in_batched
    if not lhs_batched:
        lhs = jnp.broadcast_to(lhs, (axis_size, *lhs.shape))
    if not sizes_batched:
        group_sizes = jnp.broadcast_to(group_sizes, (axis_size, *group_sizes.shape))
    if rhs_batched:
        # Different weights for each row: compute the rows one after another.
        rows = jax.lax.map(lambda a: jax.lax.ragged_dot(*a), (lhs, rhs, group_sizes))
        return rows, True
    batch, tokens, dim = lhs.shape
    groups = group_sizes.shape[-1]
    # Each row's tokens are sorted by group; give every token its group, then sort the whole
    # batch by group, so that one grouped multiplication covers every row.
    ids = jax.vmap(lambda s: jnp.repeat(jnp.arange(groups), s, total_repeat_length=tokens))(
        group_sizes
    )
    order = jnp.argsort(ids.reshape(-1), stable=True)
    sorted_out = jax.lax.ragged_dot(lhs.reshape(-1, dim)[order], rhs, group_sizes.sum(0))
    out = jnp.zeros_like(sorted_out).at[order].set(sorted_out)
    return out.reshape(batch, tokens, -1), True


class SparseMoE(eqx.Module):
    """Mixture of SwiGLU experts, with each token routed to its top-k experts.

    The router's scores are normalised with a float32 softmax; each token's top-k experts
    are applied, weighted by their (optionally renormalised) scores, and summed. Experts
    are computed together with grouped matrix multiplications (`jax.lax.ragged_dot`) over
    the token-expert assignments sorted by expert.

    Attributes:
        gate: Router projecting each token to one score per expert.
        experts: All experts as a single `SwiGLU` whose weights carry a leading expert axis.
        num_experts_per_tok: Number of experts each token is routed to.
        norm_topk_prob: Whether the selected experts' weights are renormalised to sum to one.
    """

    gate: eqx.nn.Linear
    experts: SwiGLU
    num_experts_per_tok: int = eqx.field(static=True)
    norm_topk_prob: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        key: PRNGKeyArray,
        dtype: DTypeLike = jnp.float32,
    ):
        """Create a randomly initialised block.

        Args:
            dim: Model (input and output) dimension.
            hidden_dim: Inner dimension of each expert.
            num_experts: Number of experts.
            num_experts_per_tok: Number of experts each token is routed to.
            norm_topk_prob: Whether to renormalise the selected experts' weights.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.

        Raises:
            ValueError: If `num_experts_per_tok` is not between 1 and `num_experts`.
        """
        if not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")
        gate_key, experts_key = jax.random.split(key)
        self.gate = linear(dim, num_experts, key=gate_key, dtype=dtype)
        make_expert = lambda k: SwiGLU(dim, hidden_dim, key=k, dtype=dtype)  # noqa: E731
        self.experts = eqx.filter_vmap(make_expert)(jax.random.split(experts_key, num_experts))
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
        """Apply the block to a sequence.

        Args:
            x: Hidden state of each token.

        Returns:
            The combined output of each token's selected experts.
        """
        k = self.num_experts_per_tok
        num_experts = self.gate.out_features

        # Route: float32 softmax over experts, then the top-k, as in the reference.
        logits = jax.vmap(self.gate)(x)
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        weights, experts = jax.lax.top_k(probs, k)
        if self.norm_topk_prob:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        weights = weights.astype(x.dtype)

        # Group the seq * k token-expert assignments by expert.
        flat_experts = experts.reshape(-1)
        order = jnp.argsort(flat_experts, stable=True)
        tokens = order // k
        group_sizes = jnp.bincount(flat_experts, length=num_experts)

        # Each expert's weights are stored as (out, in); `ragged_dot` wants (in, out).
        w = self.experts
        rows = x[tokens]
        gate = jax.lax.ragged_dot(rows, w.gate_proj.weight.transpose(0, 2, 1), group_sizes)
        up = jax.lax.ragged_dot(rows, w.up_proj.weight.transpose(0, 2, 1), group_sizes)
        hidden = jax.nn.silu(gate) * up
        out = jax.lax.ragged_dot(hidden, w.down_proj.weight.transpose(0, 2, 1), group_sizes)

        # Weight each assignment and sum it back into its token.
        out = out * weights.reshape(-1)[order][:, None]
        return jnp.zeros_like(x).at[tokens].add(out)
