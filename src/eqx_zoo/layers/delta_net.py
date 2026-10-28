"""The gated delta rule: linear attention with a decaying, error-correcting recurrent state."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, PRNGKeyArray

from eqx_zoo.layers._common import linear


def _l2norm(x: Float[Array, "... dim"], eps: float = 1e-6) -> Float[Array, "... dim"]:
    """Scale vectors to unit length, as the reference does: x / sqrt(sum(x^2) + eps)."""
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def gated_delta_rule(
    q: Float[Array, "seq heads k_dim"],
    k: Float[Array, "seq heads k_dim"],
    v: Float[Array, "seq heads v_dim"],
    g: Float[Array, "seq heads"],
    beta: Float[Array, "seq heads"],
    state: Float[Array, "heads k_dim v_dim"] | None = None,
) -> tuple[Float[Array, "seq heads v_dim"], Float[Array, "heads k_dim v_dim"]]:
    """Apply the gated delta rule token by token (the recurrent form).

    Each head keeps a memory matrix S. For each token, S decays by exp(g), the value it
    predicts for the key is corrected towards the token's value (scaled by beta), and the
    query reads the result. Queries and keys are L2-normalised and queries scaled by
    1/sqrt(k_dim), as in the reference. Everything is computed in float32.

    Args:
        q: Queries.
        k: Keys.
        v: Values.
        g: Log decay of each head's memory at each token; at most zero.
        beta: Write strength of each token, between zero and one.
        state: Each head's memory before the first token, or `None` for zeros.

    Returns:
        The output for each token, in the queries' dtype, and the final memory in float32.
    """
    dtype = q.dtype
    q, k, v, g, beta = (x.astype(jnp.float32) for x in (q, k, v, g, beta))
    q = _l2norm(q) * q.shape[-1] ** -0.5
    k = _l2norm(k)
    if state is None:
        state = jnp.zeros((q.shape[1], q.shape[2], v.shape[2]), dtype=jnp.float32)

    def step(s, inputs):
        q_t, k_t, v_t, g_t, beta_t = inputs
        s = s * jnp.exp(g_t)[:, None, None]
        predicted = jnp.einsum("hkv,hk->hv", s, k_t)
        delta = (v_t - predicted) * beta_t[:, None]
        s = s + k_t[:, :, None] * delta[:, None, :]
        return s, jnp.einsum("hkv,hk->hv", s, q_t)

    state, out = jax.lax.scan(step, state.astype(jnp.float32), (q, k, v, g, beta))
    return out.astype(dtype), state


class DeltaNetState(eqx.Module):
    """What a DeltaNet layer carries between calls when generating.

    Attributes:
        conv: The last `kernel - 1` inputs to the causal convolution.
        recurrent: Each head's memory matrix, in float32.
    """

    conv: Float[Array, "taps channels"]
    recurrent: Float[Array, "heads k_dim v_dim"]


class RMSNormGated(eqx.Module):
    """RMSNorm followed by a SiLU gate.

    As in the reference: normalise in float32, cast back, scale, then multiply by SiLU(gate)
    computed in float32.

    Attributes:
        weight: Per-feature scale.
        eps: Constant added to the mean square.
    """

    weight: Float[Array, " dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6, dtype: DTypeLike = jnp.float32):
        """Create a norm with unit scale.

        Args:
            dim: Size of the normalised axis.
            eps: Constant added to the mean square.
            dtype: Parameter dtype.
        """
        self.weight = jnp.ones(dim, dtype=dtype)
        self.eps = eps

    def __call__(
        self, x: Float[Array, "*batch dim"], gate: Float[Array, "*batch dim"]
    ) -> Float[Array, "*batch dim"]:
        """Normalise `x`, scale it, and gate it.

        Args:
            x: Input.
            gate: Gate, passed through SiLU.

        Returns:
            The gated, normalised input, in the input dtype.
        """
        dtype = x.dtype
        x32 = x.astype(jnp.float32)
        x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        out = self.weight * x32.astype(dtype)
        return (out * jax.nn.silu(gate.astype(jnp.float32))).astype(dtype)


class DepthwiseConv1d(eqx.Module):
    """Causal depthwise convolution: one filter per channel, no bias.

    Attributes:
        weight: Filters, shaped (channels, 1, kernel) as in PyTorch's `Conv1d`.
    """

    weight: Float[Array, "channels 1 kernel"]

    def __init__(self, channels: int, kernel: int, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create randomly initialised filters.

        Args:
            channels: Number of channels.
            kernel: Filter length.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        self.weight = jax.random.normal(key, (channels, 1, kernel), dtype=dtype) * kernel**-0.5

    def __call__(
        self, x: Float[Array, "seq channels"], history: Float[Array, "taps channels"]
    ) -> tuple[Float[Array, "seq channels"], Float[Array, "taps channels"]]:
        """Convolve causally, continuing from the previous inputs.

        Args:
            x: Inputs.
            history: The previous `kernel - 1` inputs (zeros at the start of a sequence).

        Returns:
            The convolution of each position with itself and the preceding inputs, and the
            last `kernel - 1` inputs, to continue from next time.
        """
        kernel = self.weight.shape[-1]
        padded = jnp.concatenate([history.astype(x.dtype), x])
        w = self.weight[:, 0, :].astype(x.dtype)  # (channels, kernel)
        out = sum(w[:, j] * padded[j : j + x.shape[0]] for j in range(kernel))
        return out, padded[-(kernel - 1) :]


class GatedDeltaNet(eqx.Module):
    """Gated DeltaNet linear attention, as in Qwen3.5.

    Projections feed a causal convolution; queries, keys and values then go through the
    gated delta rule, whose output is normalised, gated and projected back.

    Attributes:
        in_proj_qkv: Fused query, key and value projection.
        in_proj_z: Output-gate projection.
        in_proj_b: Write-strength (beta) projection, one value per head.
        in_proj_a: Decay projection, one value per head.
        conv1d: Causal depthwise convolution over queries, keys and values.
        dt_bias: Bias added before the decay's softplus.
        A_log: Log of each head's decay rate.
        norm: Gated RMSNorm applied to each head's output.
        out_proj: Output projection.
        num_k_heads: Number of query/key heads.
        num_v_heads: Number of value heads.
        k_dim: Query/key head dimension.
        v_dim: Value head dimension.
    """

    in_proj_qkv: eqx.nn.Linear
    in_proj_z: eqx.nn.Linear
    in_proj_b: eqx.nn.Linear
    in_proj_a: eqx.nn.Linear
    conv1d: DepthwiseConv1d
    dt_bias: Float[Array, " v_heads"]
    A_log: Float[Array, " v_heads"]
    norm: RMSNormGated
    out_proj: eqx.nn.Linear
    num_k_heads: int = eqx.field(static=True)
    num_v_heads: int = eqx.field(static=True)
    k_dim: int = eqx.field(static=True)
    v_dim: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        num_k_heads: int,
        num_v_heads: int,
        k_dim: int,
        v_dim: int,
        conv_kernel: int,
        eps: float,
        key: PRNGKeyArray,
        dtype: DTypeLike = jnp.float32,
    ):
        """Create a randomly initialised layer.

        Args:
            dim: Model dimension.
            num_k_heads: Number of query/key heads.
            num_v_heads: Number of value heads (a multiple of `num_k_heads`).
            k_dim: Query/key head dimension.
            v_dim: Value head dimension.
            conv_kernel: Length of the causal convolution.
            eps: Epsilon of the gated norm.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        keys = jax.random.split(key, 6)
        channels = 2 * num_k_heads * k_dim + num_v_heads * v_dim
        self.in_proj_qkv = linear(dim, channels, key=keys[0], dtype=dtype)
        self.in_proj_z = linear(dim, num_v_heads * v_dim, key=keys[1], dtype=dtype)
        self.in_proj_b = linear(dim, num_v_heads, key=keys[2], dtype=dtype)
        self.in_proj_a = linear(dim, num_v_heads, key=keys[3], dtype=dtype)
        self.conv1d = DepthwiseConv1d(channels, conv_kernel, key=keys[4], dtype=dtype)
        self.dt_bias = jnp.ones(num_v_heads, dtype=dtype)
        self.A_log = jnp.zeros(num_v_heads, dtype=dtype)
        self.norm = RMSNormGated(v_dim, eps=eps, dtype=dtype)
        self.out_proj = linear(num_v_heads * v_dim, dim, key=keys[5], dtype=dtype)
        self.num_k_heads, self.num_v_heads = num_k_heads, num_v_heads
        self.k_dim, self.v_dim = k_dim, v_dim

    def init_state(self) -> DeltaNetState:
        """The state before the first token: zeros."""
        channels, _, kernel = self.conv1d.weight.shape
        return DeltaNetState(
            conv=jnp.zeros((kernel - 1, channels), dtype=self.conv1d.weight.dtype),
            recurrent=jnp.zeros((self.num_v_heads, self.k_dim, self.v_dim), dtype=jnp.float32),
        )

    def __call__(
        self,
        x: Float[Array, "seq dim"],
        state: DeltaNetState | None = None,
        mask: Bool[Array, " seq"] | None = None,
    ) -> tuple[Float[Array, "seq dim"], DeltaNetState]:
        """Apply the layer to a sequence, continuing from `state`.

        Args:
            x: Hidden state of each token.
            state: State from earlier tokens, or `None` to start afresh.
            mask: `True` for real tokens; padding tokens are zeroed before the layer, as in
                the reference. Defaults to all real.

        Returns:
            The output for each token, and the state after the last token.
        """
        seq = x.shape[0]
        state = self.init_state() if state is None else state
        if mask is not None:
            x = x * mask[:, None].astype(x.dtype)

        qkv = jax.vmap(self.in_proj_qkv)(x)
        z = jax.vmap(self.in_proj_z)(x).reshape(seq, self.num_v_heads, self.v_dim)
        b = jax.vmap(self.in_proj_b)(x)
        a = jax.vmap(self.in_proj_a)(x)

        qkv, conv_state = self.conv1d(qkv, state.conv)
        qkv = jax.nn.silu(qkv)
        key_size = self.num_k_heads * self.k_dim
        q = qkv[:, :key_size].reshape(seq, self.num_k_heads, self.k_dim)
        k = qkv[:, key_size : 2 * key_size].reshape(seq, self.num_k_heads, self.k_dim)
        v = qkv[:, 2 * key_size :].reshape(seq, self.num_v_heads, self.v_dim)

        beta = jax.nn.sigmoid(b)
        g = -jnp.exp(self.A_log.astype(jnp.float32)) * jax.nn.softplus(
            a.astype(jnp.float32) + self.dt_bias
        )
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            q, k = jnp.repeat(q, repeat, axis=1), jnp.repeat(k, repeat, axis=1)

        out, recurrent = gated_delta_rule(q, k, v, g, beta, state.recurrent)
        out = self.norm(out, z).reshape(seq, self.num_v_heads * self.v_dim)
        return jax.vmap(self.out_proj)(out), DeltaNetState(conv=conv_state, recurrent=recurrent)
