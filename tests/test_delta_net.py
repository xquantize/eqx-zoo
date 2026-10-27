"""The gated delta rule against Hugging Face's recurrent and chunked reference forms."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from eqx_zoo.layers import gated_delta_rule

reference = pytest.importorskip(
    "transformers.models.qwen3_5.modeling_qwen3_5", reason="needs transformers with Qwen3.5"
)
SEQ, HEADS, K_DIM, V_DIM = 100, 4, 16, 8  # 100 tokens: the chunked form pads to 128


def random_inputs(seed: int, with_state: bool):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(SEQ, HEADS, K_DIM)).astype(np.float32)
    k = rng.normal(size=(SEQ, HEADS, K_DIM)).astype(np.float32)
    v = rng.normal(size=(SEQ, HEADS, V_DIM)).astype(np.float32)
    g = -np.logaddexp(0, rng.normal(size=(SEQ, HEADS))).astype(np.float32)  # -softplus: <= 0
    beta = (1 / (1 + np.exp(-rng.normal(size=(SEQ, HEADS))))).astype(np.float32)
    state = rng.normal(size=(HEADS, K_DIM, V_DIM)).astype(np.float32) if with_state else None
    return q, k, v, g, beta, state


@pytest.mark.parametrize(
    "form", ["torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"]
)
@pytest.mark.parametrize("with_state", [False, True])
def test_matches_reference(form, with_state):
    q, k, v, g, beta, state = random_inputs(0, with_state)
    t = lambda a: None if a is None else torch.from_numpy(a)[None]  # noqa: E731 (add a batch axis)
    expected_out, expected_state = getattr(reference, form)(
        t(q),
        t(k),
        t(v),
        g=t(g),
        beta=t(beta),
        initial_state=t(state),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out, final = gated_delta_rule(
        *map(jnp.asarray, (q, k, v, g, beta)), None if state is None else jnp.asarray(state)
    )
    np.testing.assert_allclose(out, expected_out[0].numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(final, expected_state[0].numpy(), rtol=1e-4, atol=1e-5)


def test_continuing_from_state_equals_one_pass():
    """Splitting a sequence and passing the state across gives the same result as one pass."""
    q, k, v, g, beta, _ = (
        jnp.asarray(a) if a is not None else None for a in random_inputs(1, False)
    )
    out, final = gated_delta_rule(q, k, v, g, beta)
    first, mid = gated_delta_rule(q[:37], k[:37], v[:37], g[:37], beta[:37])
    second, end = gated_delta_rule(q[37:], k[37:], v[37:], g[37:], beta[37:], mid)
    np.testing.assert_allclose(jnp.concatenate([first, second]), out, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(end, final, rtol=1e-5, atol=1e-6)
