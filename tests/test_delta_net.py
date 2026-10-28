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


def build_layers(tmp_path):
    """A small, fully randomised Hugging Face DeltaNet layer, and ours loaded from its weights."""
    import equinox as eqx
    import jax
    from safetensors.torch import save_file

    from eqx_zoo._loading import load_safetensors
    from eqx_zoo.layers import GatedDeltaNet

    config_module = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
    config = config_module.Qwen3_5TextConfig(
        hidden_size=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        num_hidden_layers=1,
        layer_types=["linear_attention"],
    )
    torch.manual_seed(0)
    hf = reference.Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    with torch.no_grad():
        for name, param in hf.named_parameters():
            if name == "norm.weight":
                param.copy_(1 + 0.1 * torch.randn_like(param))
            elif name in ("A_log", "dt_bias"):
                param.copy_(torch.randn_like(param))
            else:
                param.copy_(0.1 * torch.randn_like(param))
    save_file(
        {n: p.detach().contiguous() for n, p in hf.state_dict().items()},
        tmp_path / "layer.safetensors",
    )

    skeleton = eqx.filter_eval_shape(
        GatedDeltaNet,
        64,
        num_k_heads=4,
        num_v_heads=8,
        k_dim=16,
        v_dim=16,
        conv_kernel=4,
        eps=1e-6,
        key=jax.random.key(0),
    )
    return hf, load_safetensors(skeleton, [tmp_path / "layer.safetensors"])


def test_layer_matches_reference(tmp_path):
    hf, ours = build_layers(tmp_path)
    x = np.random.default_rng(2).normal(size=(1, 50, 64)).astype(np.float32)
    with torch.no_grad():
        expected = hf(torch.from_numpy(x)).numpy()[0]
    out, _ = ours(jnp.asarray(x[0]))
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_layer_step_by_step_equals_full_sequence(tmp_path):
    """Feeding tokens one at a time, carrying the state, reproduces the full-sequence output."""
    _, ours = build_layers(tmp_path)
    x = jnp.asarray(np.random.default_rng(3).normal(size=(20, 64)).astype(np.float32))
    full, _ = ours(x)
    state, steps = None, []
    for t in range(x.shape[0]):
        out, state = ours(x[t : t + 1], state)
        steps.append(out)
    np.testing.assert_allclose(jnp.concatenate(steps), full, rtol=1e-5, atol=1e-6)
