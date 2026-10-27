"""Qwen3.5's zero-centred RMSNorm and gated, partially rotary attention, against the reference."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from eqx_zoo._loading import load_safetensors
from eqx_zoo.layers import Attention, RMSNorm, causal_mask

reference = pytest.importorskip(
    "transformers.models.qwen3_5.modeling_qwen3_5", reason="needs transformers with Qwen3.5"
)
config_module = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")


def randomise(module):
    with torch.no_grad():
        for _, param in module.named_parameters():
            param.copy_(0.1 * torch.randn_like(param))


def load_into(skeleton, module, tmp_path):
    save_file(
        {n: p.detach().contiguous() for n, p in module.state_dict().items()},
        tmp_path / "w.safetensors",
    )
    return load_safetensors(skeleton, [tmp_path / "w.safetensors"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_centered_rmsnorm(dtype, tmp_path):
    torch.manual_seed(0)
    hf = reference.Qwen3_5RMSNorm(64, eps=1e-6)
    randomise(hf)
    hf = hf.to(dtype)
    jdtype = jnp.float32 if dtype == torch.float32 else jnp.bfloat16
    ours = load_into(
        eqx.filter_eval_shape(RMSNorm, 64, eps=1e-6, dtype=jdtype, zero_centered=True), hf, tmp_path
    )
    x = torch.randn(5, 64).to(dtype)
    with torch.no_grad():
        expected = hf(x).float().numpy()
    out = np.asarray(ours(jnp.asarray(x.float().numpy()).astype(jdtype)).astype(jnp.float32))
    np.testing.assert_array_equal(
        out, expected
    ) if dtype == torch.bfloat16 else np.testing.assert_allclose(
        out, expected, rtol=1e-6, atol=1e-6
    )


def test_gated_attention_matches_reference(tmp_path):
    config = config_module.Qwen3_5TextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
        },
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = reference.Qwen3_5Attention(config, layer_idx=0).eval()
    randomise(hf)
    rotary = reference.Qwen3_5TextRotaryEmbedding(config)

    seq = 12
    x = torch.randn(1, seq, 64)
    cos, sin = rotary(x, torch.arange(seq)[None])
    additive = torch.full((seq, seq), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        expected, _ = hf(hidden_states=x, position_embeddings=(cos, sin), attention_mask=additive)

    skeleton = eqx.filter_eval_shape(
        Attention,
        64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        rope_theta=10000.0,
        qk_norm=True,
        rotary_dim=8,
        output_gate=True,
        zero_centered_norm=True,
        key=jax.random.key(0),
    )
    ours = load_into(skeleton, hf, tmp_path)
    positions = jnp.arange(seq)
    out, _ = ours(jnp.asarray(x[0].numpy()), positions, causal_mask(positions, positions))
    np.testing.assert_allclose(out, expected[0].numpy(), rtol=1e-4, atol=1e-5)
