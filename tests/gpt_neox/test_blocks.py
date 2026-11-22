"""GPT-NeoX's feed-forward block against Transformers', with enough spread to see its GELU."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP as HFMLP

from eqx_zoo._loading import load_safetensors
from eqx_zoo.models.gpt_neox.model import GPTNeoXMLP

DIM, INNER, TOKENS = 64, 256, 256
# Inputs of this scale give pre-activations with a standard deviation of about 2, so most of
# them fall in GELU's curved region. There, tanh GELU is off by 1.1e-3 in the block's output
# (up to 7.9 in size), while the correct code is within 3.3e-6 of Transformers (measured, CPU,
# float32). Every element's error is within a tenth of the limit for the correct code, and
# tanh GELU exceeds it up to 33 times over. The tiny models' small activations make the two
# forms differ by less than their layer tolerance.
INPUT_SCALE = 2.0
TOL = dict(rtol=1e-5, atol=2e-5)


def load_mlp(tmp_path):
    """Transformers' MLP with random weights, and ours loaded from them by name."""
    config = GPTNeoXConfig(
        hidden_size=DIM, intermediate_size=INNER, num_attention_heads=4, hidden_act="gelu"
    )
    torch.manual_seed(0)
    hf = HFMLP(config).eval()
    with torch.no_grad():
        for param in hf.parameters():
            # Unit-variance pre-activations for unit inputs; biases with a deviation of 0.5.
            param.copy_(
                torch.randn_like(param) / (param.shape[-1] ** 0.5 if param.ndim == 2 else 2)
            )
    save_file(
        {n: p.detach().contiguous() for n, p in hf.state_dict().items()},
        tmp_path / "w.safetensors",
    )
    skeleton = eqx.filter_eval_shape(
        GPTNeoXMLP, DIM, INNER, key=jax.random.key(0), dtype=jnp.float32
    )
    return hf, load_safetensors(skeleton, [tmp_path / "w.safetensors"])


def test_mlp_matches_transformers(tmp_path):
    hf, ours = load_mlp(tmp_path)
    x = INPUT_SCALE * np.random.default_rng(0).normal(size=(TOKENS, DIM)).astype(np.float32)
    with torch.no_grad():
        expected = hf(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(jax.vmap(ours)(jnp.asarray(x)), expected, **TOL)


def test_inputs_distinguish_exact_from_tanh_gelu(tmp_path):
    """The block test's inputs are discriminating: tanh GELU is far outside its tolerance."""
    hf, ours = load_mlp(tmp_path)
    x = INPUT_SCALE * np.random.default_rng(0).normal(size=(TOKENS, DIM)).astype(np.float32)
    with torch.no_grad():
        expected = hf(torch.from_numpy(x)).numpy()

    def tanh_gelu(t):
        return ours.dense_4h_to_h(jax.nn.gelu(ours.dense_h_to_4h(t), approximate=True))

    error = np.abs(np.asarray(jax.vmap(tanh_gelu)(jnp.asarray(x))) - expected).max()
    assert error > 10 * TOL["atol"], f"tanh GELU differs by only {error:.1e}"
