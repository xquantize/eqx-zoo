"""Encoders in float64 against Hugging Face in float64.

Far stricter than the float32 tests: in float64, Hugging Face's encoders compute everything in
float64, so ours must match to float64 rounding. Measured: 2-4e-16 relative; a single fixed
float32 cast in a norm or the softmax raises that to about 2e-7. Decoders aren't tested this
way: Hugging Face's decoders cast to float32 explicitly (norms, RoPE, softmax), so their
float64 output is itself only float32-accurate.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from support.registry import TINY_ENCODERS
from support.tiny import build_encoder
from transformers import AutoModel

from eqx_zoo import Encoder

LIMIT = 1e-14  # relative to the largest value; see the module docstring


@pytest.mark.parametrize("name", list(TINY_ENCODERS))
def test_float64_matches_transformers(name, tmp_path):
    build_encoder(name, tmp_path)
    ids = np.array([5, 17, 42, 8, 99, 3, 61, 12])
    hf = AutoModel.from_pretrained(
        tmp_path, dtype=torch.float64, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        expected = hf(torch.from_numpy(ids)[None]).last_hidden_state[0].numpy()

    with jax.enable_x64(True):
        hidden = np.asarray(Encoder.from_pretrained(tmp_path, dtype=jnp.float64)(jnp.asarray(ids)))
    assert hidden.dtype == np.float64
    error = np.abs(hidden - expected).max() / np.abs(expected).max()
    assert error <= LIMIT, f"{name}: float64 relative error {error:.1e}; at most {LIMIT:.0e}"
