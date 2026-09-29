"""Shared fixtures: reference activations and Hugging Face weights."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from huggingface_hub import snapshot_download
from safetensors import safe_open

REPO_ID = "Qwen/Qwen3-0.6B"
REFERENCE = Path(__file__).parents[1] / "reference" / "qwen3-0.6b.npz"


@pytest.fixture(scope="session")
def reference() -> dict[str, np.ndarray]:
    """Reference activations from HF transformers, with the batch axis removed."""
    if not REFERENCE.exists():
        pytest.skip("reference missing: run `uv run python scripts/make_reference.py`")
    with np.load(REFERENCE) as ref:
        return {name: ref[name][0] for name in ref.files}


@pytest.fixture(scope="session")
def hf_weight():
    """Return a loader mapping an HF parameter name to a float32 JAX array."""
    path = Path(snapshot_download(REPO_ID, allow_patterns=["*.safetensors"]))
    with safe_open(path / "model.safetensors", framework="pt") as f:
        yield lambda name: jnp.asarray(f.get_tensor(name).float().numpy())
