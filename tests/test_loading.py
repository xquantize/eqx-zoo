"""Tests for loading checkpoints."""

import equinox as eqx
import jax
import numpy as np
from huggingface_hub import snapshot_download
from support.registry import CHECKPOINTS

from eqx_zoo import CausalLM


def test_loads_from_local_directory():
    repo_id = CHECKPOINTS["smollm2-135m"]
    directory = snapshot_download(repo_id, allow_patterns=["config.json", "*.safetensors"])

    from_dir = CausalLM.from_pretrained(directory)
    from_hub = CausalLM.from_pretrained(repo_id)

    assert from_dir.config == from_hub.config
    leaves = zip(
        jax.tree.leaves(eqx.filter(from_dir, eqx.is_array)),
        jax.tree.leaves(eqx.filter(from_hub, eqx.is_array)),
        strict=True,
    )
    for a, b in leaves:
        np.testing.assert_array_equal(a, b)
