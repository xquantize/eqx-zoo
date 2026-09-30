"""Tests for loading checkpoints."""

from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest
from huggingface_hub import snapshot_download
from safetensors.numpy import save_file
from support.registry import CHECKPOINTS

from eqx_zoo import CausalLM
from eqx_zoo._loading import load_safetensors


@pytest.mark.checkpoint
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


class Toy(eqx.Module):
    """A stack of two linear layers plus a head, like a miniature decoder."""

    layers: eqx.nn.Linear
    head: eqx.nn.Linear

    def __init__(self, key):
        layer_key, head_key = jax.random.split(key)
        make = lambda k: eqx.nn.Linear(3, 2, use_bias=False, key=k)  # noqa: E731
        self.layers = eqx.filter_vmap(make)(jax.random.split(layer_key, 2))
        self.head = eqx.nn.Linear(2, 4, use_bias=False, key=head_key)


def toy_checkpoint(tmp_path, **overrides) -> Path:
    tensors = {
        "layers.0.weight": np.full((2, 3), 0.0, np.float32),
        "layers.1.weight": np.full((2, 3), 1.0, np.float32),
        "head.weight": np.full((4, 2), 2.0, np.float32),
    }
    tensors.update(overrides)
    tensors = {k: v for k, v in tensors.items() if v is not None}
    path = tmp_path / "model.safetensors"
    save_file(tensors, path)
    return path


def test_loads_stacked_layers(tmp_path):
    loaded = load_safetensors(
        Toy(jax.random.key(0)), [toy_checkpoint(tmp_path)], stacked=["layers"]
    )
    assert loaded.layers.weight.shape == (2, 2, 3)
    np.testing.assert_array_equal(loaded.layers.weight[0], 0.0)
    np.testing.assert_array_equal(loaded.layers.weight[1], 1.0)
    np.testing.assert_array_equal(loaded.head.weight, 2.0)


def test_stacked_missing_layer(tmp_path):
    path = toy_checkpoint(tmp_path, **{"layers.1.weight": None})
    with pytest.raises(KeyError, match="layers.1.weight"):
        load_safetensors(Toy(jax.random.key(0)), [path], stacked=["layers"])


def test_stacked_shape_mismatch(tmp_path):
    path = toy_checkpoint(tmp_path, **{"layers.0.weight": np.zeros((2, 5), np.float32)})
    with pytest.raises(ValueError, match="layers.0.weight"):
        load_safetensors(Toy(jax.random.key(0)), [path], stacked=["layers"])


def test_stacked_extra_layer(tmp_path):
    path = toy_checkpoint(tmp_path, **{"layers.2.weight": np.zeros((2, 3), np.float32)})
    with pytest.raises(ValueError, match="unexpected"):
        load_safetensors(Toy(jax.random.key(0)), [path], stacked=["layers"])
