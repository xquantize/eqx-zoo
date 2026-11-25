"""Helpers shared by the LoRA tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
from support.models import model_class
from support.tiny import PROMPT_IDS

from eqx_zoo.lora import add_adapter
from eqx_zoo.lora.adapt import adapter_layers

MODELS = ["tiny-llama", "tiny-qwen2", "tiny-gpt2", "tiny-gpt-neox"]
IDS = jnp.array(PROMPT_IDS)


def load(checkpoint, name, **kwargs):
    directory = checkpoint(name)
    return model_class(directory).from_pretrained(directory, **kwargs)


def leaves(tree) -> dict[str, object]:
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {jax.tree_util.keystr(path): leaf for path, leaf in flat}


def adapted(checkpoint, name, **kwargs):
    model = load(checkpoint, name)
    options = {"r": 4, "alpha": 8, "target_modules": "all-linear", "key": jax.random.key(0)}
    return model, add_adapter(model, **(options | kwargs))


def randomise(model, seed: int = 0, scale: float = 0.1):
    """The model with every adapter weight (A, B and any bias) replaced by random values."""
    layers = adapter_layers(model)
    keys = iter(jax.random.split(jax.random.key(seed), 3 * len(layers)))

    def fresh(array):
        return None if array is None else scale * jax.random.normal(next(keys), array.shape)

    def pick(m):
        return [
            leaf
            for layer in adapter_layers(m).values()
            for leaf in (layer.lora_A, layer.lora_B, layer.lora_B_bias)
            if leaf is not None
        ]

    new = [
        fresh(leaf)
        for layer in layers.values()
        for leaf in (layer.lora_A, layer.lora_B, layer.lora_B_bias)
        if leaf is not None
    ]
    return eqx.tree_at(pick, model, new)
