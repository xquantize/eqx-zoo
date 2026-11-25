"""Folding an adapter into the base weights."""

import jax

from eqx_zoo.lora.adapt import _ADAPTED, adapter_layers


def merge(model):
    """Fold the adapter into the base layers, returning a model without LoRA layers.

    Each wrapped layer is replaced by its `merged()` base layer: `W + scaling * B @ A`
    computed in at least float32 and cast to the base dtype (the bias likewise, with
    `lora_bias`). The result is an ordinary model of the same class, with the structure and
    parameter names of the base model, so `save_pretrained` works as usual.

    Args:
        model: A model with an adapter.

    Returns:
        The model with the adapter merged into its weights.

    Raises:
        ValueError: If the model has no adapter, or an adapter has a `lora_B` bias that a
            base layer without a bias can't absorb.
    """
    if not adapter_layers(model):
        raise ValueError("the model has no adapter to merge")
    return jax.tree.map(
        lambda node: node.merged() if isinstance(node, _ADAPTED) else node,
        model,
        is_leaf=lambda node: isinstance(node, _ADAPTED),
    )
