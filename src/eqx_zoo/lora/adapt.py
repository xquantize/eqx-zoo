"""Adding LoRA adapters to a model, and selecting the parameters to train."""

import dataclasses
import functools
import math
import re
import warnings
from collections.abc import Iterator, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from eqx_zoo.layers import SparseMoE
from eqx_zoo.lora.layers import LoRAConv1D, LoRALinear
from eqx_zoo.models.causal_lm import CausalLM
from eqx_zoo.models.gpt2 import GPT2
from eqx_zoo.models.gpt2.model import Conv1D
from eqx_zoo.models.gpt_neox import GPTNeoX

ALL_LINEAR = "all-linear"

# Hugging Face prefixes module names that eqx-zoo's models don't have: GPT-2 keeps its
# blocks under `transformer.`. PEFT matches `target_modules` against Hugging Face's names.
_HF_PREFIXES: dict[type, str] = {GPT2: "transformer."}
# The output head, which PEFT's "all-linear" leaves out (Hugging Face's `get_output_embeddings`).
_OUTPUT_HEADS: dict[type, str | None] = {CausalLM: "lm_head", GPTNeoX: "embed_out", GPT2: None}

_ADAPTED = (LoRALinear, LoRAConv1D)


def _children(node) -> Iterator[tuple[str | int, object]]:
    if isinstance(node, eqx.Module):
        for field in dataclasses.fields(node):
            if not field.metadata.get("static", False):
                yield field.name, getattr(node, field.name)
    elif isinstance(node, list | tuple):
        yield from enumerate(node)
    elif isinstance(node, dict):
        yield from node.items()


def _named_modules(node, path: tuple = ()) -> Iterator[tuple[tuple, eqx.Module]]:
    """Every Equinox module in a tree, with its path: attribute names and list indices."""
    if isinstance(node, eqx.Module):
        yield path, node
    for key, child in _children(node):
        yield from _named_modules(child, (*path, key))


def _get(model, path: tuple):
    return functools.reduce(
        lambda node, key: node[key] if isinstance(key, int) else getattr(node, key), path, model
    )


def _name(model, path: tuple) -> str:
    """The Hugging Face name of the module at `path`."""
    return _HF_PREFIXES.get(type(model), "") + ".".join(map(str, path))


def adapter_layers(model) -> dict[str, LoRALinear | LoRAConv1D]:
    """The model's LoRA layers, by Hugging Face module name.

    Args:
        model: A model, with or without an adapter.

    Returns:
        The names (as PEFT reports them, e.g. `model.layers.0.self_attn.q_proj`, or
        `transformer.h.0.attn.c_attn` for GPT-2) mapped to their layers.
    """
    return {
        _name(model, path): module
        for path, module in _named_modules(model)
        if isinstance(module, _ADAPTED)
    }


def _in_features(layer: eqx.nn.Linear | Conv1D) -> int:
    return layer.weight.shape[0] if isinstance(layer, Conv1D) else layer.in_features


def _out_features(layer: eqx.nn.Linear | Conv1D) -> int:
    return layer.weight.shape[1] if isinstance(layer, Conv1D) else layer.out_features


def _matches(target_modules: str | Sequence[str], name: str) -> bool:
    """Whether PEFT's `target_modules` selects a module name.

    A string is a regex that must match the whole name; a name in a list matches exactly or as
    a suffix after a dot.
    """
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, name) is not None
    return any(name == target or name.endswith(f".{target}") for target in target_modules)


def _select(model, target_modules: str | Sequence[str]) -> list[tuple]:
    """Paths of the layers to wrap, in model order."""
    head = _OUTPUT_HEADS[type(model)]
    all_linear = isinstance(target_modules, str) and target_modules.lower() == ALL_LINEAR
    selected = []
    for path, module in _named_modules(model):
        name = _name(model, path)
        if not path or isinstance(module, _ADAPTED):
            continue
        adaptable = isinstance(module, eqx.nn.Linear | Conv1D)
        if all_linear:
            if adaptable and ".".join(map(str, path)) != head:
                selected.append(path)
        elif _matches(target_modules, name):
            if isinstance(module, eqx.nn.Embedding):
                raise NotImplementedError(
                    f"{name!r} is an embedding: PEFT can adapt embeddings, but only linear "
                    "layers are supported here"
                )
            if not adaptable:
                raise ValueError(
                    f"target module {name!r} is a {type(module).__name__}, but only linear "
                    "layers (eqx.nn.Linear, GPT-2's Conv1D) can be adapted"
                )
            selected.append(path)
    return selected


def _check_model(model) -> None:
    """Refuse models that can't take an adapter, or already have one.

    Raises:
        NotImplementedError: If the model is not a `CausalLM`, `GPT2` or `GPTNeoX`, or is a
            Qwen3.5 or mixture-of-experts model.
        ValueError: If the model already has an adapter.
    """
    if type(model) not in _OUTPUT_HEADS:
        raise NotImplementedError(
            f"adapters are supported for CausalLM, GPT2 and GPTNeoX, not {type(model).__name__}"
        )
    if isinstance(model, CausalLM) and model.config.architecture.startswith("Qwen3_5"):
        raise NotImplementedError(
            f"adapters for {model.config.architecture} are not supported yet: its checkpoints "
            "name the language model's modules with another prefix"
        )
    if any(isinstance(module, SparseMoE) for _, module in _named_modules(model)):
        raise NotImplementedError(
            "adapters for mixture-of-experts models are not supported yet: the experts are "
            "stacked arrays, and whether PEFT adapts the router is unchecked"
        )
    if adapter_layers(model):
        raise ValueError("the model already has an adapter; only one is supported")


def _linear_modules(model) -> dict[str, tuple]:
    """Paths of every linear layer, by Hugging Face module name."""
    return {
        _name(model, path): path
        for path, module in _named_modules(model)
        if path and isinstance(module, eqx.nn.Linear | Conv1D)
    }


def _wrap(base, lora_A, lora_B, lora_B_bias, *, alpha: float, rslora: bool):
    wrapper = LoRAConv1D if isinstance(base, Conv1D) else LoRALinear
    return wrapper(base, lora_A, lora_B, alpha=alpha, rslora=rslora, lora_B_bias=lora_B_bias)


def _insert(model, wrapped: dict[tuple, LoRALinear | LoRAConv1D]):
    """The model with its layers at the given paths replaced."""
    paths = list(wrapped)
    return eqx.tree_at(
        lambda m: tuple(_get(m, path) for path in paths), model, tuple(wrapped.values())
    )


def _warn_lora_bias_without_bias(model, wrapped: dict[tuple, LoRALinear | LoRAConv1D]) -> None:
    """Warn, as PEFT does, about `lora_bias` on layers without a bias: they can't be merged."""
    bare = [
        _name(model, path)
        for path, layer in wrapped.items()
        if layer.lora_B_bias is not None and layer.base.bias is None
    ]
    if bare:
        warnings.warn(
            f"lora_bias=True, but {len(bare)} targeted layers have no bias (for example "
            f"{bare[0]!r}), so this adapter can't be merged into them",
            UserWarning,
            stacklevel=3,
        )


def add_adapter(
    model,
    *,
    r: int,
    alpha: float,
    target_modules: str | Sequence[str],
    key: PRNGKeyArray,
    rslora: bool = False,
    lora_bias: bool = False,
    dropout: float = 0.0,
):
    """Add a LoRA adapter, initialised as PEFT does by default.

    Each targeted linear layer (`eqx.nn.Linear`, or GPT-2's `Conv1D`) is wrapped in a
    `LoRALinear` (or `LoRAConv1D`). `lora_A` is drawn uniformly from `±1/sqrt(in_features)`
    (PEFT's Kaiming-uniform bound) and `lora_B` is zero, both in float32, so the adapted model
    initially computes exactly what the base model does.

    Targets are matched against Hugging Face's module names, as PEFT does: for GPT-2 those
    start with `transformer.` (`transformer.h.0.attn.c_attn`), and for GPT-NeoX the output head
    is `embed_out`.

    Args:
        model: A `CausalLM`, `GPT2` or `GPTNeoX` without an adapter.
        r: Rank of the update.
        alpha: PEFT's `lora_alpha`; the update is scaled by `alpha / r`.
        target_modules: Which layers to adapt. A list of names, each matching a module's name
            exactly or as a suffix after a dot (`"q_proj"` matches `model.layers.0.self_attn.q_proj`
            but `"proj"` does not); a string, which is a regular expression that must match the
            whole name; or `"all-linear"` for every linear layer except the output head.
        key: PRNG key for initialisation, split per adapted layer.
        rslora: Scale by `alpha / sqrt(r)` instead (rank-stabilised LoRA).
        lora_bias: Give each `lora_B` a bias, initialised to zero. As in PEFT, this warns for
            targeted layers that have no bias themselves, since the adapter can't be merged
            into them.
        dropout: PEFT's `lora_dropout`. Only 0 is supported.

    Returns:
        The model with the adapter.

    Raises:
        NotImplementedError: If `dropout > 0` (training dropout needs a PRNG key in every
            forward pass), the model is not one of the supported classes, is a Qwen3.5 or
            mixture-of-experts model, or a target is an embedding.
        ValueError: If the model already has an adapter, `r` is not positive, `target_modules`
            matches no layer, or it matches a module that is not a linear layer.
    """
    if dropout > 0:
        raise NotImplementedError(
            f"dropout={dropout}: LoRA dropout in training would need a PRNG key in every "
            "forward pass, which is not supported; use dropout=0.0"
        )
    _check_model(model)

    if r <= 0:
        raise ValueError(f"r must be a positive integer, got {r}")
    paths = _select(model, target_modules)
    if not paths:
        existing = sorted(
            {
                str(path[-1])
                for path, m in _named_modules(model)
                if path and isinstance(m, eqx.nn.Linear | Conv1D)
            }
        )
        example = next(
            _name(model, p)
            for p, m in _named_modules(model)
            if isinstance(m, eqx.nn.Linear | Conv1D)
        )
        raise ValueError(
            f"target_modules={target_modules!r} matched no module. Linear layers exist under "
            f"the names {existing} (for example {example!r})"
        )

    wrapped = {}
    for path, layer_key in zip(paths, jax.random.split(key, len(paths)), strict=True):
        base = _get(model, path)
        in_features, out_features = _in_features(base), _out_features(base)
        bound = 1 / math.sqrt(in_features)
        lora_A = jax.random.uniform(
            layer_key, (r, in_features), dtype=jnp.float32, minval=-bound, maxval=bound
        )
        lora_B = jnp.zeros((out_features, r), dtype=jnp.float32)
        lora_B_bias = jnp.zeros(out_features, dtype=jnp.float32) if lora_bias else None
        wrapped[path] = _wrap(base, lora_A, lora_B, lora_B_bias, alpha=alpha, rslora=rslora)
    _warn_lora_bias_without_bias(model, wrapped)
    return _insert(model, wrapped)


def trainable(model):
    """A boolean mask selecting the adapter's parameters, for `equinox.partition`.

    Example:
        >>> trainable_part, frozen = eqx.partition(model, lora.trainable(model))

    Args:
        model: A model, with or without an adapter.

    Returns:
        A tree shaped like `model` that is `True` at every `lora_A`, `lora_B` and `lora_B_bias`
        array, and `False` everywhere else (all `False` if there is no adapter).
    """

    def mask(node):
        if not isinstance(node, _ADAPTED):
            return False
        marks = jax.tree.map(lambda _: False, node)
        marks = eqx.tree_at(lambda layer: (layer.lora_A, layer.lora_B), marks, (True, True))
        if node.lora_B_bias is not None:
            marks = eqx.tree_at(lambda layer: layer.lora_B_bias, marks, True)
        return marks

    return jax.tree.map(mask, model, is_leaf=lambda node: isinstance(node, _ADAPTED))
