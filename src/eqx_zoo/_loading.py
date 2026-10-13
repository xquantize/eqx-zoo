"""Load safetensors checkpoints into Equinox modules by parameter name."""

import json
from collections.abc import Collection, Iterable
from contextlib import ExitStack
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from safetensors import safe_open


def config_with_dtype(text: str, dtype) -> str:
    """A `config.json` text with its dtype set to `dtype`, where transformers 4 and 5 read it."""
    config = json.loads(text)
    # transformers 4 reads `torch_dtype`, transformers 5 reads `dtype`.
    for key in [k for k in ("torch_dtype", "dtype") if k in config] or ["torch_dtype"]:
        config[key] = jnp.dtype(dtype).name
    return json.dumps(config, indent=2) + "\n"


def _path_to_name(path: tuple) -> str:
    parts = []
    for key in path:
        if isinstance(key, jax.tree_util.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jax.tree_util.SequenceKey):
            parts.append(str(key.idx))
        else:
            raise TypeError(f"unsupported pytree key {key!r}")
    return ".".join(parts)


def _layer_names(name: str, num_layers: int, stacked: Collection[str]) -> list[str] | None:
    """Checkpoint names for each layer of a stacked leaf, or `None` if it isn't stacked.

    For example, `"model.layers.mlp.up_proj.weight"` under the stacked prefix
    `"model.layers"` maps to `"model.layers.0.mlp.up_proj.weight"`, `"model.layers.1..."`.
    """
    for prefix in stacked:
        if name.startswith(prefix + "."):
            suffix = name[len(prefix) + 1 :]
            return [f"{prefix}.{i}.{suffix}" for i in range(num_layers)]
    return None


def load_safetensors[T](
    skeleton: T,
    files: Iterable[str | Path],
    *,
    ignore: Iterable[str] = (),
    stacked: Collection[str] = (),
) -> T:
    """Fill a model skeleton with tensors from safetensors files.

    Each leaf of `skeleton` is looked up by its attribute path, e.g. the leaf at
    `model.norm.weight` is loaded from the checkpoint tensor `"model.norm.weight"`, and
    cast to the leaf's dtype. Leaves under a `stacked` path carry a leading layer axis
    and are assembled from one checkpoint tensor per layer: the leaf at
    `model.layers.mlp.up_proj.weight` is stacked from `"model.layers.0.mlp.up_proj.weight"`,
    `"model.layers.1.mlp.up_proj.weight"`, and so on.

    Args:
        skeleton: Module whose array leaves describe the expected shapes and dtypes,
            typically created with `equinox.filter_eval_shape`.
        files: Checkpoint files; parameters may be sharded across several.
        ignore: Checkpoint parameters that are deliberately unused.
        stacked: Attribute paths, e.g. `"model.layers"`, of modules stored with a
            leading layer axis.

    Returns:
        `skeleton` with every array leaf replaced by the corresponding tensor(s).

    Raises:
        KeyError: If the checkpoint is missing a parameter.
        ValueError: If a shape mismatches, or the checkpoint has unexpected parameters.
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(skeleton)

    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(f, framework="flax")) for f in files]
        index = {name: handle for handle in handles for name in handle.keys()}
        used: set[str] = set()

        def load(name: str, shape: tuple[int, ...], dtype) -> jax.Array:
            if name not in index:
                raise KeyError(f"checkpoint is missing parameter {name!r}")
            tensor = index[name].get_tensor(name)
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"{name}: checkpoint shape {tensor.shape} != model {shape}")
            used.add(name)
            return jnp.asarray(tensor, dtype=dtype)

        loaded = []
        for path, leaf in leaves:
            name = _path_to_name(path)
            layer_names = _layer_names(name, leaf.shape[0], stacked) if leaf.ndim else None
            if layer_names is None:
                loaded.append(load(name, leaf.shape, leaf.dtype))
            else:
                loaded.append(jnp.stack([load(n, leaf.shape[1:], leaf.dtype) for n in layer_names]))

    unexpected = sorted(set(index) - used - set(ignore))
    if unexpected:
        raise ValueError(
            f"checkpoint has {len(unexpected)} unexpected parameters: {unexpected[:5]}"
        )
    return jax.tree_util.tree_unflatten(treedef, loaded)


def save_safetensors(model, file: str | Path, *, stacked: Collection[str] = ()) -> None:
    """Write a model's arrays to a safetensors file, named as `load_safetensors` reads them.

    The inverse of `load_safetensors`: each array leaf is saved under its attribute path,
    e.g. `model.norm.weight`, and leaves under a `stacked` path are split along their leading
    axis into one tensor per layer, e.g. `model.layers.0.mlp.up_proj.weight`, ...

    Args:
        model: Module to save.
        file: Destination file.
        stacked: Attribute paths of modules stored with a leading layer axis, as for
            `load_safetensors`.
    """
    from safetensors.flax import save_file

    tensors = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(eqx.filter(model, eqx.is_array))[0]:
        name = _path_to_name(path)
        layer_names = _layer_names(name, leaf.shape[0], stacked) if leaf.ndim else None
        if layer_names is None:
            tensors[name] = leaf
        else:
            tensors.update(zip(layer_names, leaf, strict=True))
    save_file(tensors, str(file), metadata={"format": "pt"})
