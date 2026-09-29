"""Load safetensors checkpoints into Equinox modules by parameter name."""

from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors import safe_open


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


def load_safetensors[T](
    skeleton: T, files: Iterable[str | Path], *, ignore: Iterable[str] = ()
) -> T:
    """Fill a model skeleton with tensors from safetensors files.

    Each leaf of `skeleton` is looked up by its attribute path, e.g. the leaf at
    `model.layers[0].self_attn.q_proj.weight` is loaded from the checkpoint tensor
    named `"model.layers.0.self_attn.q_proj.weight"`, and cast to the leaf's dtype.

    Args:
        skeleton: Module whose array leaves describe the expected shapes and dtypes,
            typically created with `equinox.filter_eval_shape`.
        files: Checkpoint files; parameters may be sharded across several.
        ignore: Checkpoint parameters that are deliberately unused.

    Returns:
        `skeleton` with every array leaf replaced by the corresponding tensor.

    Raises:
        KeyError: If the checkpoint is missing a parameter.
        ValueError: If a shape mismatches, or the checkpoint has unexpected parameters.
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(skeleton)
    names = [_path_to_name(path) for path, _ in leaves]

    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(f, framework="flax")) for f in files]
        index = {name: handle for handle in handles for name in handle.keys()}

        loaded = []
        for name, (_, leaf) in zip(names, leaves, strict=True):
            if name not in index:
                raise KeyError(f"checkpoint is missing parameter {name!r}")
            tensor = index[name].get_tensor(name)
            if tuple(tensor.shape) != tuple(leaf.shape):
                raise ValueError(f"{name}: checkpoint shape {tensor.shape} != model {leaf.shape}")
            loaded.append(jnp.asarray(tensor, dtype=leaf.dtype))

    unexpected = sorted(set(index) - set(names) - set(ignore))
    if unexpected:
        raise ValueError(
            f"checkpoint has {len(unexpected)} unexpected parameters: {unexpected[:5]}"
        )
    return jax.tree_util.tree_unflatten(treedef, loaded)
