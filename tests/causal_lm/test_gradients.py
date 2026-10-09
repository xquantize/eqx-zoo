"""Gradient parity with PyTorch, on the tiny random models.

Forward tests can't see a bug that leaves values unchanged but breaks gradients, such as a
stray `jax.lax.stop_gradient`. Here the gradients of the mean next-token cross-entropy are
compared with PyTorch's for every parameter. Measured: correct code is within 1.4e-6
(relative error) on every parameter; a `stop_gradient` around RMSNorm's normalisation factor,
which passes every forward test, is off by 39-105%.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

LIMIT = 1e-4  # relative error per parameter; see the module docstring


def logits_of(model, ids):
    out = model(ids)
    return out[0] if isinstance(out, tuple) else out


def loss_fn(model, ids):
    """Mean next-token cross-entropy, as transformers computes it with labels=ids."""
    logp = jax.nn.log_softmax(logits_of(model, ids)[:-1].astype(jnp.float32), axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, ids[1:, None], axis=-1))


def to_pytorch_layout(ours: dict[str, np.ndarray], expected: dict) -> dict[str, np.ndarray]:
    """Rearrange stacked mixture-of-experts gradients into Hugging Face's parameter layout.

    transformers 5 fuses each layer's experts (`experts.gate_up_proj`, with gate and up
    stacked along the output axis, and `experts.down_proj`); transformers 4 keeps one module
    per expert (`experts.0.gate_proj.weight`, ...).
    """
    out = dict(ours)
    for name in list(ours):
        if not name.endswith(".experts.gate_proj.weight"):
            continue
        prefix = name.removesuffix("gate_proj.weight")
        gate, up, down = (
            out.pop(f"{prefix}{p}.weight") for p in ("gate_proj", "up_proj", "down_proj")
        )
        if f"{prefix}gate_up_proj" in expected:
            out[f"{prefix}gate_up_proj"] = np.concatenate([gate, up], axis=1)
            out[f"{prefix}down_proj"] = down
        else:
            for e in range(gate.shape[0]):
                out[f"{prefix}{e}.gate_proj.weight"] = gate[e]
                out[f"{prefix}{e}.up_proj.weight"] = up[e]
                out[f"{prefix}{e}.down_proj.weight"] = down[e]
    return out


def parameter_name(path) -> str:
    return ".".join(str(key.name) if hasattr(key, "name") else str(key.idx) for key in path)


def test_gradients_match_pytorch(case):
    ref = case.reference
    expected = {k.removeprefix("grad."): v for k, v in ref.items() if k.startswith("grad.")}
    if not expected:
        pytest.skip("no reference gradients for this model")

    grads = eqx.filter_grad(loss_fn)(case.model, jnp.asarray(ref["input_ids"]))
    flat = jax.tree_util.tree_flatten_with_path(eqx.filter(grads, eqx.is_array))[0]
    ours = to_pytorch_layout({parameter_name(path): np.asarray(g) for path, g in flat}, expected)

    unmatched = sorted(set(expected) - set(ours))
    assert not unmatched, f"{case.name}: no gradient for {unmatched}"
    errors = {
        name: float(np.linalg.norm(ours[name] - g) / max(np.linalg.norm(g), 1e-30))
        for name, g in expected.items()
    }
    worst = max(errors, key=errors.get)
    assert errors[worst] <= LIMIT, (
        f"{case.name}: gradient of {worst} differs from PyTorch's by {errors[worst]:.2e} "
        f"(relative); the limit is {LIMIT}"
    )
