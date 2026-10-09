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


def parameter_name(path) -> str:
    return ".".join(str(key.name) if hasattr(key, "name") else str(key.idx) for key in path)


def test_gradients_match_pytorch(case):
    ref = case.reference
    expected = {k.removeprefix("grad."): v for k, v in ref.items() if k.startswith("grad.")}
    if not expected:
        pytest.skip("no reference gradients for this model")
    if "moe" in case.name:
        pytest.skip("Hugging Face stores mixture-of-experts weights fused; mapped separately")

    grads = eqx.filter_grad(loss_fn)(case.model, jnp.asarray(ref["input_ids"]))
    flat = jax.tree_util.tree_flatten_with_path(eqx.filter(grads, eqx.is_array))[0]
    ours = {parameter_name(path): np.asarray(g) for path, g in flat}

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
