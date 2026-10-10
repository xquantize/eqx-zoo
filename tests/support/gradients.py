"""Shared helpers for gradient parity tests against PyTorch."""

import equinox as eqx
import jax
import numpy as np
import pytest

# Relative error allowed per parameter. Measured: correct code is within 1.4e-6, while a
# `stop_gradient` that leaves every forward value unchanged is off by 39-105%.
LIMIT = 1e-4


def expected_gradients(reference: dict) -> dict[str, np.ndarray]:
    """PyTorch's gradients from a reference, by parameter name; skips if there are none."""
    expected = {k.removeprefix("grad."): v for k, v in reference.items() if k.startswith("grad.")}
    if not expected:
        pytest.skip("no reference gradients for this model")
    return expected


def gradients_by_name(grads) -> dict[str, np.ndarray]:
    """Our gradients, keyed by parameter name (which mirrors Hugging Face's)."""
    flat = jax.tree_util.tree_flatten_with_path(eqx.filter(grads, eqx.is_array))[0]
    return {
        ".".join(str(k.name) if hasattr(k, "name") else str(k.idx) for k in path): np.asarray(g)
        for path, g in flat
    }


def check_gradients(name: str, ours: dict, expected: dict) -> None:
    """Every PyTorch gradient must have a match in ours, within LIMIT relative error."""
    unmatched = sorted(set(expected) - set(ours))
    assert not unmatched, f"{name}: no gradient for {unmatched}"
    # Each error is relative to the parameter's own gradient norm, or to 0.1% of the median
    # norm, whichever is larger: some gradients are exactly zero in theory (e.g. an attention
    # key bias without RoPE, which softmax cancels), and their float32 noise has no scale of
    # its own.
    floor = 1e-3 * float(np.median([np.linalg.norm(g) for g in expected.values()]))
    errors = {
        n: float(np.linalg.norm(ours[n] - g) / max(np.linalg.norm(g), floor))
        for n, g in expected.items()
    }
    worst = max(errors, key=errors.get)
    assert errors[worst] <= LIMIT, (
        f"{name}: gradient of {worst} differs from PyTorch's by {errors[worst]:.2e} "
        f"(relative); the limit is {LIMIT}"
    )
