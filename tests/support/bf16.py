"""Shared helpers for the bf16 accuracy tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Each accuracy test runs in both modes: op by op, and JIT-compiled.
MODES = [pytest.param(False, id="eager"), pytest.param(True, id="jit")]


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run(model, fn, *args, jit: bool) -> np.ndarray:
    """Apply `fn(model, *args)` JIT-compiled, or op by op, returning float32."""
    if jit:
        out = eqx.filter_jit(fn)(model, *args)
    else:
        with jax.disable_jit():
            out = fn(model, *args)
    return np.asarray(out.astype(jnp.float32))


def check(name: str, what: str, error: float, yardstick: float, factor: float) -> None:
    ratio = error / yardstick
    assert ratio <= factor, (
        f"{name}: bf16 {what} RMS error {error:.3g} is {ratio:.2f}x the reference's own bf16 "
        f"error ({yardstick:.3g}); the limit is {factor}x"
    )
