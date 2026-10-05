"""Shared pytest options and fixtures: the reference cache and checkpoint weights."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from huggingface_hub import snapshot_download
from safetensors import safe_open
from support.reference import capture
from support.registry import ALL_CHECKPOINTS, CHECKPOINTS

# Request the most accurate float32 matrix multiplications. On accelerators, JAX's default
# float32 matmul precision can use reduced-precision arithmetic internally (TF32 on NVIDIA
# GPUs, bfloat16 on TPUs), which would make the float32 parity tests compare something
# other than float32. No effect on CPU.
jax.config.update("jax_default_matmul_precision", "highest")

REFERENCE_DIR = Path(__file__).parents[1] / "reference"
# Fixtures that read a downloaded checkpoint, and which checkpoint each one needs.
_CHECKPOINT_FIXTURES = {"qwen3_reference": "qwen3-0.6b", "qwen3_weight": "qwen3-0.6b"}


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-reference",
        action="store_true",
        help="Recompute the Hugging Face reference activations instead of using the cache.",
    )
    parser.addoption(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME",
        help="Run only the tests that need this checkpoint; may be given more than once.",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        for fixture in set(item.fixturenames) & _CHECKPOINT_FIXTURES.keys():
            item.add_marker(pytest.mark.checkpoint(_CHECKPOINT_FIXTURES[fixture]))
        for marker in item.iter_markers("checkpoint"):
            if not marker.args:
                raise pytest.UsageError(
                    f"{item.nodeid}: `checkpoint` marker needs a checkpoint name"
                )

    selected = set(config.getoption("--checkpoint"))
    if not selected:
        return
    unknown = selected - ALL_CHECKPOINTS.keys()
    if unknown:
        raise pytest.UsageError(f"unknown checkpoints: {sorted(unknown)}")

    keep, drop = [], []
    for item in items:
        names = {marker.args[0] for marker in item.iter_markers("checkpoint")}
        (keep if names & selected else drop).append(item)
    config.hook.pytest_deselected(items=drop)
    items[:] = keep


@pytest.fixture(scope="session")
def cached_reference(request):
    """Return `get(name, capture)`: load `reference/<name>.npz`, or create it with `capture()`."""
    regenerate = request.config.getoption("--regenerate-reference")
    loaded: dict[str, dict[str, np.ndarray]] = {}

    def get(name: str, capture) -> dict[str, np.ndarray]:
        if name not in loaded:
            path = REFERENCE_DIR / f"{name}.npz"
            if regenerate or not path.exists():
                REFERENCE_DIR.mkdir(exist_ok=True)
                np.savez(path, **capture())
            with np.load(path) as ref:
                loaded[name] = {key: ref[key] for key in ref.files}
        return loaded[name]

    return get


@pytest.fixture(scope="session")
def get_reference(cached_reference):
    """Return a loader for a causal LM checkpoint's reference activations."""
    return lambda name: cached_reference(name, lambda: capture(CHECKPOINTS[name]))


@pytest.fixture(scope="session")
def qwen3_reference(get_reference):
    """Qwen3-0.6B reference activations, for block-level tests."""
    return get_reference("qwen3-0.6b")


@pytest.fixture(scope="session")
def qwen3_weight():
    """Return a loader mapping a Qwen3-0.6B parameter name to a float32 JAX array."""
    path = Path(snapshot_download(CHECKPOINTS["qwen3-0.6b"], allow_patterns=["*.safetensors"]))
    with safe_open(path / "model.safetensors", framework="pt") as f:
        yield lambda name: jnp.asarray(f.get_tensor(name).float().numpy())
