"""Pytest fixtures and options, wiring the registry and reference capture into the tests."""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from huggingface_hub import snapshot_download
from safetensors import safe_open
from support.reference import capture
from support.registry import CHECKPOINTS

from eqx_zoo import CausalLM

REFERENCE_DIR = Path(__file__).parents[1] / "reference"


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    model: CausalLM
    reference: dict[str, np.ndarray]


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-reference",
        action="store_true",
        help="Recompute the Hugging Face reference activations instead of using the cache.",
    )


@pytest.fixture(scope="session")
def get_reference(request):
    """Return a loader for a checkpoint's reference activations, capturing them if needed."""
    regenerate = request.config.getoption("--regenerate-reference")
    loaded: dict[str, dict[str, np.ndarray]] = {}

    def get(name: str) -> dict[str, np.ndarray]:
        if name not in loaded:
            path = REFERENCE_DIR / f"{name}.npz"
            if regenerate or not path.exists():
                REFERENCE_DIR.mkdir(exist_ok=True)
                np.savez(path, **capture(CHECKPOINTS[name]))
            with np.load(path) as ref:
                loaded[name] = {key: ref[key] for key in ref.files}
        return loaded[name]

    return get


@pytest.fixture(scope="session", params=list(CHECKPOINTS))
def case(request, get_reference) -> Case:
    """Each registered checkpoint in float32, with its reference activations."""
    name = request.param
    reference = get_reference(name)  # capture first, so the HF model is freed before ours loads
    model = CausalLM.from_pretrained(CHECKPOINTS[name], dtype=jnp.float32)
    return Case(name, model, reference)


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
