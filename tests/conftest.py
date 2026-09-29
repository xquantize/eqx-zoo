"""Shared fixtures: the model registry, reference activations and Hugging Face weights."""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from _reference import capture
from huggingface_hub import snapshot_download
from safetensors import safe_open

from eqx_zoo import CausalLM

# Checkpoints verified by the parity suite: test id -> Hugging Face repository.
MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}
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
    """Return a loader for a model's reference activations, capturing them if needed."""
    regenerate = request.config.getoption("--regenerate-reference")
    loaded: dict[str, dict[str, np.ndarray]] = {}

    def get(name: str) -> dict[str, np.ndarray]:
        if name not in loaded:
            path = REFERENCE_DIR / f"{name}.npz"
            if regenerate or not path.exists():
                REFERENCE_DIR.mkdir(exist_ok=True)
                np.savez(path, **capture(MODELS[name]))
            with np.load(path) as ref:
                loaded[name] = {key: ref[key] for key in ref.files}
        return loaded[name]

    return get


@pytest.fixture(scope="session", params=list(MODELS))
def case(request, get_reference) -> Case:
    """Each registered model in float32, with its reference activations."""
    name = request.param
    model = CausalLM.from_pretrained(MODELS[name], dtype=jnp.float32)
    return Case(name, model, get_reference(name))


@pytest.fixture(scope="session")
def qwen3_reference(get_reference):
    """Qwen3-0.6B reference activations, for block-level tests."""
    return get_reference("qwen3-0.6b")


@pytest.fixture(scope="session")
def qwen3_weight():
    """Return a loader mapping a Qwen3-0.6B parameter name to a float32 JAX array."""
    path = Path(snapshot_download(MODELS["qwen3-0.6b"], allow_patterns=["*.safetensors"]))
    with safe_open(path / "model.safetensors", framework="pt") as f:
        yield lambda name: jnp.asarray(f.get_tensor(name).float().numpy())
