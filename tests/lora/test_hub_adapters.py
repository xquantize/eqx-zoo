"""Real LoRA adapters from the Hub, on their base checkpoints, against PEFT.

These are checkpoint tests (`pytest --checkpoint NAME`, names in `support/registry.py`): the
reference is PEFT's own output, captured once and cached in `reference/`. The base model is
loaded in float32, with `PeftModel.from_pretrained`.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from support.models import model_class
from support.reference import capture_adapter
from support.registry import ADAPTER_CHECKPOINTS

pytest.importorskip("peft")
import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from eqx_zoo import generate  # noqa: E402
from eqx_zoo.lora import load_adapter, merge, save_adapter  # noqa: E402
from eqx_zoo.lora.adapt import adapter_layers  # noqa: E402

# The adapter that also makes a save_adapter round trip through PEFT (the others would only
# repeat it on bigger models).
ROUND_TRIP = "qwen3-0.6b-tldr-lora"
# Logit errors are relative to the largest logit, and allowed to be this many times the error
# of the base model alone (ours against Transformers), which the adapter doesn't change.
# Measured (float32, Qwen3-0.6B, Qwen2.5-0.5B-Instruct and SmolLM2-135M adapters): the base
# model's error is 2.9e-6 to 4.4e-6; the adapted and merged logits differ from PEFT's by 2.5e-6
# to 7.7e-6, at most 1.8x that error. The adapters change the logits by 1.7e-2 to 2.1e-1, and
# a wrong scaling or layer changes them by that much.
YARDSTICK_FACTOR = 4.0
# Smallest tolerance, for when the base model's own error is very small.
FLOOR = 1e-6


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    base: object
    model: object
    reference: dict[str, np.ndarray]


@pytest.fixture(
    scope="module",
    params=[pytest.param(n, marks=pytest.mark.checkpoint(n)) for n in ADAPTER_CHECKPOINTS],
)
def case(request, cached_reference) -> Case:
    """The base model with the adapter loaded here, and PEFT's reference outputs."""
    name = request.param
    base_repo, adapter_repo, revision = ADAPTER_CHECKPOINTS[name]
    # Capture first, so that the Hugging Face model is freed before ours is loaded.
    reference = cached_reference(name, lambda: capture_adapter(base_repo, adapter_repo, revision))
    base = model_class(base_repo).from_pretrained(base_repo, dtype=jnp.float32)
    model = load_adapter(base, adapter_repo, revision=revision)
    return Case(name, base, model, reference)


def relative_error(got, expected) -> float:
    return float(np.abs(np.asarray(got) - expected).max() / np.abs(expected).max())


def tolerance(case) -> float:
    """What a logits comparison may differ by: a multiple of the base model's own error."""
    ids = jnp.asarray(case.reference["input_ids"])
    base_error = relative_error(case.base(ids)[0], case.reference["base_logits"])
    return max(YARDSTICK_FACTOR * base_error, FLOOR)


def test_the_same_layers_are_adapted(case):
    assert set(adapter_layers(case.model)) == set(case.reference["adapted_modules"].tolist())


def test_the_adapter_matters(case):
    """PEFT's logits with and without the adapter differ, so matching them is meaningful."""
    assert relative_error(case.reference["logits"], case.reference["base_logits"]) > 1e-3


def test_logits_match_peft(case):
    ids = jnp.asarray(case.reference["input_ids"])
    error = relative_error(case.model(ids)[0], case.reference["logits"])
    assert error <= tolerance(case), f"{error:.2e} against {tolerance(case):.2e}"


def test_greedy_tokens_match_peft(case):
    expected = case.reference["generated"]
    tokens = generate(case.model, jnp.asarray(case.reference["input_ids"]), len(expected))
    np.testing.assert_array_equal(tokens, expected)


def test_merged_logits_match_peft(case):
    ids = jnp.asarray(case.reference["input_ids"])
    error = relative_error(merge(case.model)(ids)[0], case.reference["merged_logits"])
    assert error <= tolerance(case), f"{error:.2e} against {tolerance(case):.2e}"


def test_a_saved_adapter_loads_in_peft(case, tmp_path):
    if case.name != ROUND_TRIP:
        pytest.skip("the round trip is checked on one adapter")
    base_repo = ADAPTER_CHECKPOINTS[case.name][0]
    save_adapter(case.model, tmp_path)
    # Loaded and then converted, so that this works with every transformers version (older ones
    # call the loading argument `torch_dtype`, newer ones `dtype`). Exact for these bases, which
    # are stored in a single dtype no wider than float32.
    base = transformers.AutoModelForCausalLM.from_pretrained(
        base_repo, attn_implementation="eager"
    ).float()
    reloaded = peft.PeftModel.from_pretrained(base, tmp_path).eval()
    ids = torch.tensor(case.reference["input_ids"])[None]
    with torch.no_grad():
        logits = reloaded(ids).logits[0].float().numpy()
    error = relative_error(logits, case.reference["logits"])
    assert error <= tolerance(case), f"{error:.2e} against {tolerance(case):.2e}"
