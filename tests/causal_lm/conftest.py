"""Fixtures for the causal language model tests: every tiny model and checkpoint as a case."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from support.registry import CHECKPOINTS, TINY_MODELS
from support.tiny import build

from eqx_zoo import CausalLM


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    source: str
    model: CausalLM
    reference: dict[str, np.ndarray]


@pytest.fixture(
    scope="package",
    params=[
        *TINY_MODELS,
        *(pytest.param(name, marks=pytest.mark.checkpoint(name)) for name in CHECKPOINTS),
    ],
)
def case(request, get_reference, tmp_path_factory) -> Case:
    """Every tiny model and registered checkpoint in float32, with its reference activations."""
    name = request.param
    if name in TINY_MODELS:
        directory = tmp_path_factory.mktemp(name)
        reference = build(name, directory)
        source = str(directory)
    else:
        reference = get_reference(name)  # capture first, so the HF model is freed before ours
        source = CHECKPOINTS[name]
    return Case(name, source, CausalLM.from_pretrained(source, dtype=jnp.float32), reference)


@pytest.fixture(scope="package")
def bf16_model(case) -> CausalLM:
    """The same model as `case`, loaded in bfloat16 as a user would."""
    return CausalLM.from_pretrained(case.source, dtype=jnp.bfloat16)
