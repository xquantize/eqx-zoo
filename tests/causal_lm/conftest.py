"""Fixtures for the causal language model tests: every tiny model and checkpoint as a case."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from support.models import model_class
from support.registry import CHECKPOINTS, TINY_MODELS
from support.tiny import build

from eqx_zoo import CausalLM
from eqx_zoo.models.gpt2 import GPT2
from eqx_zoo.models.gpt_neox import GPTNeoX


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    source: str
    model: CausalLM | GPT2 | GPTNeoX
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
    model = model_class(source).from_pretrained(source, dtype=jnp.float32)
    return Case(name, source, model, reference)


@pytest.fixture(scope="package")
def bf16_model(case) -> CausalLM | GPT2 | GPTNeoX:
    """The same model as `case`, loaded in bfloat16 as a user would."""
    return model_class(case.source).from_pretrained(case.source, dtype=jnp.bfloat16)
