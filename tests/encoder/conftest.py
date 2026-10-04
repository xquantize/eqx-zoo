"""Fixtures for the encoder tests: every tiny encoder and checkpoint as a case."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from support.reference import capture_encoder_checkpoint
from support.registry import ENCODER_CHECKPOINTS, TINY_ENCODERS
from support.tiny import build_encoder

from eqx_zoo import Encoder


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    source: str
    model: Encoder
    reference: dict[str, np.ndarray]


@pytest.fixture(
    scope="session",
    params=[
        *TINY_ENCODERS,
        *(pytest.param(name, marks=pytest.mark.checkpoint(name)) for name in ENCODER_CHECKPOINTS),
    ],
)
def case(request, cached_reference, tmp_path_factory) -> Case:
    """Every tiny encoder and registered checkpoint in float32, with its reference."""
    name = request.param
    if name in TINY_ENCODERS:
        directory = tmp_path_factory.mktemp(name)
        reference = build_encoder(name, directory)
        source = str(directory)
    else:
        source = ENCODER_CHECKPOINTS[name]
        reference = cached_reference(name, lambda: capture_encoder_checkpoint(source))
    return Case(name, source, Encoder.from_pretrained(source, dtype=jnp.float32), reference)


@pytest.fixture(scope="session")
def bf16_model(case) -> Encoder:
    """The same encoder as `case`, loaded in bfloat16 as a user would."""
    return Encoder.from_pretrained(case.source, dtype=jnp.bfloat16)
