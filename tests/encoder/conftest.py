"""Fixtures for the encoder tests: every tiny encoder as a case."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from support.registry import TINY_ENCODERS
from support.tiny import build_encoder

from eqx_zoo import Encoder


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    source: str
    model: Encoder
    reference: dict[str, np.ndarray]


@pytest.fixture(scope="session", params=list(TINY_ENCODERS))
def case(request, tmp_path_factory) -> Case:
    """Every tiny encoder in float32, with its reference activations."""
    name = request.param
    directory = tmp_path_factory.mktemp(name)
    reference = build_encoder(name, directory)
    return Case(
        name, str(directory), Encoder.from_pretrained(directory, dtype=jnp.float32), reference
    )
