"""Fixtures for the decoder embedder tests: every tiny embedder and checkpoint as a case."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from support.reference import capture_embedder_checkpoint
from support.registry import EMBEDDER_CHECKPOINTS, TINY_EMBEDDERS
from support.tiny import build_embedder

from eqx_zoo import DecoderEmbedder


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    source: str
    model: DecoderEmbedder
    reference: dict[str, np.ndarray]


@pytest.fixture(
    scope="session",
    params=[
        *TINY_EMBEDDERS,
        *(pytest.param(name, marks=pytest.mark.checkpoint(name)) for name in EMBEDDER_CHECKPOINTS),
    ],
)
def case(request, cached_reference, tmp_path_factory) -> Case:
    """Every tiny embedder and registered checkpoint in float32, with its reference."""
    name = request.param
    if name in TINY_EMBEDDERS:
        directory = tmp_path_factory.mktemp(name)
        reference = build_embedder(name, directory)
        source = str(directory)
    else:
        source = EMBEDDER_CHECKPOINTS[name]
        reference = cached_reference(name, lambda: capture_embedder_checkpoint(source))
    model = DecoderEmbedder.from_pretrained(source, dtype=jnp.float32)
    return Case(name, source, model, reference)
