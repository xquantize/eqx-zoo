"""GPT-2-specific behaviour, beyond the shared causal-LM tests."""

import jax.numpy as jnp
import pytest
from support.tiny import build

from eqx_zoo.models.gpt2 import GPT2


def test_refuses_inputs_longer_than_its_positions(tmp_path):
    """Learned positions end at n_positions; JAX would silently clamp longer inputs."""
    build("tiny-gpt2", tmp_path)
    model = GPT2.from_pretrained(tmp_path)
    limit = model.config.n_positions
    model(jnp.zeros(limit, dtype=jnp.int32))  # the full context is fine
    with pytest.raises(ValueError, match="positions"):
        model(jnp.zeros(limit + 1, dtype=jnp.int32))
    with pytest.raises(ValueError, match="positions"):
        model.init_cache(limit + 1)
