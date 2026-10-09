"""Gradient parity with PyTorch for encoders, on the tiny random models.

Encoders have no language-model head, so the loss is sum(hidden * probe), with a fixed random
probe stored in the reference; see tests/support/gradients.py for the limit.
"""

import equinox as eqx
import jax.numpy as jnp
from support.gradients import check_gradients, expected_gradients, gradients_by_name


def test_gradients_match_pytorch(case):
    expected = expected_gradients(case.reference)
    probe = jnp.asarray(case.reference["grad_probe"])
    grads = eqx.filter_grad(lambda m, ids: jnp.sum(m(ids) * probe))(
        case.model, jnp.asarray(case.reference["input_ids"])
    )
    check_gradients(case.name, gradients_by_name(grads), expected)
