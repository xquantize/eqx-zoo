"""Gradient parity with PyTorch for decoder embedders, on the tiny random models.

The loss is sum(hidden * probe), with a fixed random probe stored in the reference; see
tests/support/gradients.py for the limit.
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
    ours = {k.removeprefix("model."): v for k, v in gradients_by_name(grads).items()}
    check_gradients(case.name, ours, expected)
