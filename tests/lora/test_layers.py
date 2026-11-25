"""LoRA layers against PEFT's own `lora.Linear` layer, around the same random weights.

Tolerances come from measurement. Over 20 seeds, 4 LoRA options, 2 shapes and both layer
kinds, the largest disagreements with PEFT (relative to the largest output magnitude) were:

- float32 forward: 2.9e-7 (the base layer alone disagrees by up to 2.0e-7 for Linear, 0 for
  Conv1D, so this is ordinary float32 summation-order noise);
- bfloat16 forward: 7.75e-3, one bfloat16 rounding step at the largest outputs (2**-7), and
  the same as the base layer alone, which disagrees by up to 7.5e-3 (JAX and PyTorch round
  bfloat16 matmuls differently);
- merged weights and biases: exactly 0 in both dtypes.

Each tolerance below is about 2-3x the measurement, and far below what the mutations that
the tests are meant to catch produce (a wrong scaling or transposition changes the output
by order 1).
"""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("peft")

from eqx_zoo.lora import LoRAConv1D, LoRALinear, lora_scaling  # noqa: E402
from lora.peft_reference import (  # noqa: E402
    Case,
    build,
    our_forward,
    peft_forward,
    peft_merged,
)

# Relative to the largest magnitude in the PEFT output.
FORWARD_TOLERANCE = {"float32": 1e-6, "bfloat16": 2**-6}
MERGE_TOLERANCE = {"float32": 1e-6, "bfloat16": 2**-7}

OPTIONS = {
    "plain": {},
    "alpha-not-r": {"alpha": 10},
    "rslora": {"alpha": 10, "rslora": True},
    "lora-bias": {"alpha": 10, "lora_bias": True},
    "rslora-lora-bias": {"alpha": 10, "rslora": True, "lora_bias": True},
}
# Non-square shapes catch a swapped A and B, square ones a missing Conv1D transposition.
SHAPES = [(24, 40), (32, 32)]
CASES = [
    pytest.param(
        Case(kind=kind, in_features=i, out_features=o, dtype=dtype, **options),
        id=f"{kind}-{i}x{o}-{name}-{dtype}",
    )
    for kind, (i, o), (name, options), dtype in itertools.product(
        ["linear", "conv1d"], SHAPES, OPTIONS.items(), ["float32", "bfloat16"]
    )
]


def _inputs(case: Case, inputs):
    x = np.random.default_rng(1).standard_normal((7, case.in_features)).astype(np.float32)
    return inputs(x)


@pytest.mark.parametrize("case", CASES)
def test_forward_matches_peft(case):
    peft_layer, ours, inputs = build(case)
    torch_x, jax_x = _inputs(case, inputs)
    expected = peft_forward(peft_layer, torch_x)
    out = jax.vmap(ours)(jax_x) if case.kind == "linear" else ours(jax_x)
    assert out.dtype == jax_x.dtype, "the output must be in the base layer's dtype, as in PEFT"
    error = np.abs(np.asarray(out.astype(jnp.float32)) - expected).max() / np.abs(expected).max()
    assert error <= FORWARD_TOLERANCE[case.dtype], f"relative error {error:.3g}"


@pytest.mark.parametrize("case", CASES)
def test_merged_layer_matches_peft_merge(case):
    peft_layer, ours, _ = build(case)
    merged = ours.merged()
    weight, bias = peft_merged(peft_layer)
    assert merged.weight.dtype == ours.base.weight.dtype
    assert merged.bias.dtype == ours.base.bias.dtype
    for got, expected in [(merged.weight, weight), (merged.bias, bias)]:
        error = np.abs(np.asarray(got.astype(jnp.float32)) - expected).max()
        assert error / np.abs(expected).max() <= MERGE_TOLERANCE[case.dtype]


@pytest.mark.parametrize("kind", ["linear", "conv1d"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_merged_layer_agrees_with_the_adapter(kind, dtype):
    """Folding the update into the weights doesn't change what the layer computes."""
    case = Case(kind=kind, dtype=dtype, alpha=10, lora_bias=True)
    _, ours, inputs = build(case)
    _, jax_x = _inputs(case, inputs)
    merged = ours.merged()
    expected = our_forward(case, ours, jax_x)
    got = our_forward(case, merged, jax_x)
    # Measured: 4.3e-7 in float32 and 7.5e-3 (one bfloat16 rounding step) in bfloat16.
    tolerance = {"float32": 2e-6, "bfloat16": 2**-6}[dtype]
    assert np.abs(got - expected).max() / np.abs(expected).max() <= tolerance


def test_zero_b_is_the_identity():
    """PEFT initialises B to zero, so a fresh adapter leaves the layer unchanged."""
    _, ours, inputs = build(Case())
    _, jax_x = _inputs(Case(), inputs)
    zeroed = eqx.tree_at(lambda m: m.lora_B, ours, jnp.zeros_like(ours.lora_B))
    assert jnp.array_equal(jax.vmap(zeroed)(jax_x), jax.vmap(ours.base)(jax_x))


def test_scaling():
    assert lora_scaling(4, 10) == 2.5
    assert lora_scaling(4, 10, rslora=True) == 5.0
    assert lora_scaling(1, 3, rslora=True) == 3.0
    with pytest.raises(ValueError, match="positive"):
        lora_scaling(0, 1)


def test_scaling_is_static():
    _, ours, _ = build(Case())
    assert not any(isinstance(x, float) for x in jax.tree_util.tree_leaves(ours))


@pytest.mark.parametrize("wrapper", ["linear", "conv1d"])
def test_shape_mismatches_are_rejected(wrapper):
    _, ours, _ = build(Case(kind=wrapper))
    cls = LoRALinear if wrapper == "linear" else LoRAConv1D
    a, b = ours.lora_A, ours.lora_B
    with pytest.raises(ValueError, match="lora_A"):
        cls(ours.base, a.T, b, alpha=1.0)
    with pytest.raises(ValueError, match="lora_B"):
        cls(ours.base, a, b.T, alpha=1.0)
    with pytest.raises(ValueError, match="lora_B_bias"):
        cls(ours.base, a, b, alpha=1.0, lora_B_bias=jnp.zeros(3))


def test_lora_bias_needs_a_base_bias_to_merge():
    base = eqx.nn.Linear(4, 3, use_bias=False, key=jax.random.key(0))
    layer = LoRALinear(base, jnp.ones((2, 4)), jnp.ones((3, 2)), alpha=1.0, lora_B_bias=jnp.ones(3))
    with pytest.raises(ValueError, match="no bias"):
        layer.merged()


def test_layers_without_a_base_bias_merge_without_one():
    base = eqx.nn.Linear(4, 3, use_bias=False, key=jax.random.key(0))
    layer = LoRALinear(base, jnp.ones((2, 4)), jnp.ones((3, 2)), alpha=1.0)
    assert layer.merged().bias is None
