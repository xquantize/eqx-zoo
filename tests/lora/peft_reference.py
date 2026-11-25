"""Build a PEFT LoRA layer and the matching eqx-zoo layer around the same random weights."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import peft
import torch
from transformers.pytorch_utils import Conv1D as TorchConv1D

from eqx_zoo.lora import LoRAConv1D, LoRALinear
from eqx_zoo.models.gpt2.model import Conv1D

DTYPES = {"float32": (torch.float32, jnp.float32), "bfloat16": (torch.bfloat16, jnp.bfloat16)}


@dataclasses.dataclass(frozen=True)
class Case:
    """One LoRA configuration: the layer kind, its sizes and PEFT's options."""

    kind: str = "linear"  # "linear" or "conv1d"
    in_features: int = 24
    out_features: int = 40
    r: int = 4
    alpha: float = 4
    rslora: bool = False
    lora_bias: bool = False
    dtype: str = "float32"  # the base layer's dtype; adapters are float32, as in PEFT


class _Holder(torch.nn.Module):
    def __init__(self, proj):
        super().__init__()
        self.proj = proj


def build(case: Case, seed: int = 0):
    """PEFT's LoRA layer (with weights loaded by name) and ours, with identical weights.

    Returns:
        The PEFT layer, our layer, and a function turning a NumPy input into the
        `(torch input, JAX input)` pair of the base dtype.
    """
    rng = np.random.default_rng(seed)
    torch_dtype, jax_dtype = DTYPES[case.dtype]
    i, o, r = case.in_features, case.out_features, case.r

    def normal(*shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    # The base layer's weights, in the layout of the layer kind ((in, out) for Conv1D).
    if case.kind == "linear":
        base = torch.nn.Linear(i, o)
        weight = normal(o, i, scale=i**-0.5)
    else:
        base = TorchConv1D(o, i)
        weight = normal(i, o, scale=i**-0.5)
    bias = normal(o, scale=1.0)
    with torch.no_grad():
        base.weight.copy_(torch.from_numpy(weight))
        base.bias.copy_(torch.from_numpy(bias))
    base.to(torch_dtype)

    # Random, non-zero adapter weights, in PEFT's file layout (names without the adapter name).
    lora = {
        "lora_A.weight": normal(r, i, scale=i**-0.5),
        "lora_B.weight": normal(o, r, scale=r**-0.5),
    }
    if case.lora_bias:
        lora["lora_B.bias"] = normal(o, scale=1.0)

    config = peft.LoraConfig(
        r=r,
        lora_alpha=case.alpha,
        use_rslora=case.rslora,
        lora_bias=case.lora_bias,
        fan_in_fan_out=case.kind == "conv1d",
        target_modules=["proj"],
    )
    peft_model = peft.get_peft_model(_Holder(base), config)
    state = {f"base_model.model.proj.{k}": torch.from_numpy(v) for k, v in lora.items()}
    peft.set_peft_model_state_dict(peft_model, state)
    peft_layer = peft_model.base_model.model.proj

    # Our layer, around the same base weights (exactly representable in the base dtype).
    key = jax.random.key(0)
    # Copies, because PEFT's merge updates the torch weights in place and NumPy shares memory.
    base_weight = jnp.asarray(base.weight.detach().float().numpy().copy(), dtype=jax_dtype)
    base_bias = jnp.asarray(base.bias.detach().float().numpy().copy(), dtype=jax_dtype)
    if case.kind == "linear":
        jax_base = eqx.nn.Linear(i, o, key=key, dtype=jax_dtype)
        wrapper = LoRALinear
    else:
        jax_base = Conv1D(i, o, key=key, dtype=jax_dtype)
        wrapper = LoRAConv1D
    jax_base = eqx.tree_at(lambda b: (b.weight, b.bias), jax_base, (base_weight, base_bias))
    ours = wrapper(
        jax_base,
        jnp.asarray(lora["lora_A.weight"]),
        jnp.asarray(lora["lora_B.weight"]),
        alpha=case.alpha,
        rslora=case.rslora,
        lora_B_bias=jnp.asarray(lora["lora_B.bias"]) if case.lora_bias else None,
    )

    def inputs(x: np.ndarray):
        return torch.from_numpy(x).to(torch_dtype), jnp.asarray(x, dtype=jax_dtype)

    return peft_layer, ours, inputs


def peft_forward(peft_layer, x) -> np.ndarray:
    with torch.no_grad():
        return peft_layer(x).float().numpy()


def our_forward(case: Case, ours, x) -> np.ndarray:
    out = jax.vmap(ours)(x) if case.kind == "linear" else ours(x)
    return np.asarray(out.astype(jnp.float32))


def peft_merged(peft_layer) -> tuple[np.ndarray, np.ndarray]:
    """The base layer's weight and bias after PEFT merges the adapter into it."""
    peft_layer.merge()
    base = peft_layer.get_base_layer()
    return base.weight.detach().float().numpy(), base.bias.detach().float().numpy()
