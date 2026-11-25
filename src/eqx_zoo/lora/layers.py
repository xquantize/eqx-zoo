"""LoRA wrappers for linear layers, reproducing the arithmetic of PEFT's `lora.Linear`."""

import math
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from eqx_zoo.layers._common import at_least_float32
from eqx_zoo.models.gpt2.model import Conv1D


def lora_scaling(r: int, alpha: float, *, rslora: bool = False) -> float:
    """The factor PEFT multiplies a LoRA update by.

    Args:
        r: Rank of the update.
        alpha: PEFT's `lora_alpha`.
        rslora: Whether to use rank-stabilised scaling, `alpha / sqrt(r)`, rather than
            `alpha / r`.

    Returns:
        `alpha / sqrt(r)` with `rslora`, else `alpha / r`.

    Raises:
        ValueError: If `r` is not positive.
    """
    if r <= 0:
        raise ValueError(f"r must be a positive integer, got {r}")
    return alpha / math.sqrt(r) if rslora else alpha / r


def _check_shapes(
    in_features: int,
    out_features: int,
    lora_A: Array,
    lora_B: Array,
    lora_B_bias: Array | None,
) -> None:
    if lora_A.ndim != 2 or lora_A.shape[1] != in_features:
        raise ValueError(f"lora_A must be (r, {in_features}), got {lora_A.shape}")
    r = lora_A.shape[0]
    if lora_B.shape != (out_features, r):
        raise ValueError(f"lora_B must be ({out_features}, {r}), got {lora_B.shape}")
    if lora_B_bias is not None and lora_B_bias.shape != (out_features,):
        raise ValueError(f"lora_B_bias must be ({out_features},), got {lora_B_bias.shape}")


def _update(
    x: Array, lora_A: Array, lora_B: Array, lora_B_bias: Array | None, scaling: float
) -> Array:
    """`scaling * (B (A x) + bias)`, with `x` cast to the adapter's dtype, as in PEFT."""
    update = (x.astype(lora_A.dtype) @ lora_A.T) @ lora_B.T
    if lora_B_bias is not None:
        update = update + lora_B_bias
    return update * scaling


def _merged_delta(lora_A: Array, lora_B: Array, scaling: float) -> Array:
    """`scaling * B @ A`, shaped (out, in), computed in at least float32."""
    dtype = at_least_float32(jnp.promote_types(lora_A.dtype, lora_B.dtype))
    return (lora_B.astype(dtype) @ lora_A.astype(dtype)) * scaling


def _merged_bias(
    bias: Array | None, lora_B_bias: Array | None, scaling: float, dtype: jnp.dtype
) -> Array | None:
    if lora_B_bias is None:
        return bias
    if bias is None:
        raise ValueError("cannot merge a LoRA bias into a layer that has no bias")
    return (bias.astype(dtype) + lora_B_bias.astype(dtype) * scaling).astype(bias.dtype)


class LoRALinear(eqx.Module):
    """An `equinox.nn.Linear` with a low-rank update: `base(x) + scaling * B (A x)`.

    The input is cast to the adapter's dtype for the update, the update is added to the base
    output, and the sum is cast back to the base output's dtype, as in PEFT. PEFT keeps
    adapters in float32 even when the base weights are bfloat16.

    Attributes:
        base: The wrapped layer.
        lora_A: Down-projection, shaped (r, in_features).
        lora_B: Up-projection, shaped (out_features, r).
        lora_B_bias: Bias of the up-projection (PEFT's `lora_bias`), or `None`.
        alpha: PEFT's `lora_alpha`.
        rslora: Whether the scaling is rank-stabilised, `alpha / sqrt(r)`.
    """

    _is_adapter_layer: ClassVar[bool] = True  # read by `refuse_adapters`; see `_loading.py`

    base: eqx.nn.Linear
    lora_A: Float[Array, "r in_features"]
    lora_B: Float[Array, "out_features r"]
    lora_B_bias: Float[Array, " out_features"] | None
    alpha: float = eqx.field(static=True)
    rslora: bool = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.nn.Linear,
        lora_A: Float[Array, "r in_features"],
        lora_B: Float[Array, "out_features r"],
        *,
        alpha: float,
        rslora: bool = False,
        lora_B_bias: Float[Array, " out_features"] | None = None,
    ):
        """Wrap a layer.

        Args:
            base: The layer to adapt.
            lora_A: Down-projection, shaped (r, in_features).
            lora_B: Up-projection, shaped (out_features, r).
            alpha: PEFT's `lora_alpha`; the update is scaled by `alpha / r`.
            rslora: Scale by `alpha / sqrt(r)` instead (rank-stabilised LoRA).
            lora_B_bias: Bias of the up-projection, or `None`.

        Raises:
            ValueError: If the shapes don't fit `base`, or the rank is not positive.
        """
        _check_shapes(base.in_features, base.out_features, lora_A, lora_B, lora_B_bias)
        self.base = base
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_B_bias = lora_B_bias
        self.alpha = float(alpha)
        self.rslora = rslora
        lora_scaling(lora_A.shape[0], alpha, rslora=rslora)  # validates the rank

    @property
    def r(self) -> int:
        """The rank of the update."""
        return self.lora_A.shape[0]

    @property
    def scaling(self) -> float:
        """The factor the update is multiplied by: `alpha / r`, or `alpha / sqrt(r)`."""
        return lora_scaling(self.r, self.alpha, rslora=self.rslora)

    def __call__(self, x: Float[Array, " in_features"]) -> Float[Array, " out_features"]:
        """Apply the layer to one vector.

        Args:
            x: Input.

        Returns:
            The base output plus the low-rank update, in the base output's dtype.
        """
        out = self.base(x)
        update = _update(x, self.lora_A, self.lora_B, self.lora_B_bias, self.scaling)
        return (out + update).astype(out.dtype)

    def merged(self) -> eqx.nn.Linear:
        """The base layer with the update folded in.

        The sum is computed in at least float32 and cast to the base dtype.

        Returns:
            A layer equal to this one, with weight `W + scaling * B @ A` (and bias
            `b + scaling * lora_B_bias`, if there is one).

        Raises:
            ValueError: If there is a `lora_B_bias` but the base layer has no bias.
        """
        delta = _merged_delta(self.lora_A, self.lora_B, self.scaling)
        weight = (self.base.weight.astype(delta.dtype) + delta).astype(self.base.weight.dtype)
        bias = _merged_bias(self.base.bias, self.lora_B_bias, self.scaling, delta.dtype)
        return eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            self.base,
            (weight, bias),
            is_leaf=lambda leaf: leaf is None,
        )


class LoRAConv1D(eqx.Module):
    """GPT-2's `Conv1D` with a low-rank update: `base(x) + scaling * B (A x)`.

    `Conv1D` stores its weight as (in, out), but the update is shaped as for a linear layer,
    with `lora_A` (r, in) and `lora_B` (out, r), as PEFT stores it (its `fan_in_fan_out`):
    only merging transposes it.

    Attributes:
        base: The wrapped layer.
        lora_A: Down-projection, shaped (r, in_features).
        lora_B: Up-projection, shaped (out_features, r).
        lora_B_bias: Bias of the up-projection (PEFT's `lora_bias`), or `None`.
        alpha: PEFT's `lora_alpha`.
        rslora: Whether the scaling is rank-stabilised, `alpha / sqrt(r)`.
    """

    _is_adapter_layer: ClassVar[bool] = True  # read by `refuse_adapters`; see `_loading.py`

    base: Conv1D
    lora_A: Float[Array, "r in_features"]
    lora_B: Float[Array, "out_features r"]
    lora_B_bias: Float[Array, " out_features"] | None
    alpha: float = eqx.field(static=True)
    rslora: bool = eqx.field(static=True)

    def __init__(
        self,
        base: Conv1D,
        lora_A: Float[Array, "r in_features"],
        lora_B: Float[Array, "out_features r"],
        *,
        alpha: float,
        rslora: bool = False,
        lora_B_bias: Float[Array, " out_features"] | None = None,
    ):
        """Wrap a layer.

        Args:
            base: The layer to adapt.
            lora_A: Down-projection, shaped (r, in_features).
            lora_B: Up-projection, shaped (out_features, r).
            alpha: PEFT's `lora_alpha`; the update is scaled by `alpha / r`.
            rslora: Scale by `alpha / sqrt(r)` instead (rank-stabilised LoRA).
            lora_B_bias: Bias of the up-projection, or `None`.

        Raises:
            ValueError: If the shapes don't fit `base`, or the rank is not positive.
        """
        in_features, out_features = base.weight.shape
        _check_shapes(in_features, out_features, lora_A, lora_B, lora_B_bias)
        self.base = base
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_B_bias = lora_B_bias
        self.alpha = float(alpha)
        self.rslora = rslora
        lora_scaling(lora_A.shape[0], alpha, rslora=rslora)  # validates the rank

    @property
    def r(self) -> int:
        """The rank of the update."""
        return self.lora_A.shape[0]

    @property
    def scaling(self) -> float:
        """The factor the update is multiplied by: `alpha / r`, or `alpha / sqrt(r)`."""
        return lora_scaling(self.r, self.alpha, rslora=self.rslora)

    def __call__(
        self, x: Float[Array, "*batch in_features"]
    ) -> Float[Array, "*batch out_features"]:
        """Apply the layer.

        Args:
            x: Input.

        Returns:
            The base output plus the low-rank update, in the base output's dtype.
        """
        out = self.base(x)
        update = _update(x, self.lora_A, self.lora_B, self.lora_B_bias, self.scaling)
        return (out + update).astype(out.dtype)

    def merged(self) -> Conv1D:
        """The base layer with the update folded in.

        The sum is computed in at least float32 and cast to the base dtype.

        Returns:
            A layer equal to this one, with weight `W + scaling * (B @ A).T` (and bias
            `b + scaling * lora_B_bias`, if there is one).

        """
        delta = _merged_delta(self.lora_A, self.lora_B, self.scaling).T
        weight = (self.base.weight.astype(delta.dtype) + delta).astype(self.base.weight.dtype)
        bias = _merged_bias(self.base.bias, self.lora_B_bias, self.scaling, delta.dtype)
        return eqx.tree_at(lambda layer: (layer.weight, layer.bias), self.base, (weight, bias))
