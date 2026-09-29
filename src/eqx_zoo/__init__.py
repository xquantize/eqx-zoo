"""Verified Equinox ports of pretrained models, numerically matched against Hugging Face."""

from importlib.metadata import version

from eqx_zoo.generate import generate
from eqx_zoo.layers import Cache
from eqx_zoo.qwen3 import Qwen3Config, Qwen3ForCausalLM

__all__ = ["Cache", "Qwen3Config", "Qwen3ForCausalLM", "generate"]
__version__ = version("eqx-zoo")
