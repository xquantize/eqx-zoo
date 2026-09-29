"""Verified Equinox ports of pretrained models, numerically matched against Hugging Face."""

from importlib.metadata import version

from eqx_zoo.qwen3 import Qwen3Config, Qwen3ForCausalLM

__all__ = ["Qwen3Config", "Qwen3ForCausalLM"]
__version__ = version("eqx-zoo")
