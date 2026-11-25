"""Low-rank adaptation (LoRA), compatible with Hugging Face's PEFT library.

A LoRA layer wraps a linear layer `y = W x + b` and adds a trainable low-rank update,
`y = W x + b + scaling * B (A x)`, where `A` has shape `(r, in)` and `B` has shape `(out, r)`,
exactly as in PEFT's `lora.Linear`. Only plain LoRA is implemented; variants such as DoRA are not.
"""

from eqx_zoo.lora.layers import LoRAConv1D, LoRALinear, lora_scaling

__all__ = ["LoRAConv1D", "LoRALinear", "lora_scaling"]
