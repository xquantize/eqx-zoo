"""Build tiny randomly initialised Hugging Face models for fast architecture tests."""

from pathlib import Path

import numpy as np

from support.reference import bf16_logits, capture_encoder, capture_model
from support.registry import TINY_ENCODERS, TINY_MODELS

# Shared by every tiny causal LM; per-model overrides live in `TINY_MODELS`.
BASE_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
}
# Shared by every tiny encoder; per-model overrides live in `TINY_ENCODERS`.
ENCODER_BASE_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "max_position_embeddings": 64,
    "type_vocab_size": 2,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-12,
}
PROMPT_IDS = [1, 5, 9, 17, 33, 65, 127, 3]


def _randomise(model) -> None:
    """Randomise every parameter so that bugs such as a dropped norm scale can't hide.

    Default initialisation sets norm weights to 1 and biases to 0, which would make such
    bugs invisible. Norm scales become `1 + noise`; everything else becomes small noise.
    """
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("weight") and "norm" in name.lower():
                param.copy_(1 + 0.1 * torch.randn_like(param))
            else:
                param.copy_(0.05 * torch.randn_like(param))


def build(name: str, directory: Path) -> dict[str, np.ndarray]:
    """Create a seeded random causal LM, save it to `directory`, and capture its references."""
    import torch
    import transformers

    config_class, overrides = TINY_MODELS[name]
    config = getattr(transformers, config_class)(**(BASE_CONFIG | overrides))
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    _randomise(model)

    model.save_pretrained(directory)
    ids = torch.tensor([PROMPT_IDS])
    return {**capture_model(model.eval(), ids), "bf16_logits": bf16_logits(model, ids)}


def build_encoder(name: str, directory: Path) -> dict[str, np.ndarray]:
    """Create a seeded random encoder, save it to `directory`, and capture its references."""
    import torch
    import transformers

    config_class, overrides = TINY_ENCODERS[name]
    config = getattr(transformers, config_class)(**(ENCODER_BASE_CONFIG | overrides))
    torch.manual_seed(0)
    model = transformers.AutoModel.from_config(config, attn_implementation="eager")
    _randomise(model)

    model.save_pretrained(directory)
    return capture_encoder(model.eval(), torch.tensor([PROMPT_IDS]))
