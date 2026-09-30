"""Build tiny randomly initialised Hugging Face models for fast architecture tests."""

from pathlib import Path

import numpy as np

from support.reference import capture_model
from support.registry import TINY_MODELS

# Shared by every tiny model; per-model overrides live in `TINY_MODELS`.
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
PROMPT_IDS = [1, 5, 9, 17, 33, 65, 127, 3]


def build(name: str, directory: Path) -> dict[str, np.ndarray]:
    """Create a seeded random model, save it to `directory`, and capture its references."""
    import torch
    import transformers

    config_class, overrides = TINY_MODELS[name]
    config = getattr(transformers, config_class)(**BASE_CONFIG, **overrides)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager")

    # Default initialisation sets norm weights to 1 and biases to 0, which would hide bugs
    # such as a dropped norm scale or bias. Randomise every parameter instead.
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            if "norm" in param_name:
                param.copy_(1 + 0.1 * torch.randn_like(param))
            else:
                param.copy_(0.05 * torch.randn_like(param))

    model.save_pretrained(directory)
    return capture_model(model.eval(), torch.tensor([PROMPT_IDS]))
