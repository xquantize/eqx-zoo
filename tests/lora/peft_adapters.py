"""Making PEFT adapters, and reading PEFT models' outputs, for the LoRA tests."""

import numpy as np
import peft
import torch
import transformers
from support.tiny import PROMPT_IDS

OPTIONS = {
    "plain": {"r": 4, "alpha": 4},
    "alpha-not-r": {"r": 4, "alpha": 10},
    "rslora": {"r": 4, "alpha": 10, "rslora": True},
    "lora-bias": {"r": 4, "alpha": 10, "lora_bias": True},
}


def hf_model(checkpoint, name, dtype=torch.float32):
    """A tiny Hugging Face model in `dtype`.

    Loaded and then converted, which is exact for the tiny models (saved entirely in float32,
    so converting to bfloat16 rounds as loading in bfloat16 does) and works on every supported
    transformers version: older ones call the loading argument `torch_dtype`, newer ones `dtype`.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint(name), attn_implementation="eager"
    )
    return model.to(dtype)


def peft_config(options) -> peft.LoraConfig:
    return peft.LoraConfig(
        r=options["r"],
        lora_alpha=options["alpha"],
        use_rslora=options.get("rslora", False),
        lora_bias=options.get("lora_bias", False),
        target_modules="all-linear",
    )


def make_peft_adapter(checkpoint, name, options, seed=0, dtype=torch.float32):
    """PEFT's model with a random adapter (trainable), around the tiny checkpoint."""
    torch.manual_seed(seed)
    model = peft.get_peft_model(hf_model(checkpoint, name, dtype), peft_config(options))
    with torch.no_grad():
        for parameter_name, parameter in model.named_parameters():
            if "lora_" in parameter_name:
                parameter.normal_(0.0, 0.1)
    return model.eval()


def peft_logits(model) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor([PROMPT_IDS])).logits[0].float().numpy()


def relative_error(got, expected) -> float:
    return float(
        np.abs(np.asarray(got, dtype=np.float32) - expected).max() / np.abs(expected).max()
    )
