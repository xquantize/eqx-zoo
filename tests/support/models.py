"""Pick the eqx-zoo model class for a checkpoint."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from eqx_zoo import CausalLM
from eqx_zoo.models.gpt2 import GPT2
from eqx_zoo.models.gpt_neox import GPTNeoX

# Architectures with their own model class; every other architecture is a `CausalLM`.
CLASSES = {
    "GPT2LMHeadModel": GPT2,
    "GPTNeoXForCausalLM": GPTNeoX,
}


def model_class(source: str | Path):
    """The eqx-zoo class that loads a checkpoint, from its `config.json` architecture.

    Args:
        source: A local checkpoint directory, or a Hub repository id.

    Returns:
        The model class: `CausalLM` unless the architecture has its own class.
    """
    path = Path(source)
    config = path / "config.json" if path.is_dir() else Path(hf_hub_download(source, "config.json"))
    architecture = json.loads(config.read_text())["architectures"][0]
    return CLASSES.get(architecture, CausalLM)
