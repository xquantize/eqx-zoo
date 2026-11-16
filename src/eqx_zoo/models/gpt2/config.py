"""GPT-2 configuration, translated from Hugging Face `config.json` files."""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class GPT2Config:
    """Hyperparameters of a GPT-2 language model.

    Field names follow Hugging Face's GPT-2 configuration. Build one from a checkpoint's
    `config.json` with `GPT2Config.from_hf`.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        n_embd: Model (residual stream) dimension.
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads.
        n_positions: Number of learned positions, which is the longest supported input.
        n_inner: Inner dimension of each MLP.
        layer_norm_epsilon: Epsilon of every LayerNorm.
        pad_token_id: Id of the padding token, or `None`.
    """

    vocab_size: int
    n_embd: int
    n_layer: int
    n_head: int
    n_positions: int
    n_inner: int
    layer_norm_epsilon: float
    pad_token_id: int | None = None

    @classmethod
    def from_hf(cls, config: dict[str, Any]) -> "GPT2Config":
        """Translate a parsed Hugging Face `config.json`.

        Args:
            config: The parsed JSON.

        Returns:
            The corresponding config.

        Raises:
            ValueError: If the architecture is not GPT-2, or the sizes are inconsistent.
            NotImplementedError: If the checkpoint needs a feature not implemented yet.
        """
        architecture = (config.get("architectures") or ["<missing>"])[0]
        if architecture != "GPT2LMHeadModel":
            raise ValueError(f"expected GPT2LMHeadModel, not {architecture!r}")
        activation = config.get("activation_function", "gelu_new")
        if activation != "gelu_new":
            raise NotImplementedError(f"activation {activation!r} is not supported yet")
        # Options that change the attention arithmetic; Hugging Face's defaults are supported.
        for option, default in [
            ("scale_attn_weights", True),
            ("scale_attn_by_inverse_layer_idx", False),
            ("reorder_and_upcast_attn", False),
            ("add_cross_attention", False),
        ]:
            if config.get(option, default) != default:
                raise NotImplementedError(f"{option}={config[option]} is not supported yet")
        if not config.get("tie_word_embeddings", True):
            raise NotImplementedError("an output head untied from the token embedding")
        n_embd, n_head = config["n_embd"], config["n_head"]
        if n_embd % n_head:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        return cls(
            vocab_size=config["vocab_size"],
            n_embd=n_embd,
            n_layer=config["n_layer"],
            n_head=n_head,
            n_positions=config["n_positions"],
            n_inner=config.get("n_inner") or 4 * n_embd,
            layer_norm_epsilon=config["layer_norm_epsilon"],
            pad_token_id=config.get("pad_token_id"),
        )
