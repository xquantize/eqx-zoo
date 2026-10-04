"""Encoder configuration, translated from Hugging Face `config.json` files."""

import dataclasses
from typing import Any

_ARCHITECTURES: dict[str, dict[str, Any]] = {
    "BertModel": {"padding_aware_positions": False},
    "RobertaModel": {"padding_aware_positions": True},
    "XLMRobertaModel": {"padding_aware_positions": True},
}


@dataclasses.dataclass(frozen=True)
class EncoderConfig:
    """Hyperparameters of a BERT-style bidirectional encoder.

    Field names follow Hugging Face conventions. Build one from a checkpoint's
    `config.json` with `EncoderConfig.from_hf`.

    Attributes:
        architecture: Hugging Face architecture name, e.g. `"BertModel"`.
        vocab_size: Number of tokens in the vocabulary.
        hidden_size: Model dimension.
        num_hidden_layers: Number of encoder layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Inner dimension of each MLP.
        max_position_embeddings: Number of learned position embeddings.
        type_vocab_size: Number of token types (segments).
        layer_norm_eps: Epsilon for every LayerNorm.
        pad_token_id: Id of the padding token.
        padding_aware_positions: Whether position ids count only non-padding tokens, starting
            after `pad_token_id`, as in RoBERTa; otherwise positions are 0, 1, 2, ...
    """

    architecture: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    type_vocab_size: int
    layer_norm_eps: float
    layer_norm_eps: float
    pad_token_id: int = 0
    padding_aware_positions: bool = False

    @classmethod
    def from_hf(cls, config: dict[str, Any]) -> "EncoderConfig":
        """Translate a parsed Hugging Face `config.json`.

        Args:
            config: The parsed JSON.

        Returns:
            The corresponding config.

        Raises:
            ValueError: If the architecture is not supported.
            NotImplementedError: If the checkpoint needs a feature not implemented yet.
        """
        architecture = (config.get("architectures") or ["<missing>"])[0]
        if architecture not in _ARCHITECTURES:
            supported = ", ".join(sorted(_ARCHITECTURES))
            raise ValueError(f"unsupported architecture {architecture!r}; supported: {supported}")
        if config.get("hidden_act", "gelu") != "gelu":
            raise NotImplementedError(f"activation {config['hidden_act']!r} is not supported")
        position_type = config.get("position_embedding_type", "absolute")
        if position_type != "absolute":
            raise NotImplementedError(f"position embeddings {position_type!r} are not supported")

        return cls(
            architecture=architecture,
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            type_vocab_size=config.get("type_vocab_size", 2),
            layer_norm_eps=config.get("layer_norm_eps", 1e-12),
            pad_token_id=config.get("pad_token_id", 0),
            padding_aware_positions=_ARCHITECTURES[architecture]["padding_aware_positions"],
        )
