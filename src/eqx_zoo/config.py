"""Model configuration, translated from Hugging Face `config.json` files."""

import dataclasses
from typing import Any

# Attention features implied by each architecture rather than stated in `config.json`.
_ARCHITECTURES: dict[str, dict[str, bool]] = {
    "Qwen2ForCausalLM": {"attention_bias": True, "qk_norm": False},
    "Qwen3ForCausalLM": {"attention_bias": False, "qk_norm": True},
}


@dataclasses.dataclass(frozen=True)
class Config:
    """Hyperparameters of a decoder-only language model.

    Field names follow Hugging Face conventions. Build one from a checkpoint's
    `config.json` with `Config.from_hf`.

    Attributes:
        architecture: Hugging Face architecture name, e.g. `"Qwen3ForCausalLM"`.
        vocab_size: Number of tokens in the vocabulary.
        hidden_size: Model (residual stream) dimension.
        intermediate_size: Inner dimension of each MLP.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads.
        head_dim: Per-head dimension.
        rms_norm_eps: Epsilon for every RMSNorm.
        rope_theta: RoPE base frequency.
        tie_word_embeddings: Whether the output head reuses the embedding matrix.
        attention_bias: Whether the query, key and value projections have a bias.
        qk_norm: Whether each query and key head is RMS-normalised.
    """

    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    attention_bias: bool
    qk_norm: bool

    @classmethod
    def from_hf(cls, config: dict[str, Any]) -> "Config":
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
        features = _ARCHITECTURES[architecture]

        # RoPE settings live in `rope_scaling` (transformers 4.x) or `rope_parameters` (5.x).
        rope = config.get("rope_scaling") or config.get("rope_parameters") or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise NotImplementedError(f"RoPE scaling type {rope_type!r} is not supported yet")
        if config.get("use_sliding_window", False):
            raise NotImplementedError("sliding-window attention is not supported yet")
        if config.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"activation {config['hidden_act']!r} is not supported")

        num_heads = config["num_attention_heads"]
        return cls(
            architecture=architecture,
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=num_heads,
            num_key_value_heads=config.get("num_key_value_heads", num_heads),
            head_dim=config.get("head_dim") or config["hidden_size"] // num_heads,
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config.get("rope_theta", rope.get("rope_theta", 10_000.0)),
            tie_word_embeddings=config.get("tie_word_embeddings", False),
            attention_bias=config.get("attention_bias", features["attention_bias"]),
            qk_norm=features["qk_norm"],
        )
