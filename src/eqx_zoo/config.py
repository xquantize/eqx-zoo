"""Model configuration, translated from Hugging Face `config.json` files."""

import dataclasses
from typing import Any

from eqx_zoo.layers import Llama3RopeScaling

# Attention features implied by each architecture rather than stated in `config.json`.
_ARCHITECTURES: dict[str, dict[str, bool]] = {
    "LlamaForCausalLM": {"attention_bias": False, "qk_norm": False},
    "Qwen2ForCausalLM": {"attention_bias": True, "qk_norm": False},
    "Qwen3ForCausalLM": {"attention_bias": False, "qk_norm": True},
    "Qwen3MoeForCausalLM": {"attention_bias": False, "qk_norm": True},
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
        rope_scaling: RoPE frequency scaling, or None for standard RoPE.
        num_experts: Number of experts in each mixture-of-experts layer; 0 for dense models.
        num_experts_per_tok: Number of experts each token is routed to.
        moe_intermediate_size: Inner dimension of each expert's MLP.
        norm_topk_prob: Whether the selected experts' routing weights are renormalised to
            sum to one.
        moe_layers: Indices of the decoder layers that are mixture-of-experts; the others
            use a dense MLP.
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
    rope_scaling: Llama3RopeScaling | None
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = False
    moe_layers: tuple[int, ...] = ()

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
        if rope_type == "default":
            rope_scaling = None
        elif rope_type == "llama3":
            rope_scaling = Llama3RopeScaling(
                factor=rope["factor"],
                low_freq_factor=rope["low_freq_factor"],
                high_freq_factor=rope["high_freq_factor"],
                original_max_position_embeddings=rope["original_max_position_embeddings"],
            )
        else:
            raise NotImplementedError(f"RoPE scaling type {rope_type!r} is not supported yet")
        if config.get("mlp_bias", False):
            raise NotImplementedError("MLP bias is not supported yet")
        if config.get("use_sliding_window", False):
            raise NotImplementedError("sliding-window attention is not supported yet")
        if config.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"activation {config['hidden_act']!r} is not supported")

        num_heads = config["num_attention_heads"]
        num_layers = config["num_hidden_layers"]
        num_experts = config.get("num_experts", 0)
        if num_experts:
            # A layer is mixture-of-experts unless listed in `mlp_only_layers`, and only
            # every `decoder_sparse_step`-th layer is; as in the Hugging Face implementation.
            step = config.get("decoder_sparse_step", 1)
            dense = set(config.get("mlp_only_layers", []))
            moe_layers = tuple(
                i for i in range(num_layers) if i not in dense and (i + 1) % step == 0
            )
        else:
            moe_layers = ()
        return cls(
            architecture=architecture,
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=config.get("num_key_value_heads", num_heads),
            head_dim=config.get("head_dim") or config["hidden_size"] // num_heads,
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config.get("rope_theta", rope.get("rope_theta", 10_000.0)),
            tie_word_embeddings=config.get("tie_word_embeddings", False),
            attention_bias=config.get("attention_bias", features["attention_bias"]),
            qk_norm=features["qk_norm"],
            rope_scaling=rope_scaling,
            num_experts=num_experts,
            num_experts_per_tok=config.get("num_experts_per_tok", 0),
            moe_intermediate_size=config.get("moe_intermediate_size", 0),
            norm_topk_prob=config.get("norm_topk_prob", False),
            moe_layers=moe_layers,
        )
