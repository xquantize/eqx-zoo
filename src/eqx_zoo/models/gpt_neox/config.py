"""GPT-NeoX configuration, translated from Hugging Face `config.json` files."""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class GPTNeoXConfig:
    """Hyperparameters of a GPT-NeoX language model.

    Field names follow Hugging Face's GPT-NeoX configuration. Build one from a checkpoint's
    `config.json` with `GPTNeoXConfig.from_hf`.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        hidden_size: Model (residual stream) dimension.
        intermediate_size: Inner dimension of each MLP.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        max_position_embeddings: Context length the model was trained for. Rotary embeddings
            have no learned positions, so this is informational and isn't enforced.
        layer_norm_eps: Epsilon of every LayerNorm.
        rotary_dim: Leading dimensions of each head that the rotary embedding rotates:
            `int(head_size * partial_rotary_factor)`.
        rope_theta: Base frequency of the rotary embedding.
        use_parallel_residual: Whether attention and the MLP both read the layer's input and
            are added to it together (`x + attn(norm1(x)) + mlp(norm2(x))`), rather than one
            after the other.
        pad_token_id: Id of the padding token, or `None`.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    layer_norm_eps: float
    rotary_dim: int
    rope_theta: float
    use_parallel_residual: bool
    pad_token_id: int | None = None

    @property
    def head_size(self) -> int:
        """Per-head dimension: `hidden_size // num_attention_heads`."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf(cls, config: dict[str, Any]) -> "GPTNeoXConfig":
        """Translate a parsed Hugging Face `config.json`.

        Rotary settings come from `rope_parameters` (Transformers 5) or `rope_scaling` (its
        legacy name), where either is present; otherwise from the original `rotary_pct` and
        `rotary_emb_base`. Where the dictionary exists, it wins key by key, and the legacy
        keys only fill what it lacks, as in Transformers' `convert_rope_params_to_dict`.
        Unknown keys, such as dropout rates and training options, are ignored.

        Args:
            config: The parsed JSON.

        Returns:
            The corresponding config.

        Raises:
            ValueError: If the architecture is not GPT-NeoX, the hidden size isn't divisible
                by the number of heads, or the rotary dimension is not a positive even
                number within the head size.
            NotImplementedError: If the checkpoint needs a feature not implemented yet.
        """
        architecture = (config.get("architectures") or ["<missing>"])[0]
        if architecture != "GPTNeoXForCausalLM":
            raise ValueError(f"expected GPTNeoXForCausalLM, not {architecture!r}")
        activation = config.get("hidden_act", "gelu")
        if activation != "gelu":
            raise NotImplementedError(f"activation {activation!r} is not supported yet")
        if not config.get("attention_bias", True):
            raise NotImplementedError("attention without biases is not supported yet")
        if config.get("tie_word_embeddings", False):
            raise NotImplementedError("an output head tied to the token embedding")

        hidden_size, heads = config["hidden_size"], config["num_attention_heads"]
        if hidden_size % heads:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({heads})"
            )
        head_size = hidden_size // heads
        # Attention always uses `hidden_size // num_attention_heads`, but Transformers' rotary
        # embedding reads a `head_dim` attribute if there is one, so the two could disagree.
        if config.get("head_dim") not in (None, head_size):
            raise NotImplementedError(f"head_dim={config['head_dim']} differs from the head size")

        rope = config.get("rope_scaling") or config.get("rope_parameters") or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise NotImplementedError(f"RoPE type {rope_type!r} is not supported yet")
        factor = rope.get("partial_rotary_factor", config.get("rotary_pct", 0.25))
        rotary_dim = int(head_size * factor)
        if not 0 < rotary_dim <= head_size or rotary_dim % 2:
            raise ValueError(
                f"rotary dimension {rotary_dim} (head size {head_size}, partial rotary "
                f"factor {factor}) must be a positive even number within the head size"
            )

        return cls(
            vocab_size=config["vocab_size"],
            hidden_size=hidden_size,
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=heads,
            max_position_embeddings=config.get("max_position_embeddings", 2048),
            layer_norm_eps=config.get("layer_norm_eps", 1e-5),
            rotary_dim=rotary_dim,
            rope_theta=float(rope.get("rope_theta", config.get("rotary_emb_base", 10_000.0))),
            use_parallel_residual=config.get("use_parallel_residual", True),
            pad_token_id=config.get("pad_token_id"),
        )
