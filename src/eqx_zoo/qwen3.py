"""Qwen3 dense causal language models.

Example:
    >>> import jax.numpy as jnp
    >>> from eqx_zoo import Qwen3ForCausalLM
    >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    >>> logits = model(jnp.array([785, 6722, 315, 9625, 374]))
"""

import dataclasses
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from eqx_zoo._loading import load_safetensors
from eqx_zoo.layers import Attention, RMSNorm, SwiGLU


@dataclasses.dataclass(frozen=True)
class Qwen3Config:
    """Architecture hyperparameters, named as in the Hugging Face `config.json`.

    Attributes:
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
    """

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

    @classmethod
    def from_hf(cls, config: dict) -> "Qwen3Config":
        """Build a config from a parsed Hugging Face `config.json`, ignoring extra keys.

        Args:
            config: The parsed JSON.

        Returns:
            The corresponding config.
        """
        return cls(**{field.name: config[field.name] for field in dataclasses.fields(cls)})


class Qwen3DecoderLayer(eqx.Module):
    """Pre-norm transformer block: attention then SwiGLU, each with a residual connection.

    Attributes:
        input_layernorm: Norm applied before attention.
        self_attn: Grouped-query attention with per-head query/key norm.
        post_attention_layernorm: Norm applied before the MLP (named as in Hugging Face).
        mlp: SwiGLU feed-forward block.
    """

    input_layernorm: RMSNorm
    self_attn: Attention
    post_attention_layernorm: RMSNorm
    mlp: SwiGLU

    def __init__(self, config: Qwen3Config, *, key: PRNGKeyArray, dtype: DTypeLike = jnp.float32):
        """Create a randomly initialised layer.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        attn_key, mlp_key = jax.random.split(key)
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = Attention(
            config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            eps=eps,
            key=attn_key,
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, key=mlp_key, dtype=dtype)

    def __call__(
        self, x: Float[Array, "seq dim"], positions: Int[Array, " seq"]
    ) -> Float[Array, "seq dim"]:
        """Apply the layer to a sequence.

        Args:
            x: Residual stream for each token.
            positions: Absolute position of each token.

        Returns:
            The updated residual stream.
        """
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + jax.vmap(self.mlp)(self.post_attention_layernorm(x))


class Qwen3Model(eqx.Module):
    """Qwen3 decoder stack: token embedding, decoder layers and a final norm.

    Attributes:
        embed_tokens: Token embedding table.
        layers: Decoder layers, applied in order.
        norm: Final RMSNorm.
    """

    embed_tokens: eqx.nn.Embedding
    layers: list[Qwen3DecoderLayer]
    norm: RMSNorm

    def __init__(self, config: Qwen3Config, *, key: PRNGKeyArray, dtype: DTypeLike = jnp.float32):
        """Create a randomly initialised decoder stack.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        embed_key, *layer_keys = jax.random.split(key, config.num_hidden_layers + 1)
        self.embed_tokens = eqx.nn.Embedding(
            config.vocab_size, config.hidden_size, key=embed_key, dtype=dtype
        )
        self.layers = [Qwen3DecoderLayer(config, key=k, dtype=dtype) for k in layer_keys]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def __call__(
        self, input_ids: Int[Array, " seq"], positions: Int[Array, " seq"]
    ) -> Float[Array, "seq dim"]:
        """Compute final hidden states.

        Args:
            input_ids: Token ids.
            positions: Absolute position of each token.

        Returns:
            Normalised hidden states for each token.
        """
        x = self.embed_tokens.weight[input_ids]
        for layer in self.layers:
            x = layer(x, positions)
        return self.norm(x)


class Qwen3ForCausalLM(eqx.Module):
    """Qwen3 language model with a vocabulary projection head.

    Attributes:
        model: The decoder stack.
        lm_head: Output projection, or `None` when tied to the token embedding.
        config: Model hyperparameters.
    """

    model: Qwen3Model
    lm_head: eqx.nn.Linear | None
    config: Qwen3Config = eqx.field(static=True)

    def __init__(self, config: Qwen3Config, *, key: PRNGKeyArray, dtype: DTypeLike = jnp.float32):
        """Create a randomly initialised model.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        model_key, head_key = jax.random.split(key)
        self.model = Qwen3Model(config, key=model_key, dtype=dtype)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = eqx.nn.Linear(
                config.hidden_size, config.vocab_size, use_bias=False, key=head_key, dtype=dtype
            )
        self.config = config

    def __call__(self, input_ids: Int[Array, " seq"]) -> Float[Array, "seq vocab"]:
        """Compute next-token logits for every position.

        Args:
            input_ids: Token ids, starting at position 0.

        Returns:
            Unnormalised next-token logits for each position.
        """
        hidden = self.model(input_ids, jnp.arange(input_ids.shape[0]))
        if self.lm_head is None:
            return hidden @ self.model.embed_tokens.weight.T
        return jax.vmap(self.lm_head)(hidden)

    @classmethod
    def from_pretrained(
        cls, repo_id: str, *, dtype: DTypeLike = jnp.float32, revision: str | None = None
    ) -> "Qwen3ForCausalLM":
        """Load a pretrained checkpoint from the Hugging Face Hub.

        Args:
            repo_id: Hub repository, e.g. `"Qwen/Qwen3-0.6B"`.
            dtype: Dtype to cast the parameters to.
            revision: Optional git revision (branch, tag or commit) of the repository.

        Returns:
            The model with pretrained weights.
        """
        path = Path(
            snapshot_download(
                repo_id, revision=revision, allow_patterns=["config.json", "*.safetensors"]
            )
        )
        config = Qwen3Config.from_hf(json.loads((path / "config.json").read_text()))
        skeleton = eqx.filter_eval_shape(cls, config, key=jax.random.key(0), dtype=dtype)
        ignore = ["lm_head.weight"] if config.tie_word_embeddings else []
        return load_safetensors(skeleton, sorted(path.glob("*.safetensors")), ignore=ignore)
