"""Decoder-only causal language models (Llama, Qwen2, Qwen3, Qwen3-MoE).

Example:
    >>> import jax.numpy as jnp
    >>> from eqx_zoo import CausalLM
    >>> model = CausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    >>> logits, _ = model(jnp.array([785, 6722, 315, 9625, 374]))
"""

import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from eqx_zoo._loading import load_safetensors
from eqx_zoo.layers import Attention, Cache, KVCache, RMSNorm, SparseMoE, SwiGLU, causal_mask
from eqx_zoo.models.causal_lm.config import Config


class DecoderLayer(eqx.Module):
    """Pre-norm transformer block: attention then SwiGLU, each with a residual connection.

    Attributes:
        input_layernorm: Norm applied before attention.
        self_attn: Grouped-query attention.
        post_attention_layernorm: Norm applied before the MLP (named as in Hugging Face).
        mlp: Feed-forward block: a SwiGLU, or a sparse mixture of SwiGLU experts.
    """

    input_layernorm: RMSNorm
    self_attn: Attention
    post_attention_layernorm: RMSNorm
    mlp: SwiGLU | SparseMoE

    def __init__(
        self,
        config: Config,
        *,
        moe: bool = False,
        key: PRNGKeyArray,
        dtype: DTypeLike = jnp.float32,
    ):
        """Create a randomly initialised layer.

        Args:
            config: Model hyperparameters.
            moe: Whether the feed-forward block is a mixture of experts.
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
            rope_scaling=config.rope_scaling,
            eps=eps,
            qkv_bias=config.attention_bias,
            qk_norm=config.qk_norm,
            key=attn_key,
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps, dtype=dtype)
        if moe:
            self.mlp = SparseMoE(
                config.hidden_size,
                config.moe_intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
                key=mlp_key,
                dtype=dtype,
            )
        else:
            self.mlp = SwiGLU(
                config.hidden_size, config.intermediate_size, key=mlp_key, dtype=dtype
            )

    def __call__(
        self,
        x: Float[Array, "seq dim"],
        positions: Int[Array, " seq"],
        mask: Bool[Array, "seq keys"],
        cache: KVCache | None = None,
        cache_index: Int[Array, ""] | None = None,
    ) -> tuple[Float[Array, "seq dim"], KVCache | None]:
        """Apply the layer to a sequence.

        Args:
            x: Residual stream for each token.
            positions: Position of each token, used for the rotary embedding.
            mask: Attention mask; see `Attention`.
            cache: This layer's key/value cache, or `None`.
            cache_index: Cache slot for the first token; required with a cache.

        Returns:
            The updated residual stream, and the updated cache (or `None`).
        """
        attn_out, cache = self.self_attn(
            self.input_layernorm(x), positions, mask, cache, cache_index
        )
        x = x + attn_out
        h = self.post_attention_layernorm(x)
        # A SwiGLU acts on one token, so it is vmapped; the MoE routes the whole sequence.
        out = self.mlp(h) if isinstance(self.mlp, SparseMoE) else jax.vmap(self.mlp)(h)
        return x + out, cache


class DecoderModel(eqx.Module):
    """Decoder stack: token embedding, decoder layers and a final norm.

    Attributes:
        embed_tokens: Token embedding table.
        layers: Decoder layers, applied in order.
        norm: Final RMSNorm.
    """

    embed_tokens: eqx.nn.Embedding
    layers: list[DecoderLayer]
    norm: RMSNorm

    def __init__(self, config: Config, *, key: PRNGKeyArray, dtype: DTypeLike = jnp.float32):
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
        self.layers = [
            DecoderLayer(config, moe=i in config.moe_layers, key=k, dtype=dtype)
            for i, k in enumerate(layer_keys)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def __call__(
        self,
        input_ids: Int[Array, " seq"],
        positions: Int[Array, " seq"],
        mask: Bool[Array, "seq keys"],
        caches: list[KVCache] | None = None,
        cache_index: Int[Array, ""] | None = None,
    ) -> tuple[Float[Array, "seq dim"], list[KVCache] | None]:
        """Compute final hidden states.

        Args:
            input_ids: Token ids.
            positions: Position of each token, used for the rotary embedding.
            mask: Attention mask; see `Attention`.
            caches: One key/value cache per layer, or `None`.
            cache_index: Cache slot for the first token; required with caches.

        Returns:
            Normalised hidden states for each token, and the updated caches (or `None`).
        """
        x = self.embed_tokens.weight[input_ids]
        if caches is None:
            for layer in self.layers:
                x, _ = layer(x, positions, mask)
            return self.norm(x), None

        new_caches = []
        for layer, cache in zip(self.layers, caches, strict=True):
            x, cache = layer(x, positions, mask, cache, cache_index)
            new_caches.append(cache)
        return self.norm(x), new_caches


class CausalLM(eqx.Module):
    """Decoder-only language model with a vocabulary projection head.

    Attributes:
        model: The decoder stack.
        lm_head: Output projection, or `None` when tied to the token embedding.
        config: Model hyperparameters.
    """

    model: DecoderModel
    lm_head: eqx.nn.Linear | None
    config: Config = eqx.field(static=True)

    def __init__(self, config: Config, *, key: PRNGKeyArray, dtype: DTypeLike = jnp.float32):
        """Create a randomly initialised model.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        model_key, head_key = jax.random.split(key)
        self.model = DecoderModel(config, key=model_key, dtype=dtype)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = eqx.nn.Linear(
                config.hidden_size, config.vocab_size, use_bias=False, key=head_key, dtype=dtype
            )
        self.config = config

    def __call__(
        self,
        input_ids: Int[Array, " seq"],
        cache: Cache | None = None,
        attention_mask: Bool[Array, " seq"] | Int[Array, " seq"] | None = None,
    ) -> tuple[Float[Array, "seq vocab"], Cache | None]:
        """Compute next-token logits, optionally continuing from a cache.

        Args:
            input_ids: Token ids.
            cache: State from earlier calls, created with `init_cache`, or `None`.
            attention_mask: `1` or `True` for real tokens and `0` or `False` for padding, as
                produced by tokenizers. Padding is ignored by attention and does not advance
                positions; the logits at padding tokens are unspecified. Defaults to all real.

        Returns:
            Unnormalised next-token logits for each position, and the updated cache (or
            `None` if no cache was given).
        """
        seq = input_ids.shape[0]
        if attention_mask is None:
            valid = jnp.ones(seq, dtype=bool)
        else:
            valid = attention_mask.astype(bool)

        if cache is None:
            slots = jnp.arange(seq)
            positions = jnp.maximum(jnp.cumsum(valid) - 1, 0)
            mask = causal_mask(slots, slots, valid)
            hidden, layer_caches = self.model(input_ids, positions, mask)
        else:
            slots = cache.length + jnp.arange(seq)
            positions = jnp.maximum(cache.valid.sum() + jnp.cumsum(valid) - 1, 0)
            key_valid = jax.lax.dynamic_update_slice(cache.valid, valid, (cache.length,))
            mask = causal_mask(slots, jnp.arange(key_valid.shape[0]), key_valid)
            hidden, layer_caches = self.model(
                input_ids, positions, mask, cache.layers, cache_index=cache.length
            )

        if self.lm_head is None:
            logits = hidden @ self.model.embed_tokens.weight.T
        else:
            logits = jax.vmap(self.lm_head)(hidden)

        if cache is not None:
            cache = Cache(layers=layer_caches, length=cache.length + seq, valid=key_valid)
        return logits, cache

    def init_cache(self, max_len: int) -> Cache:
        """Allocate an empty key/value cache.

        Args:
            max_len: Maximum total number of tokens (prompt plus generated) it can hold.

        Returns:
            An empty cache in the model's parameter dtype.
        """
        c = self.config
        dtype = self.model.embed_tokens.weight.dtype
        empty = KVCache.empty(max_len, c.num_key_value_heads, c.head_dim, dtype)
        return Cache(
            layers=[empty] * c.num_hidden_layers,
            length=jnp.array(0, dtype=jnp.int32),
            valid=jnp.zeros(max_len, dtype=bool),
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | os.PathLike[str],
        *,
        dtype: DTypeLike = jnp.float32,
        revision: str | None = None,
    ) -> "CausalLM":
        """Load a pretrained checkpoint from the Hugging Face Hub or a local directory.

        The architecture is read from the checkpoint's `config.json`; see `Config.from_hf`
        for the supported architectures.

        Args:
            repo_id: Hub repository, e.g. `"Qwen/Qwen3-0.6B"`, or a local directory
                containing `config.json` and safetensors files. An existing directory takes
                precedence over a Hub repository of the same name.
            dtype: Dtype to cast the parameters to.
            revision: Optional git revision of a Hub repository; ignored for directories.

        Returns:
            The model with pretrained weights.
        """
        if Path(repo_id).is_dir():
            path = Path(repo_id)
        else:
            path = Path(
                snapshot_download(
                    str(repo_id),
                    revision=revision,
                    allow_patterns=["config.json", "*.safetensors"],
                )
            )
        config = Config.from_hf(json.loads((path / "config.json").read_text()))
        skeleton = eqx.filter_eval_shape(cls, config, key=jax.random.key(0), dtype=dtype)
        ignore = ["lm_head.weight"] if config.tie_word_embeddings else []
        stacked = [f"model.layers.{i}.mlp.experts" for i in config.moe_layers]
        return load_safetensors(
            skeleton, sorted(path.glob("*.safetensors")), ignore=ignore, stacked=stacked
        )
