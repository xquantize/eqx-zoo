"""GPT-NeoX language models: partial rotary attention, LayerNorm and a parallel residual.

Example:
    >>> model = GPTNeoX.from_pretrained("EleutherAI/pythia-160m")
    >>> logits, _ = model(jnp.array([510, 5347, 273, 6181, 310]))
"""

import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from eqx_zoo._loading import config_with_dtype, load_safetensors, save_safetensors
from eqx_zoo.layers import (
    Cache,
    KVCache,
    LayerNorm,
    apply_rope,
    causal_mask,
    dot_product_attention,
    rope_cos_sin,
)
from eqx_zoo.layers._common import linear
from eqx_zoo.models.gpt_neox.config import GPTNeoXConfig


class GPTNeoXAttention(eqx.Module):
    """Causal multi-head attention with a fused projection and partial rotary embeddings.

    The fused projection is interleaved per head: each head's output is its query, key and
    value in turn, `[q0, k0, v0, q1, k1, v1, ...]`, unlike GPT-2's `[all q, all k, all v]`.
    Only the first `rotary_dim` dimensions of each query and key head are rotated.

    Attributes:
        query_key_value: Fused query, key and value projection, with a bias.
        dense: Output projection, with a bias.
        num_heads: Number of attention heads.
        rotary_dim: Leading dimensions of each head that the rotary embedding rotates.
        rope_theta: Base frequency of the rotary embedding.
    """

    query_key_value: eqx.nn.Linear
    dense: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    rotary_dim: int = eqx.field(static=True)
    rope_theta: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rotary_dim: int,
        rope_theta: float,
        key: PRNGKeyArray,
        dtype: DTypeLike,
    ):
        """Create a randomly initialised layer.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            rotary_dim: Leading dimensions of each head to rotate.
            rope_theta: Base frequency of the rotary embedding.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        qkv_key, dense_key = jax.random.split(key)
        self.query_key_value = linear(dim, 3 * dim, key=qkv_key, dtype=dtype, bias=True)
        self.dense = linear(dim, dim, key=dense_key, dtype=dtype, bias=True)
        self.num_heads = num_heads
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta

    def __call__(
        self,
        x: Float[Array, "seq dim"],
        positions: Int[Array, " seq"],
        mask: Bool[Array, "seq keys"],
        cache: KVCache | None = None,
        cache_index: Int[Array, ""] | None = None,
    ) -> tuple[Float[Array, "seq dim"], KVCache | None]:
        """Attend over a sequence, optionally continuing from a cache.

        Args:
            x: Hidden state of each new token.
            positions: Position of each new token, used for the rotary embedding.
            mask: `True` where a new token may attend to a key.
            cache: Keys and values of earlier tokens, or `None`.
            cache_index: Cache slot for the first new token; required with a cache.

        Returns:
            The attention output, and the updated cache (or `None`).

        Raises:
            ValueError: If a cache is given without `cache_index`.
        """
        seq, dim = x.shape
        head_size = dim // self.num_heads
        # Interleaved per head: (seq, heads, [q, k, v]), as in the reference's `view`.
        qkv = jax.vmap(self.query_key_value)(x).reshape(seq, self.num_heads, 3 * head_size)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        r = self.rotary_dim
        cos, sin = rope_cos_sin(positions, r, self.rope_theta)
        q = jnp.concatenate([apply_rope(q[..., :r], cos, sin), q[..., r:]], axis=-1)
        k = jnp.concatenate([apply_rope(k[..., :r], cos, sin), k[..., r:]], axis=-1)

        if cache is not None:
            if cache_index is None:
                raise ValueError("cache_index is required when a cache is given")
            cache = cache.update(k, v, cache_index)
            k, v = cache.k, cache.v
        out = dot_product_attention(q, k, v, mask)
        return jax.vmap(self.dense)(out.reshape(seq, dim)), cache


class GPTNeoXMLP(eqx.Module):
    """Feed-forward block: `dense_h_to_4h`, exact GELU, then `dense_4h_to_h`.

    Attributes:
        dense_h_to_4h: Expansion, with a bias.
        dense_4h_to_h: Projection back to the model dimension, with a bias.
    """

    dense_h_to_4h: eqx.nn.Linear
    dense_4h_to_h: eqx.nn.Linear

    def __init__(self, dim: int, inner: int, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised block.

        Args:
            dim: Model dimension.
            inner: Inner dimension.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        up_key, down_key = jax.random.split(key)
        self.dense_h_to_4h = linear(dim, inner, key=up_key, dtype=dtype, bias=True)
        self.dense_4h_to_h = linear(inner, dim, key=down_key, dtype=dtype, bias=True)

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        """Apply the block to a single token.

        Args:
            x: One token's hidden state.

        Returns:
            The block's output.
        """
        return self.dense_4h_to_h(jax.nn.gelu(self.dense_h_to_4h(x), approximate=False))


class GPTNeoXLayer(eqx.Module):
    """Transformer layer with a parallel or sequential residual connection.

    Parallel: `x + attention(input_layernorm(x)) + mlp(post_attention_layernorm(x))`.
    Sequential: `h = x + attention(input_layernorm(x))`, then
    `h + mlp(post_attention_layernorm(h))`.

    Attributes:
        input_layernorm: Norm before attention.
        post_attention_layernorm: Norm before the MLP (named as in Hugging Face).
        attention: Attention.
        mlp: Feed-forward block.
        use_parallel_residual: Whether attention and the MLP both read the layer's input.
    """

    input_layernorm: LayerNorm
    post_attention_layernorm: LayerNorm
    attention: GPTNeoXAttention
    mlp: GPTNeoXMLP
    use_parallel_residual: bool = eqx.field(static=True)

    def __init__(self, config: GPTNeoXConfig, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised layer.

        Args:
            config: Model hyperparameters.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        attn_key, mlp_key = jax.random.split(key)
        dim, eps = config.hidden_size, config.layer_norm_eps
        self.input_layernorm = LayerNorm(dim, eps=eps, dtype=dtype)
        self.post_attention_layernorm = LayerNorm(dim, eps=eps, dtype=dtype)
        self.attention = GPTNeoXAttention(
            dim,
            config.num_attention_heads,
            rotary_dim=config.rotary_dim,
            rope_theta=config.rope_theta,
            key=attn_key,
            dtype=dtype,
        )
        self.mlp = GPTNeoXMLP(dim, config.intermediate_size, key=mlp_key, dtype=dtype)
        self.use_parallel_residual = config.use_parallel_residual

    def __call__(
        self,
        x: Float[Array, "seq dim"],
        positions: Int[Array, " seq"],
        mask: Bool[Array, "seq keys"],
        cache: KVCache | None = None,
        cache_index: Int[Array, ""] | None = None,
    ) -> tuple[Float[Array, "seq dim"], KVCache | None]:
        """Apply the layer.

        Args:
            x: Residual stream for each token.
            positions: Position of each token, used for the rotary embedding.
            mask: Attention mask; see `GPTNeoXAttention`.
            cache: This layer's key/value cache, or `None`.
            cache_index: Cache slot for the first token; required with a cache.

        Returns:
            The updated residual stream, and the updated cache (or `None`).
        """
        attn, cache = self.attention(self.input_layernorm(x), positions, mask, cache, cache_index)
        if self.use_parallel_residual:
            mlp = jax.vmap(self.mlp)(self.post_attention_layernorm(x))
            return x + attn + mlp, cache
        x = x + attn
        return x + jax.vmap(self.mlp)(self.post_attention_layernorm(x)), cache


class GPTNeoXModel(eqx.Module):
    """The decoder stack: token embedding, layers and a final norm.

    Attributes:
        embed_in: Token embedding table.
        layers: The transformer layers, applied in order.
        final_layer_norm: Final norm.
    """

    embed_in: eqx.nn.Embedding
    layers: list[GPTNeoXLayer]
    final_layer_norm: LayerNorm

    def __init__(self, config: GPTNeoXConfig, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised decoder stack.

        Args:
            config: Model hyperparameters.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        embed_key, *layer_keys = jax.random.split(key, config.num_hidden_layers + 1)
        self.embed_in = eqx.nn.Embedding(
            config.vocab_size, config.hidden_size, key=embed_key, dtype=dtype
        )
        self.layers = [GPTNeoXLayer(config, key=k, dtype=dtype) for k in layer_keys]
        self.final_layer_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )

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
            mask: Attention mask; see `GPTNeoXAttention`.
            caches: One key/value cache per layer, or `None`.
            cache_index: Cache slot for the first token; required with caches.

        Returns:
            Normalised hidden states for each token, and the updated caches (or `None`).
        """
        x = self.embed_in.weight[input_ids]
        if caches is None:
            for layer in self.layers:
                x, _ = layer(x, positions, mask)
            return self.final_layer_norm(x), None
        new_caches = []
        for layer, cache in zip(self.layers, caches, strict=True):
            x, cache = layer(x, positions, mask, cache, cache_index)
            new_caches.append(cache)
        return self.final_layer_norm(x), new_caches


class GPTNeoX(eqx.Module):
    """GPT-NeoX language model (Pythia), with the same interface as `CausalLM`.

    Example:
        >>> model = GPTNeoX.from_pretrained("EleutherAI/pythia-160m")
        >>> logits, _ = model(jnp.array([510, 5347, 273, 6181, 310]))

    Attributes:
        gpt_neox: The decoder stack.
        embed_out: Output head, untied from the token embedding and without a bias.
        config: Model hyperparameters.
        hf_config: The checkpoint's `config.json` text, kept for `save_pretrained`, or `None`.
    """

    gpt_neox: GPTNeoXModel
    embed_out: eqx.nn.Linear
    config: GPTNeoXConfig = eqx.field(static=True)
    hf_config: str | None = eqx.field(static=True)

    def __init__(
        self,
        config: GPTNeoXConfig,
        *,
        hf_config: str | None = None,
        key: PRNGKeyArray,
        dtype: DTypeLike = jnp.float32,
    ):
        """Create a randomly initialised model.

        Args:
            config: Model hyperparameters.
            hf_config: The checkpoint's config.json text, or None.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        model_key, head_key = jax.random.split(key)
        self.gpt_neox = GPTNeoXModel(config, key=model_key, dtype=dtype)
        self.embed_out = linear(
            config.hidden_size, config.vocab_size, key=head_key, dtype=dtype, bias=False
        )
        self.config = config
        self.hf_config = hf_config

    def __call__(
        self,
        input_ids: Int[Array, " seq"],
        cache: Cache | None = None,
        attention_mask: Bool[Array, " seq"] | Int[Array, " seq"] | None = None,
        *,
        last_only: bool = False,
    ) -> tuple[Float[Array, "seq vocab"], Cache | None]:
        """Compute next-token logits, optionally continuing from a cache.

        Args:
            input_ids: Token ids of the new tokens.
            cache: Cache from `init_cache` or an earlier call, or `None`.
            attention_mask: `1` or `True` for real tokens and `0` or `False` for left padding,
                which doesn't advance positions. Defaults to all real.
            last_only: Compute logits only for the last position, as generation needs.

        Returns:
            Logits for each position (only the last, with `last_only`), and the updated
            cache, or `None` without one.
        """
        seq = input_ids.shape[0]
        valid = jnp.ones(seq, dtype=bool) if attention_mask is None else attention_mask.astype(bool)
        if cache is None:
            slots = jnp.arange(seq)
            positions = jnp.maximum(jnp.cumsum(valid) - 1, 0)
            mask = causal_mask(slots, slots, valid)
            hidden, layer_caches = self.gpt_neox(input_ids, positions, mask)
        else:
            slots = cache.length + jnp.arange(seq)
            positions = jnp.maximum(cache.valid.sum() + jnp.cumsum(valid) - 1, 0)
            key_valid = jax.lax.dynamic_update_slice(cache.valid, valid, (cache.length,))
            mask = causal_mask(slots, jnp.arange(key_valid.shape[0]), key_valid)
            hidden, layer_caches = self.gpt_neox(
                input_ids, positions, mask, cache.layers, cache_index=cache.length
            )
        if last_only:
            hidden = hidden[-1:]
        logits = jax.vmap(self.embed_out)(hidden)
        if cache is not None:
            cache = Cache(layers=layer_caches, length=cache.length + seq, valid=key_valid)
        return logits, cache

    def init_cache(self, max_len: int) -> Cache:
        """Allocate an empty key/value cache.

        Args:
            max_len: Number of token slots: the prompt plus the tokens to generate.

        Returns:
            An empty cache in the model's parameter dtype.
        """
        c = self.config
        dtype = self.gpt_neox.embed_in.weight.dtype
        return Cache(
            layers=[
                KVCache.empty(max_len, c.num_attention_heads, c.head_size, dtype)
                for _ in range(c.num_hidden_layers)
            ],
            length=jnp.array(0, dtype=jnp.int32),
            valid=jnp.zeros(max_len, dtype=bool),
        )

    @classmethod
    def from_pretrained(
        cls, repo_or_path: str | os.PathLike[str], dtype: DTypeLike = jnp.float32
    ) -> "GPTNeoX":
        """Load a Hugging Face GPT-NeoX checkpoint.

        Args:
            repo_or_path: Hub repository id, or a local checkpoint directory.
            dtype: Parameter dtype.

        Returns:
            The loaded model.
        """
        path = Path(repo_or_path)
        if not path.is_dir():
            path = Path(
                snapshot_download(
                    str(repo_or_path), allow_patterns=["config.json", "*.safetensors"]
                )
            )
        hf_config = (path / "config.json").read_text()
        config = GPTNeoXConfig.from_hf(json.loads(hf_config))
        skeleton = eqx.filter_eval_shape(
            cls, config, hf_config=hf_config, key=jax.random.key(0), dtype=dtype
        )
        # Buffers older checkpoints save in every layer: the causal mask, its fill value and the
        # rotary frequencies. Transformers doesn't load them, and we compute them.
        ignore = [
            f"gpt_neox.layers.{i}.attention.{buffer}"
            for i in range(config.num_hidden_layers)
            for buffer in ("bias", "masked_bias", "rotary_emb.inv_freq")
        ]
        return load_safetensors(skeleton, sorted(path.glob("*.safetensors")), ignore=ignore)

    def save_pretrained(self, directory: str | os.PathLike[str]) -> None:
        """Save in Hugging Face format, as `config.json` and `model.safetensors`.

        Writes back the `config.json` the model was loaded with (its dtype set to the saved
        parameters'), and the weights named as the checkpoints name them.

        Args:
            directory: Destination directory; created if needed.

        Raises:
            ValueError: If the model wasn't loaded with `from_pretrained`.
        """
        if self.hf_config is None:
            raise ValueError(
                "only models loaded with from_pretrained can be saved: there is no original "
                "config.json to write back"
            )
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            config_with_dtype(self.hf_config, self.gpt_neox.embed_in.weight.dtype)
        )
        save_safetensors(self, path / "model.safetensors")
