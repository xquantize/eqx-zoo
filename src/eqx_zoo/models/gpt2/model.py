"""GPT-2 language models: learned positions, LayerNorm, and fused `Conv1D` projections."""

import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray
from safetensors import safe_open

from eqx_zoo._loading import (
    config_with_dtype,
    load_safetensors,
    refuse_adapters,
    save_safetensors,
)
from eqx_zoo.layers import Cache, KVCache, LayerNorm, causal_mask, dot_product_attention
from eqx_zoo.layers._common import at_least_float32
from eqx_zoo.models.gpt2.config import GPT2Config


class Conv1D(eqx.Module):
    """A linear layer with its weight stored as (in, out), as in GPT-2's checkpoints.

    Hugging Face calls it `Conv1D` for historical reasons; it's a plain affine map, with its
    weight transposed compared with `equinox.nn.Linear`.

    Attributes:
        weight: Weight, shaped (in_features, out_features).
        bias: Bias.
    """

    weight: Float[Array, "in_features out_features"]
    bias: Float[Array, " out_features"]

    def __init__(self, in_features: int, out_features: int, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised layer.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        self.weight = 0.02 * jax.random.normal(key, (in_features, out_features), dtype=dtype)
        self.bias = jnp.zeros(out_features, dtype=dtype)

    def __call__(
        self, x: Float[Array, "*batch in_features"]
    ) -> Float[Array, "*batch out_features"]:
        """Apply the layer.

        Args:
            x: Input.

        Returns:
            `x @ weight + bias`.
        """
        return x @ self.weight + self.bias


class GPT2Attention(eqx.Module):
    """Causal multi-head attention with a fused query/key/value projection.

    Attributes:
        c_attn: Fused query, key and value projection.
        c_proj: Output projection.
        num_heads: Number of attention heads.
    """

    c_attn: Conv1D
    c_proj: Conv1D
    num_heads: int = eqx.field(static=True)

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised layer.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        attn_key, proj_key = jax.random.split(key)
        self.c_attn = Conv1D(dim, 3 * dim, key=attn_key, dtype=dtype)
        self.c_proj = Conv1D(dim, dim, key=proj_key, dtype=dtype)
        self.num_heads = num_heads

    def __call__(
        self,
        x: Float[Array, "seq dim"],
        mask: Bool[Array, "seq keys"],
        cache: KVCache | None = None,
        cache_index: Int[Array, ""] | None = None,
    ) -> tuple[Float[Array, "seq dim"], KVCache | None]:
        """Attend over a sequence, optionally continuing from a cache.

        Args:
            x: Hidden state of each new token.
            mask: `True` where a new token may attend to a key.
            cache: Keys and values of earlier tokens, or `None`.
            cache_index: Cache slot for the first new token; required with a cache.

        Returns:
            The attention output, and the updated cache (or `None`).
        """
        seq, dim = x.shape
        heads = lambda t: t.reshape(seq, self.num_heads, dim // self.num_heads)  # noqa: E731
        q, k, v = (heads(t) for t in jnp.split(self.c_attn(x), 3, axis=-1))
        if cache is not None:
            cache = cache.update(k, v, cache_index)
            k, v = cache.k, cache.v
        # The reference's softmax computes in the input dtype (PyTorch upcasts bf16 internally).
        out = dot_product_attention(q, k, v, mask, softmax_dtype=at_least_float32(q.dtype))
        return self.c_proj(out.reshape(seq, dim)), cache


class GPT2MLP(eqx.Module):
    """Feed-forward block: `c_fc`, GELU (tanh approximation), then `c_proj`.

    Attributes:
        c_fc: Expansion.
        c_proj: Projection back to the model dimension.
    """

    c_fc: Conv1D
    c_proj: Conv1D

    def __init__(self, dim: int, inner: int, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised block.

        Args:
            dim: Model dimension.
            inner: Inner dimension.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        fc_key, proj_key = jax.random.split(key)
        self.c_fc = Conv1D(dim, inner, key=fc_key, dtype=dtype)
        self.c_proj = Conv1D(inner, dim, key=proj_key, dtype=dtype)

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
        """Apply the block.

        Args:
            x: Hidden state of each token.

        Returns:
            The block's output for each token.
        """
        return self.c_proj(jax.nn.gelu(self.c_fc(x), approximate=True))


class GPT2Block(eqx.Module):
    """Pre-norm transformer block: attention, then the MLP, each with a residual connection.

    Attributes:
        ln_1: Norm before attention.
        attn: Attention.
        ln_2: Norm before the MLP.
        mlp: Feed-forward block.
    """

    ln_1: LayerNorm
    attn: GPT2Attention
    ln_2: LayerNorm
    mlp: GPT2MLP

    def __init__(self, config: GPT2Config, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised block.

        Args:
            config: Model hyperparameters.
            key: PRNG key for initialisation.
            dtype: Parameter dtype.
        """
        attn_key, mlp_key = jax.random.split(key)
        dim, eps = config.n_embd, config.layer_norm_epsilon
        self.ln_1 = LayerNorm(dim, eps=eps, dtype=dtype)
        self.attn = GPT2Attention(dim, config.n_head, key=attn_key, dtype=dtype)
        self.ln_2 = LayerNorm(dim, eps=eps, dtype=dtype)
        self.mlp = GPT2MLP(dim, config.n_inner, key=mlp_key, dtype=dtype)

    def __call__(
        self,
        x: Float[Array, "seq dim"],
        mask: Bool[Array, "seq keys"],
        cache: KVCache | None = None,
        cache_index: Int[Array, ""] | None = None,
    ) -> tuple[Float[Array, "seq dim"], KVCache | None]:
        """Apply the block.

        Args:
            x: Residual stream for each token.
            mask: Attention mask; see `GPT2Attention`.
            cache: This block's key/value cache, or `None`.
            cache_index: Cache slot for the first token; required with a cache.

        Returns:
            The updated residual stream, and the updated cache (or `None`).
        """
        attn, cache = self.attn(self.ln_1(x), mask, cache, cache_index)
        x = x + attn
        return x + self.mlp(self.ln_2(x)), cache


class GPT2(eqx.Module):
    """GPT-2 language model, with the same interface as `CausalLM`.

    Example:
        >>> model = GPT2.from_pretrained("openai-community/gpt2")
        >>> logits, _ = model(jnp.array([464, 3139, 286, 4881, 318]))

    Attributes:
        wte: Token embedding, also used as the output head.
        wpe: Learned position embedding.
        h: The transformer blocks.
        ln_f: Final norm.
        config: Model hyperparameters.
        hf_config: The checkpoint's `config.json` text, kept for `save_pretrained`, or `None`.
    """

    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    h: list[GPT2Block]
    ln_f: LayerNorm
    config: GPT2Config = eqx.field(static=True)
    hf_config: str | None = eqx.field(static=True)

    def __init__(
        self,
        config: GPT2Config,
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
        wte_key, wpe_key, *block_keys = jax.random.split(key, 2 + config.n_layer)
        self.wte = eqx.nn.Embedding(config.vocab_size, config.n_embd, key=wte_key, dtype=dtype)
        self.wpe = eqx.nn.Embedding(config.n_positions, config.n_embd, key=wpe_key, dtype=dtype)
        self.h = [GPT2Block(config, key=k, dtype=dtype) for k in block_keys]
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon, dtype=dtype)
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

        Raises:
            ValueError: If the input is longer than the model's learned positions.
        """
        seq = input_ids.shape[0]
        if seq > self.config.n_positions:
            raise ValueError(f"{seq} tokens exceed the model's {self.config.n_positions} positions")
        valid = jnp.ones(seq, dtype=bool) if attention_mask is None else attention_mask.astype(bool)
        if cache is None:
            slots = jnp.arange(seq)
            positions = jnp.maximum(jnp.cumsum(valid) - 1, 0)
            mask = causal_mask(slots, slots, valid)
        else:
            slots = cache.length + jnp.arange(seq)
            positions = jnp.maximum(cache.valid.sum() + jnp.cumsum(valid) - 1, 0)
            key_valid = jax.lax.dynamic_update_slice(cache.valid, valid, (cache.length,))
            mask = causal_mask(slots, jnp.arange(key_valid.shape[0]), key_valid)

        x = self.wte.weight[input_ids] + self.wpe.weight[positions]
        layer_caches = []
        for i, block in enumerate(self.h):
            layer_cache = None if cache is None else cache.layers[i]
            index = None if cache is None else cache.length
            x, layer_cache = block(x, mask, layer_cache, index)
            layer_caches.append(layer_cache)
        x = self.ln_f(x)
        if last_only:
            x = x[-1:]
        logits = x @ self.wte.weight.T
        if cache is not None:
            cache = Cache(layers=layer_caches, length=cache.length + seq, valid=key_valid)
        return logits, cache

    def init_cache(self, max_len: int) -> Cache:
        """Allocate an empty cache for up to `max_len` tokens.

        Args:
            max_len: Number of token slots: the prompt plus the tokens to generate.

        Returns:
            An empty cache.

        Raises:
            ValueError: If `max_len` exceeds the model's learned positions.
        """
        c = self.config
        if max_len > c.n_positions:
            raise ValueError(f"{max_len} tokens exceed the model's {c.n_positions} positions")
        dtype = self.wte.weight.dtype
        head_dim = c.n_embd // c.n_head
        return Cache(
            layers=[KVCache.empty(max_len, c.n_head, head_dim, dtype) for _ in range(c.n_layer)],
            length=jnp.array(0, dtype=jnp.int32),
            valid=jnp.zeros(max_len, dtype=bool),
        )

    @classmethod
    def from_pretrained(
        cls, repo_or_path: str | os.PathLike[str], dtype: DTypeLike = jnp.float32
    ) -> "GPT2":
        """Load a Hugging Face GPT-2 checkpoint.

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
        config = GPT2Config.from_hf(json.loads(hf_config))
        skeleton = eqx.filter_eval_shape(
            cls, config, hf_config=hf_config, key=jax.random.key(0), dtype=dtype
        )
        files = sorted(path.glob("*.safetensors"))
        names = set()
        for file in files:
            with safe_open(file, framework="flax") as f:
                names |= set(f.keys())
        # The original checkpoints name tensors `h.0...`; Transformers saves `transformer.h.0...`.
        prefix = "transformer." if any(n.startswith("transformer.") for n in names) else ""
        # Saved causal-mask buffers, and the output head tied to the token embedding.
        ignore = ["lm_head.weight"] + [
            f"{prefix}h.{i}.attn.{buffer}"
            for i in range(config.n_layer)
            for buffer in ("bias", "masked_bias")
        ]
        return load_safetensors(
            skeleton, files, ignore=ignore, prefixes={"": prefix} if prefix else None
        )

    def save_pretrained(self, directory: str | os.PathLike[str]) -> None:
        """Save in Hugging Face format, as `config.json` and `model.safetensors`.

        Writes back the `config.json` the model was loaded with (its dtype set to the saved
        parameters'), and the weights named as Transformers saves GPT-2 (`transformer.h.0...`),
        with the output head tied to the token embedding.

        Args:
            directory: Destination directory; created if needed.

        Raises:
            ValueError: If the model has a LoRA adapter (merge it with `eqx_zoo.lora.merge`),
                or wasn't loaded with `from_pretrained`.
        """
        refuse_adapters(self)
        if self.hf_config is None:
            raise ValueError(
                "only models loaded with from_pretrained can be saved: there is no original "
                "config.json to write back"
            )
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(config_with_dtype(self.hf_config, self.wte.weight.dtype))
        save_safetensors(self, path / "model.safetensors", prefixes={"": "transformer."})
