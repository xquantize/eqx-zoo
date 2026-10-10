"""BERT- and RoBERTa-style bidirectional encoders.

Module and attribute names mirror Hugging Face's `BertModel`, so checkpoints load by name.

Example:
    >>> import jax.numpy as jnp
    >>> from eqx_zoo import Encoder
    >>> model = Encoder.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    >>> hidden = model(jnp.array([101, 7592, 2088, 102]))
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
from eqx_zoo._sentence_transformers import pool, read_pipeline
from eqx_zoo.layers import LayerNorm as _LayerNorm
from eqx_zoo.layers import dot_product_attention
from eqx_zoo.layers._common import linear
from eqx_zoo.models.encoder.config import EncoderConfig

# Checkpoint tensors that are deliberately unused: the stored position-id buffer (positions
# are computed instead) and BERT's pooler, which sentence-transformers models don't use.
_IGNORED = ["embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"]


class Embeddings(eqx.Module):
    """Sum of token, learned position and token-type embeddings, then a LayerNorm.

    Attributes:
        word_embeddings: Token embedding table.
        position_embeddings: Learned embedding for each position.
        token_type_embeddings: Embedding for each token type (segment).
        LayerNorm: Norm applied to the summed embeddings.
        pad_token_id: Id of the padding token.
        padding_aware_positions: Whether positions count only non-padding tokens, as in RoBERTa.
    """

    word_embeddings: eqx.nn.Embedding
    position_embeddings: eqx.nn.Embedding
    token_type_embeddings: eqx.nn.Embedding
    LayerNorm: _LayerNorm
    pad_token_id: int = eqx.field(static=True)
    padding_aware_positions: bool = eqx.field(static=True)

    def __init__(self, config: EncoderConfig, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create randomly initialised embeddings.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        word_key, position_key, type_key = jax.random.split(key, 3)
        dim = config.hidden_size
        self.word_embeddings = eqx.nn.Embedding(config.vocab_size, dim, key=word_key, dtype=dtype)
        self.position_embeddings = eqx.nn.Embedding(
            config.max_position_embeddings, dim, key=position_key, dtype=dtype
        )
        self.token_type_embeddings = eqx.nn.Embedding(
            config.type_vocab_size, dim, key=type_key, dtype=dtype
        )
        self.LayerNorm = _LayerNorm(dim, eps=config.layer_norm_eps, dtype=dtype)
        self.pad_token_id = config.pad_token_id
        self.padding_aware_positions = config.padding_aware_positions

    def __call__(
        self, input_ids: Int[Array, " seq"], token_type_ids: Int[Array, " seq"]
    ) -> Float[Array, "seq dim"]:
        """Embed a sequence.

        Args:
            input_ids: Token ids.
            token_type_ids: Token type (segment) of each token.

        Returns:
            The normalised embedding of each token.
        """
        if self.padding_aware_positions:
            # RoBERTa: real tokens are numbered from pad_token_id + 1, counting only
            # non-padding tokens (identified by id); padding tokens get pad_token_id itself.
            real = input_ids != self.pad_token_id
            positions = jnp.cumsum(real) * real + self.pad_token_id
        else:
            positions = jnp.arange(input_ids.shape[0])
        words = self.word_embeddings.weight[input_ids]
        position = self.position_embeddings.weight[positions]
        # As with PyTorch's `padding_idx`: the padding token's embedding is used as is but
        # never trained, so no gradient reaches it. In RoBERTa, padding tokens also take the
        # padding position, whose embedding is likewise never trained.
        is_pad = (input_ids == self.pad_token_id)[:, None]
        words = jnp.where(is_pad, jax.lax.stop_gradient(words), words)
        if self.padding_aware_positions:
            position = jnp.where(is_pad, jax.lax.stop_gradient(position), position)
        x = words + position + self.token_type_embeddings.weight[token_type_ids]
        return self.LayerNorm(x)


class SelfAttention(eqx.Module):
    """Bidirectional multi-head attention projections and dot-product attention.

    Attributes:
        query: Query projection.
        key: Key projection.
        value: Value projection.
        num_heads: Number of attention heads.
    """

    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)

    def __init__(self, config: EncoderConfig, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create randomly initialised projections.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        q_key, k_key, v_key = jax.random.split(key, 3)
        dim = config.hidden_size
        self.query = linear(dim, dim, key=q_key, dtype=dtype, bias=True)
        self.key = linear(dim, dim, key=k_key, dtype=dtype, bias=True)
        self.value = linear(dim, dim, key=v_key, dtype=dtype, bias=True)
        self.num_heads = config.num_attention_heads

    def __call__(
        self, x: Float[Array, "seq dim"], mask: Bool[Array, "seq seq"]
    ) -> Float[Array, "seq dim"]:
        """Attend over a sequence.

        Args:
            x: Hidden state of each token.
            mask: `True` where a token may attend to another.

        Returns:
            The attention output for each token, before the output projection.
        """
        seq, dim = x.shape
        heads = (seq, self.num_heads, dim // self.num_heads)
        q = jax.vmap(self.query)(x).reshape(heads)
        k = jax.vmap(self.key)(x).reshape(heads)
        v = jax.vmap(self.value)(x).reshape(heads)
        return dot_product_attention(q, k, v, mask).reshape(seq, dim)


class ResidualOutput(eqx.Module):
    """Dense projection, residual connection, then LayerNorm (post-norm).

    Used for both the attention output and the MLP output, as in Hugging Face's BERT.

    Attributes:
        dense: Projection back to the model dimension.
        LayerNorm: Norm applied after the residual connection.
    """

    dense: eqx.nn.Linear
    LayerNorm: _LayerNorm

    def __init__(self, in_dim: int, config: EncoderConfig, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised block.

        Args:
            in_dim: Input dimension of the dense projection.
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        self.dense = linear(in_dim, config.hidden_size, key=key, dtype=dtype, bias=True)
        self.LayerNorm = _LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype)

    def __call__(
        self, h: Float[Array, "seq in_dim"], residual: Float[Array, "seq dim"]
    ) -> Float[Array, "seq dim"]:
        """Project `h`, add the residual, and normalise.

        Args:
            h: Output of the preceding sub-block.
            residual: Input to the preceding sub-block.

        Returns:
            The normalised sum.
        """
        return self.LayerNorm(jax.vmap(self.dense)(h) + residual)


class AttentionBlock(eqx.Module):
    """Self-attention followed by its output projection, residual and LayerNorm.

    The attribute is named `self`, as in Hugging Face, so methods name their first
    parameter `module` instead.

    Attributes:
        self: Attention projections and dot-product attention.
        output: Output projection, residual connection and LayerNorm.
    """

    self: SelfAttention
    output: ResidualOutput

    def __init__(module, config: EncoderConfig, *, key: PRNGKeyArray, dtype: DTypeLike):  # noqa: N805
        """Create a randomly initialised block.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        attn_key, out_key = jax.random.split(key)
        module.self = SelfAttention(config, key=attn_key, dtype=dtype)
        module.output = ResidualOutput(config.hidden_size, config, key=out_key, dtype=dtype)

    def __call__(  # noqa: N805
        module, x: Float[Array, "seq dim"], mask: Bool[Array, "seq seq"]
    ) -> Float[Array, "seq dim"]:
        """Apply attention with its residual connection and norm.

        Args:
            x: Hidden state of each token.
            mask: `True` where a token may attend to another.

        Returns:
            The updated hidden states.
        """
        return module.output(module.self(x, mask), x)


class Intermediate(eqx.Module):
    """Expansion of each token to the MLP dimension, followed by exact GELU.

    Attributes:
        dense: Projection to the intermediate dimension.
    """

    dense: eqx.nn.Linear

    def __init__(self, config: EncoderConfig, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised block.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        self.dense = linear(
            config.hidden_size, config.intermediate_size, key=key, dtype=dtype, bias=True
        )

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq intermediate"]:
        """Expand each token.

        Args:
            x: Hidden state of each token.

        Returns:
            The activated expansion.
        """
        return jax.nn.gelu(jax.vmap(self.dense)(x), approximate=False)


class EncoderLayer(eqx.Module):
    """Post-norm transformer encoder layer.

    Attributes:
        attention: Self-attention block.
        intermediate: MLP expansion and activation.
        output: MLP projection, residual connection and LayerNorm.
    """

    attention: AttentionBlock
    intermediate: Intermediate
    output: ResidualOutput

    def __init__(self, config: EncoderConfig, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create a randomly initialised layer.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        attn_key, mid_key, out_key = jax.random.split(key, 3)
        self.attention = AttentionBlock(config, key=attn_key, dtype=dtype)
        self.intermediate = Intermediate(config, key=mid_key, dtype=dtype)
        self.output = ResidualOutput(config.intermediate_size, config, key=out_key, dtype=dtype)

    def __call__(
        self, x: Float[Array, "seq dim"], mask: Bool[Array, "seq seq"]
    ) -> Float[Array, "seq dim"]:
        """Apply the layer.

        Args:
            x: Hidden state of each token.
            mask: `True` where a token may attend to another.

        Returns:
            The updated hidden states.
        """
        x = self.attention(x, mask)
        return self.output(self.intermediate(x), x)


class EncoderStack(eqx.Module):
    """The encoder layers, applied in order.

    Attributes:
        layer: Encoder layers (named in the singular, as in Hugging Face).
    """

    layer: list[EncoderLayer]

    def __init__(self, config: EncoderConfig, *, key: PRNGKeyArray, dtype: DTypeLike):
        """Create randomly initialised layers.

        Args:
            config: Model hyperparameters.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        keys = jax.random.split(key, config.num_hidden_layers)
        self.layer = [EncoderLayer(config, key=k, dtype=dtype) for k in keys]


class Encoder(eqx.Module):
    """BERT-style bidirectional encoder, returning a hidden state for every token.

    Attributes:
        embeddings: Token, position and token-type embeddings.
        encoder: The encoder layers.
        config: Model hyperparameters.
        pooling: How `embed` pools token states (`"mean"` or `"cls"`), or `None` if the
            checkpoint has no sentence-transformers configuration. Defaults to `None`.
        normalize: Whether `embed` scales embeddings to unit length.
    """

    embeddings: Embeddings
    encoder: EncoderStack
    config: EncoderConfig = eqx.field(static=True)
    pooling: str | None = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        config: EncoderConfig,
        *,
        pooling: str | None = None,
        normalize: bool = False,
        key: PRNGKeyArray,
        dtype: DTypeLike = jnp.float32,
    ):
        """Create a randomly initialised encoder.

        Args:
            config: Model hyperparameters.
            pooling: How `embed` pools token states (`"mean"` or `"cls"`), or `None` if the
                checkpoint has no sentence-transformers configuration.
            normalize: Whether `embed` scales embeddings to unit length.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        embed_key, encoder_key = jax.random.split(key)
        self.embeddings = Embeddings(config, key=embed_key, dtype=dtype)
        self.encoder = EncoderStack(config, key=encoder_key, dtype=dtype)
        self.config = config
        self.pooling = pooling
        self.normalize = normalize

    def __call__(
        self,
        input_ids: Int[Array, " seq"],
        attention_mask: Bool[Array, " seq"] | Int[Array, " seq"] | None = None,
        token_type_ids: Int[Array, " seq"] | None = None,
    ) -> Float[Array, "seq dim"]:
        """Compute a hidden state for every token.

        Args:
            input_ids: Token ids.
            attention_mask: `1` or `True` for real tokens and `0` or `False` for padding.
                Padding is ignored by attention; its hidden states are unspecified.
                Defaults to all real.
            token_type_ids: Token type (segment) of each token. Defaults to all zeros.

        Returns:
            The final hidden state of each token.
        """
        seq = input_ids.shape[0]
        valid = jnp.ones(seq, dtype=bool) if attention_mask is None else attention_mask.astype(bool)
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        mask = jnp.broadcast_to(valid[None, :], (seq, seq))  # every token sees every real token

        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.encoder.layer:
            x = layer(x, mask)
        return x

    def embed(
        self,
        input_ids: Int[Array, " seq"],
        attention_mask: Bool[Array, " seq"] | Int[Array, " seq"] | None = None,
        token_type_ids: Int[Array, " seq"] | None = None,
    ) -> Float[Array, " dim"]:
        """Compute a sentence embedding, as the checkpoint's sentence-transformers pipeline does.

        Args:
            input_ids: Token ids.
            attention_mask: `1` or `True` for real tokens and `0` or `False` for padding.
                Defaults to all real.
            token_type_ids: Token type (segment) of each token. Defaults to all zeros.

        Returns:
            The pooled (and, if configured, normalised) embedding.

        Raises:
            ValueError: If the checkpoint has no sentence-transformers pooling configuration.
        """
        if self.pooling is None:
            raise ValueError("this checkpoint has no sentence-transformers pooling configuration")
        hidden = self(input_ids, attention_mask, token_type_ids)
        mask = (
            jnp.ones(input_ids.shape[0], dtype=bool)
            if attention_mask is None
            else attention_mask.astype(bool)
        )
        return pool(hidden, mask, self.pooling, self.normalize)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | os.PathLike[str],
        *,
        dtype: DTypeLike = jnp.float32,
        revision: str | None = None,
    ) -> "Encoder":
        """Load a pretrained encoder from the Hugging Face Hub or a local directory.

        Args:
            repo_id: Hub repository, e.g. `"sentence-transformers/all-MiniLM-L6-v2"`, or a
                local directory containing `config.json` and safetensors files.
            dtype: Dtype to cast the parameters to.
            revision: Optional git revision of a Hub repository; ignored for directories.

        Returns:
            The encoder with pretrained weights.
        """
        if Path(repo_id).is_dir():
            path = Path(repo_id)
        else:
            path = Path(
                snapshot_download(
                    str(repo_id),
                    revision=revision,
                    allow_patterns=[
                        "config.json",
                        "*.safetensors",
                        "modules.json",
                        "*/config.json",
                    ],
                )
            )
        config = EncoderConfig.from_hf(json.loads((path / "config.json").read_text()))
        pooling, normalize = read_pipeline(path)
        skeleton = eqx.filter_eval_shape(
            cls, config, pooling=pooling, normalize=normalize, key=jax.random.key(0), dtype=dtype
        )
        return load_safetensors(skeleton, sorted(path.glob("*.safetensors")), ignore=_IGNORED)
