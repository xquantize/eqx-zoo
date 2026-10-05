"""Decoder-based sentence-embedding models, such as Qwen3-Embedding.

Example:
    >>> import jax.numpy as jnp
    >>> from eqx_zoo import DecoderEmbedder
    >>> model = DecoderEmbedder.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    >>> embedding = model.embed(jnp.array([785, 6722, 315, 9625, 374]))
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
from eqx_zoo.layers import causal_mask
from eqx_zoo.models.causal_lm.config import Config
from eqx_zoo.models.causal_lm.model import DecoderModel


class DecoderEmbedder(eqx.Module):
    """A decoder stack used as a sentence-embedding model.

    Runs the decoder with causal attention, then pools its final hidden states as the
    checkpoint's sentence-transformers configuration specifies (for example, the last
    real token) and normalises them.

    Attributes:
        model: The decoder stack.
        config: Model hyperparameters.
        pooling: How `embed` pools token states (`"last_token"`, `"mean"` or `"cls"`), or
            `None` if the checkpoint has no sentence-transformers configuration.
        normalize: Whether `embed` scales embeddings to unit length.
    """

    model: DecoderModel
    config: Config = eqx.field(static=True)
    pooling: str | None = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        config: Config,
        *,
        pooling: str | None = None,
        normalize: bool = False,
        key: PRNGKeyArray,
        dtype: DTypeLike = jnp.float32,
    ):
        """Create a randomly initialised embedder.

        Args:
            config: Model hyperparameters.
            pooling: How `embed` pools token states, or `None`.
            normalize: Whether `embed` normalises embeddings.
            key: PRNG key for parameter initialisation.
            dtype: Parameter dtype.
        """
        self.model = DecoderModel(config, key=key, dtype=dtype)
        self.config = config
        self.pooling = pooling
        self.normalize = normalize

    def __call__(
        self,
        input_ids: Int[Array, " seq"],
        attention_mask: Bool[Array, " seq"] | Int[Array, " seq"] | None = None,
    ) -> Float[Array, "seq dim"]:
        """Compute the final hidden state of every token.

        Args:
            input_ids: Token ids.
            attention_mask: `1` or `True` for real tokens and `0` or `False` for padding.
                Defaults to all real.

        Returns:
            The final (normalised) hidden state of each token.
        """
        seq = input_ids.shape[0]
        valid = jnp.ones(seq, dtype=bool) if attention_mask is None else attention_mask.astype(bool)
        slots = jnp.arange(seq)
        positions = jnp.maximum(jnp.cumsum(valid) - 1, 0)
        hidden, _ = self.model(input_ids, positions, causal_mask(slots, slots, valid))
        return hidden

    def embed(
        self,
        input_ids: Int[Array, " seq"],
        attention_mask: Bool[Array, " seq"] | Int[Array, " seq"] | None = None,
    ) -> Float[Array, " dim"]:
        """Compute a sentence embedding, as the checkpoint's sentence-transformers pipeline does.

        Args:
            input_ids: Token ids, including any prompt the model expects.
            attention_mask: `1` or `True` for real tokens and `0` or `False` for padding.
                Defaults to all real.

        Returns:
            The pooled (and, if configured, normalised) embedding.

        Raises:
            ValueError: If the checkpoint has no sentence-transformers pooling configuration.
        """
        if self.pooling is None:
            raise ValueError("this checkpoint has no sentence-transformers pooling configuration")
        mask = (
            jnp.ones(input_ids.shape[0], dtype=bool)
            if attention_mask is None
            else attention_mask.astype(bool)
        )
        return pool(self(input_ids, attention_mask), mask, self.pooling, self.normalize)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | os.PathLike[str],
        *,
        dtype: DTypeLike = jnp.float32,
        revision: str | None = None,
    ) -> "DecoderEmbedder":
        """Load a decoder embedding model from the Hugging Face Hub or a local directory.

        The checkpoint holds the decoder stack alone (`embed_tokens`, `layers`, `norm`),
        without a language-model head, as embedding models are usually saved.

        Args:
            repo_id: Hub repository, e.g. `"Qwen/Qwen3-Embedding-0.6B"`, or a local directory.
            dtype: Dtype to cast the parameters to.
            revision: Optional git revision of a Hub repository; ignored for directories.

        Returns:
            The embedder with pretrained weights.
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
        config = Config.from_hf(json.loads((path / "config.json").read_text()))
        pooling, normalize = read_pipeline(path)
        skeleton = eqx.filter_eval_shape(
            cls, config, pooling=pooling, normalize=normalize, key=jax.random.key(0), dtype=dtype
        )
        stacked = [f"layers.{i}.mlp.experts" for i in config.moe_layers]
        decoder = load_safetensors(
            skeleton.model, sorted(path.glob("*.safetensors")), stacked=stacked
        )
        return eqx.tree_at(lambda e: e.model, skeleton, decoder)
