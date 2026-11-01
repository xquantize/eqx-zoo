"""Pooling supplied to `from_pretrained`, for encoders without a sentence-transformers pipeline."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from support.tiny import build_encoder
from transformers import AutoModel

from eqx_zoo import Encoder
from eqx_zoo._sentence_transformers import read_pipeline

IDS = np.array([5, 17, 42, 8, 99, 3, 61, 12])


@pytest.fixture
def tiny(tmp_path):
    """A tiny BERT without a sentence-transformers pipeline, as MedCPT's encoders are."""
    build_encoder("tiny-bert", tmp_path / "tiny")
    assert not (tmp_path / "tiny" / "modules.json").exists()
    return tmp_path / "tiny"


def test_supplied_cls_pooling_matches_transformers(tiny):
    hf = AutoModel.from_pretrained(tiny, attn_implementation="eager").eval()
    with torch.no_grad():
        expected = hf(torch.from_numpy(IDS)[None]).last_hidden_state[0, 0].numpy()
    model = Encoder.from_pretrained(tiny, pooling="cls")
    assert (model.pooling, model.normalize) == ("cls", False)
    np.testing.assert_allclose(model.embed(jnp.asarray(IDS)), expected, rtol=1e-5, atol=1e-6)


def test_supplied_normalize(tiny):
    embedding = Encoder.from_pretrained(tiny, pooling="mean", normalize=True).embed(
        jnp.asarray(IDS)
    )
    np.testing.assert_allclose(jnp.linalg.norm(embedding), 1.0, rtol=1e-6)


def test_saved_copy_keeps_supplied_pooling(tiny, tmp_path):
    model = Encoder.from_pretrained(tiny, pooling="cls")
    model.save_pretrained(tmp_path / "saved")
    assert read_pipeline(tmp_path / "saved") == ("cls", False)
    reloaded = Encoder.from_pretrained(tmp_path / "saved")
    np.testing.assert_array_equal(reloaded.embed(jnp.asarray(IDS)), model.embed(jnp.asarray(IDS)))


def test_arguments_may_restate_but_not_contradict_a_pipeline(tiny, tmp_path):
    Encoder.from_pretrained(tiny, pooling="cls").save_pretrained(tmp_path / "saved")
    Encoder.from_pretrained(tmp_path / "saved", pooling="cls", normalize=False)  # restates: fine
    with pytest.raises(ValueError, match="contradicts"):
        Encoder.from_pretrained(tmp_path / "saved", pooling="mean")
    with pytest.raises(ValueError, match="contradicts"):
        Encoder.from_pretrained(tmp_path / "saved", normalize=True)


def test_rejects_unknown_pooling(tiny):
    with pytest.raises(ValueError, match="pooling must be one of"):
        Encoder.from_pretrained(tiny, pooling="max")


def test_without_pooling_embed_still_explains(tiny):
    with pytest.raises(ValueError, match="pooling"):
        Encoder.from_pretrained(tiny).embed(jnp.asarray(IDS))


def test_decoder_embedder_supplied_pooling(tmp_path):
    from support.tiny import build_embedder

    from eqx_zoo import DecoderEmbedder

    build_embedder("tiny-qwen3-embedder", tmp_path / "tiny")
    model = DecoderEmbedder.from_pretrained(tmp_path / "tiny", pooling="last_token", normalize=True)
    model.save_pretrained(tmp_path / "saved")
    assert read_pipeline(tmp_path / "saved") == ("last_token", True)
    with pytest.raises(ValueError, match="contradicts"):
        DecoderEmbedder.from_pretrained(tmp_path / "saved", pooling="cls")
