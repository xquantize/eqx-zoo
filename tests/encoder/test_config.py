"""Tests for translating Hugging Face encoder configs."""

import pytest

from eqx_zoo import EncoderConfig

BERT = {
    "architectures": ["BertModel"],
    "vocab_size": 30522,
    "hidden_size": 384,
    "num_hidden_layers": 6,
    "num_attention_heads": 12,
    "intermediate_size": 1536,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
    "hidden_act": "gelu",
    "position_embedding_type": "absolute",
}


def test_bert():
    config = EncoderConfig.from_hf(BERT)
    assert (config.hidden_size, config.num_hidden_layers) == (384, 6)
    assert config.layer_norm_eps == 1e-12


def test_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="unsupported architecture"):
        EncoderConfig.from_hf(BERT | {"architectures": ["MysteryModel"]})


@pytest.mark.parametrize(
    "extra", [{"hidden_act": "gelu_new"}, {"position_embedding_type": "relative_key"}]
)
def test_rejects_unsupported_features(extra):
    with pytest.raises(NotImplementedError):
        EncoderConfig.from_hf(BERT | extra)


@pytest.mark.parametrize("architecture", ["RobertaModel", "XLMRobertaModel"])
def test_roberta_positions(architecture):
    roberta = BERT | {"architectures": [architecture], "type_vocab_size": 1, "pad_token_id": 1}
    config = EncoderConfig.from_hf(roberta)
    assert config.padding_aware_positions and config.pad_token_id == 1
    assert not EncoderConfig.from_hf(BERT).padding_aware_positions
