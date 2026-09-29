"""Tests for translating Hugging Face configs."""

import pytest

from eqx_zoo import Config

QWEN3 = {
    "architectures": ["Qwen3ForCausalLM"],
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "tie_word_embeddings": True,
    "attention_bias": False,
}


def test_qwen3():
    config = Config.from_hf(QWEN3)
    assert config.qk_norm and not config.attention_bias
    assert config.head_dim == 128


def test_qwen2_defaults():
    qwen2 = {k: v for k, v in QWEN3.items() if k not in ("head_dim", "attention_bias")}
    config = Config.from_hf(qwen2 | {"architectures": ["Qwen2ForCausalLM"]})
    assert config.attention_bias and not config.qk_norm
    assert config.head_dim == 1024 // 16  # derived when not given explicitly


def test_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="unsupported architecture"):
        Config.from_hf(QWEN3 | {"architectures": ["MysteryForCausalLM"]})


@pytest.mark.parametrize(
    "extra",
    [
        {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        {"use_sliding_window": True},
        {"hidden_act": "gelu"},
    ],
)
def test_rejects_unsupported_features(extra):
    with pytest.raises(NotImplementedError):
        Config.from_hf(QWEN3 | extra)
