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


def test_llama3_rope_scaling():
    rope = {
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    config = Config.from_hf(QWEN3 | {"architectures": ["LlamaForCausalLM"], "rope_scaling": rope})
    assert config.rope_scaling is not None and config.rope_scaling.factor == 32.0
    assert not config.qk_norm and not config.attention_bias


def test_rejects_mlp_bias():
    with pytest.raises(NotImplementedError, match="MLP bias"):
        Config.from_hf(QWEN3 | {"mlp_bias": True})


QWEN3_MOE = QWEN3 | {
    "architectures": ["Qwen3MoeForCausalLM"],
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
}


def test_qwen3_moe():
    config = Config.from_hf(QWEN3_MOE)
    assert (config.num_experts, config.num_experts_per_tok) == (128, 8)
    assert config.moe_intermediate_size == 768 and config.norm_topk_prob
    assert config.qk_norm and not config.attention_bias
    assert config.moe_layers == tuple(range(28))  # every layer is MoE


def test_moe_layer_selection():
    config = Config.from_hf(QWEN3_MOE | {"decoder_sparse_step": 2, "mlp_only_layers": [1]})
    # With a step of 2, layers 1, 3, 5, ... are MoE; layer 1 is then forced to be dense.
    assert config.moe_layers == tuple(i for i in range(28) if (i + 1) % 2 == 0 and i != 1)


def test_dense_configs_have_no_experts():
    config = Config.from_hf(QWEN3)
    assert config.num_experts == 0 and config.moe_layers == ()
