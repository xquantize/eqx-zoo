"""GPT-2 configuration parsing."""

import pytest

from eqx_zoo.models.gpt2.config import GPT2Config

GPT2 = {
    "architectures": ["GPT2LMHeadModel"],
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-5,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50257,
}


def test_gpt2():
    config = GPT2Config.from_hf(GPT2)
    assert (config.n_embd, config.n_head, config.n_layer, config.n_inner) == (768, 12, 12, 3072)
    assert (config.n_positions, config.vocab_size, config.layer_norm_epsilon) == (1024, 50257, 1e-5)


@pytest.mark.parametrize(
    "extra",
    [
        {"activation_function": "relu"},
        {"scale_attn_by_inverse_layer_idx": True},
        {"reorder_and_upcast_attn": True},
        {"scale_attn_weights": False},
        {"add_cross_attention": True},
        {"tie_word_embeddings": False},
    ],
)
def test_rejects_unsupported_features(extra):
    with pytest.raises(NotImplementedError):
        GPT2Config.from_hf(GPT2 | extra)


def test_rejects_other_architectures():
    with pytest.raises(ValueError, match="GPT2LMHeadModel"):
        GPT2Config.from_hf(GPT2 | {"architectures": ["LlamaForCausalLM"]})
