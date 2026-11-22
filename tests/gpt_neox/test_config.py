"""GPT-NeoX configuration parsing."""

import pytest

from eqx_zoo.models.gpt_neox.config import GPTNeoXConfig

# EleutherAI/pythia-160m's config.json, as saved by Transformers 4.24 (original key names).
PYTHIA_160M = {
    "architectures": ["GPTNeoXForCausalLM"],
    "bos_token_id": 0,
    "eos_token_id": 0,
    "hidden_act": "gelu",
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 2048,
    "model_type": "gpt_neox",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "rotary_emb_base": 10000,
    "rotary_pct": 0.25,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "transformers_version": "4.24.0",
    "use_cache": True,
    "use_parallel_residual": True,
    "vocab_size": 50304,
}

# The same model as Transformers 5 saves it: rotary settings in `rope_parameters`.
PYTHIA_160M_V5 = {
    k: v for k, v in PYTHIA_160M.items() if k not in ("rotary_emb_base", "rotary_pct")
} | {
    "attention_bias": True,
    "rope_parameters": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000.0,
        "rope_type": "default",
    },
    "dtype": "float16",
}


def test_pythia_160m():
    config = GPTNeoXConfig.from_hf(PYTHIA_160M)
    assert config == GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
        layer_norm_eps=1e-5,
        rotary_dim=16,
        rope_theta=10000.0,
        use_parallel_residual=True,
        pad_token_id=None,
    )
    assert config.head_size == 64


def test_rope_parameters_form_gives_the_same_config():
    assert GPTNeoXConfig.from_hf(PYTHIA_160M_V5) == GPTNeoXConfig.from_hf(PYTHIA_160M)


def test_rope_parameters_win_over_the_legacy_keys():
    """As in Transformers: the dictionary wins, the legacy keys only fill what it lacks."""
    both = PYTHIA_160M | {
        "rotary_pct": 1.0,
        "rotary_emb_base": 500,
        "rope_parameters": {"partial_rotary_factor": 0.5, "rope_theta": 20000.0},
    }
    config = GPTNeoXConfig.from_hf(both)
    assert (config.rotary_dim, config.rope_theta) == (32, 20000.0)

    partial = PYTHIA_160M | {"rotary_pct": 0.5, "rope_parameters": {"rope_theta": 20000.0}}
    config = GPTNeoXConfig.from_hf(partial)
    assert (config.rotary_dim, config.rope_theta) == (32, 20000.0)  # rotary_pct fills the gap

    legacy_name = PYTHIA_160M | {"rope_scaling": {"partial_rotary_factor": 0.5}}
    assert (
        GPTNeoXConfig.from_hf(
            legacy_name | {"rope_parameters": {"partial_rotary_factor": 1.0}}
        ).rotary_dim
        == 32
    )


def test_defaults_and_unknown_keys():
    minimal = {
        k: PYTHIA_160M[k]
        for k in (
            "architectures",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
        )
    }
    config = GPTNeoXConfig.from_hf(minimal | {"hidden_dropout": 0.1, "something_new": 1})
    assert (config.rotary_dim, config.rope_theta) == (16, 10000.0)
    assert (config.use_parallel_residual, config.layer_norm_eps) == (True, 1e-5)


def test_sequential_residual_and_full_rotary():
    config = GPTNeoXConfig.from_hf(
        PYTHIA_160M | {"use_parallel_residual": False, "rotary_pct": 1.0, "pad_token_id": 3}
    )
    assert (config.use_parallel_residual, config.rotary_dim, config.pad_token_id) == (False, 64, 3)


@pytest.mark.parametrize(
    "extra",
    [
        {"hidden_act": "gelu_fast"},
        {"hidden_act": "gelu_new"},
        {"attention_bias": False},
        {"tie_word_embeddings": True},
        {"rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}},
        {"rope_scaling": {"type": "dynamic", "factor": 2.0}},
        {"head_dim": 32},
    ],
)
def test_rejects_unsupported_features(extra):
    with pytest.raises(NotImplementedError):
        GPTNeoXConfig.from_hf(PYTHIA_160M | extra)


def test_rejects_other_architectures():
    for architectures in (["LlamaForCausalLM"], ["GPTNeoXModel"], []):
        with pytest.raises(ValueError, match="GPTNeoXForCausalLM"):
            GPTNeoXConfig.from_hf(PYTHIA_160M | {"architectures": architectures})


def test_rejects_hidden_size_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        GPTNeoXConfig.from_hf(PYTHIA_160M | {"num_attention_heads": 7})


@pytest.mark.parametrize("pct", [0.0, 1.5, 0.3])  # no rotation, beyond the head, odd (19 dims)
def test_rejects_invalid_rotary_dimension(pct):
    with pytest.raises(ValueError, match="rotary dimension"):
        GPTNeoXConfig.from_hf(PYTHIA_160M | {"rotary_pct": pct})
