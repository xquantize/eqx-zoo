"""Saving causal LMs in Hugging Face format, verified by loading them with transformers."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from safetensors import safe_open
from support.registry import TINY_MODELS
from transformers import AutoModelForCausalLM

from eqx_zoo import CausalLM, Config


def tiny_only(case):
    if case.name not in TINY_MODELS:
        pytest.skip("saving is tested on the tiny models")


def test_transformers_loads_saved_model(case, tmp_path):
    """Hugging Face loads what we save, and computes the reference logits from it."""
    tiny_only(case)
    case.model.save_pretrained(tmp_path)

    hf = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation="eager").eval()
    with torch.no_grad():
        logits = hf(torch.from_numpy(case.reference["input_ids"])[None]).logits[0].float().numpy()
    np.testing.assert_allclose(logits, case.reference["logits"], rtol=1e-5, atol=1e-5)

    original = json.loads((Path(case.source) / "config.json").read_text())
    saved = json.loads((tmp_path / "config.json").read_text())
    differing = {k for k in original.keys() | saved.keys() if original.get(k) != saved.get(k)}
    assert differing <= {"torch_dtype", "dtype"}, f"config fields changed: {sorted(differing)}"


def test_saves_bf16(case, bf16_model, tmp_path):
    """A bf16 model is saved as bf16, and its config says so."""
    tiny_only(case)
    bf16_model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert all(config[k] == "bfloat16" for k in ("torch_dtype", "dtype") if k in config)
    with safe_open(tmp_path / "model.safetensors", "pt") as f:
        assert all(f.get_tensor(n).dtype == torch.bfloat16 for n in f.keys())


def test_refuses_model_without_original_config(tmp_path):
    config = Config.from_hf(
        json.loads(
            json.dumps(
                {
                    "architectures": ["LlamaForCausalLM"],
                    "vocab_size": 8,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                }
            )
        )
    )
    model = CausalLM(config, key=jax.random.key(0), dtype=jnp.float32)
    with pytest.raises(ValueError, match="from_pretrained"):
        model.save_pretrained(tmp_path)


def test_saves_multimodal_as_text_only(tmp_path):
    """A multimodal checkpoint's language model is saved as a text-only checkpoint."""
    from causal_lm.test_config import QWEN35

    small = QWEN35 | {
        "text_config": QWEN35["text_config"]
        | {
            "hidden_size": 32,
            "intermediate_size": 64,
            "head_dim": 16,
            "vocab_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
        }
    }
    model = CausalLM(Config.from_hf(small), hf_config=json.dumps(small), key=jax.random.key(0))
    model.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["architectures"] == ["Qwen3_5ForCausalLM"]
    assert "text_config" not in saved and "vision_config" not in saved

    reloaded = CausalLM.from_pretrained(tmp_path)
    leaves = lambda m: jax.tree.leaves(eqx.filter(m, eqx.is_array))  # noqa: E731
    for a, b in zip(leaves(model), leaves(reloaded), strict=True):
        np.testing.assert_array_equal(a, b)
