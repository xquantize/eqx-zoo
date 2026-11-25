"""Merging adapters, and refusing to save adapted models, on the tiny models, against PEFT.

Every adapter is random and non-zero. The base model is float32, or bfloat16 with float32
adapters (as PEFT keeps them). Tolerances come from measurement: over 8 seeds, 28 model/option
cases per dtype (112 adapters per dtype), the largest disagreements were

- merged weights and biases against PEFT's merge_and_unload: exactly 0, in both dtypes;
- logits of the merged model against the adapted model: 2.5e-6 in float32, 3.7e-2 in bfloat16;
- logits of the merged model against PEFT's merged model: 1.5e-6 and 2.1e-2;
- logits of the saved merged model, reloaded in Transformers, against PEFT's merged model:
  exactly 0 (the same weights);
- yardstick: the adapter-free base model against Transformers' logits: 4.7e-7 in float32 and
  7.9e-3 in bfloat16. In bfloat16 the adapted layer rounds its base output and the update
  separately where the merged layer rounds the sum, which is why those two disagree by a few
  times the yardstick; a wrong merge (a missing scaling, say) changes the logits by order 1.

Each tolerance is 2-7x the measurement.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("peft")
import torch  # noqa: E402
import transformers  # noqa: E402
from support.gradients import gradients_by_name  # noqa: E402

from eqx_zoo.lora import add_adapter, load_adapter, merge, trainable  # noqa: E402
from eqx_zoo.lora.adapt import adapter_layers  # noqa: E402
from lora.helpers import IDS, MODELS, adapted, load  # noqa: E402
from lora.peft_adapters import (  # noqa: E402
    OPTIONS,
    make_peft_adapter,
    peft_logits,
    relative_error,
)

DTYPES = {"float32": (torch.float32, jnp.float32), "bfloat16": (torch.bfloat16, jnp.bfloat16)}

# Merged weights and biases against PEFT's, relative to the largest magnitude. Measured: exactly
# 0 in both dtypes; the bfloat16 limit allows one rounding step (2**-7 of the largest value).
WEIGHT_TOLERANCE = {"float32": 1e-6, "bfloat16": 2**-7}
# Logits, relative to the largest logit: the merged model against the adapted model; our merged
# model against PEFT's merged model; and the saved merged model, loaded in Transformers, against
# PEFT's merged model (the same weights, so only the weights' rounding can differ).
ADAPTED_TOLERANCE = {"float32": 1e-5, "bfloat16": 7.5e-2}
PEFT_TOLERANCE = {"float32": 1e-5, "bfloat16": 4e-2}
SAVED_TOLERANCE = {"float32": 1e-6, "bfloat16": 1e-2}

# `lora_B` biases need a bias to merge into: Llama and Qwen2 have layers without one.
HAS_BIASES = ("tiny-gpt2", "tiny-gpt-neox")
CASES = [
    pytest.param(name, option, dtype, id=f"{name}-{option}-{dtype}")
    for name in MODELS
    for option in OPTIONS
    for dtype in DTYPES
    if option != "lora-bias" or name in HAS_BIASES
]


def parameters(model) -> dict[str, np.ndarray]:
    """The model's parameters by name, keeping their dtype."""
    return gradients_by_name(model)


def hf_name(model, name: str) -> str:
    """Our parameter name as Hugging Face's: GPT-2 keeps everything under `transformer.`."""
    return ("transformer." if type(model).__name__ == "GPT2" else "") + name


def float32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def scenario(checkpoint, tmp_path, name, option, dtype, seed=0):
    """PEFT's random adapter, loaded here and merged by both libraries.

    Returns:
        The base model, the adapted model, our merged model, and PEFT's merged model.
    """
    torch_dtype, jax_dtype = DTYPES[dtype]
    reference = make_peft_adapter(checkpoint, name, OPTIONS[option], seed, torch_dtype)
    reference.save_pretrained(tmp_path / "adapter")
    base = load(checkpoint, name, dtype=jax_dtype)
    model = load_adapter(base, tmp_path / "adapter")
    return base, model, merge(model), reference.merge_and_unload().eval()


@pytest.mark.parametrize(("name", "option", "dtype"), CASES)
def test_merged_weights_match_pefts_merge(checkpoint, tmp_path, name, option, dtype):
    base, model, merged, unloaded = scenario(checkpoint, tmp_path, name, option, dtype)
    expected = {n: p.detach().float().numpy() for n, p in unloaded.named_parameters()}
    before, after = parameters(base), parameters(merged)
    assert before.keys() == after.keys()
    adapted_modules = set(adapter_layers(model))
    changed_weights = set()
    for key, value in after.items():
        assert value.dtype == before[key].dtype, f"{key}: the dtype must not change"
        module = hf_name(merged, key).rsplit(".", 1)[0]
        unchanged = np.array_equal(float32(value), float32(before[key]))
        if module not in adapted_modules:
            assert unchanged, f"{key} changed, but its layer has no adapter"
            continue
        reference = expected[hf_name(merged, key)]
        error = np.abs(float32(value) - reference).max() / np.abs(reference).max()
        assert error <= WEIGHT_TOLERANCE[dtype], f"{key}: relative error {error:.3g}"
        if key.endswith(".weight"):
            assert not unchanged, f"{key} should have changed"
            changed_weights.add(module)
    assert changed_weights == adapted_modules


@pytest.mark.parametrize(("name", "option", "dtype"), CASES)
def test_merged_logits(checkpoint, tmp_path, name, option, dtype):
    base, model, merged, unloaded = scenario(checkpoint, tmp_path, name, option, dtype)
    adapted_logits = float32(model(IDS)[0])
    assert relative_error(base(IDS)[0], adapted_logits) > 1e-2, "the adapter must matter"
    assert relative_error(merged(IDS)[0], adapted_logits) <= ADAPTED_TOLERANCE[dtype]
    assert relative_error(merged(IDS)[0], peft_logits(unloaded)) <= PEFT_TOLERANCE[dtype]


@pytest.mark.parametrize(("name", "option", "dtype"), CASES)
def test_saved_merged_model_loads_in_transformers(checkpoint, tmp_path, name, option, dtype):
    torch_dtype, _ = DTYPES[dtype]
    _, _, merged, unloaded = scenario(checkpoint, tmp_path, name, option, dtype)
    merged.save_pretrained(tmp_path / "merged")
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "merged", dtype=torch_dtype, attn_implementation="eager"
    ).eval()
    assert relative_error(peft_logits(reloaded), peft_logits(unloaded)) <= SAVED_TOLERANCE[dtype]
    assert relative_error(merged(IDS)[0], peft_logits(reloaded)) <= PEFT_TOLERANCE[dtype]


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_merging_a_fresh_adapter_changes_nothing(checkpoint, name, dtype):
    base = load(checkpoint, name, dtype=DTYPES[dtype][1])
    model = add_adapter(
        base,
        r=4,
        alpha=8,
        target_modules="all-linear",
        key=jax.random.key(0),
        lora_bias=name in HAS_BIASES,
    )
    merged = merge(model)
    before, after = parameters(base), parameters(merged)
    assert before.keys() == after.keys()
    for key in before:
        assert after[key].dtype == before[key].dtype
        assert np.array_equal(float32(after[key]), float32(before[key])), key
    assert np.array_equal(float32(merged(IDS)[0]), float32(base(IDS)[0]))


@pytest.mark.parametrize("name", MODELS)
def test_a_merged_model_is_a_plain_model(checkpoint, name):
    base, model = adapted(checkpoint, name)
    merged = merge(model)
    assert jax.tree_util.tree_structure(merged) == jax.tree_util.tree_structure(base)
    assert type(merged) is type(base)
    assert not adapter_layers(merged)
    assert not any(jax.tree_util.tree_leaves(trainable(merged)))
    with pytest.raises(ValueError, match="no adapter to merge"):
        merge(merged)


def test_merging_needs_a_base_bias_for_lora_bias(checkpoint, tmp_path):
    """PEFT raises too: a `lora_B` bias has nowhere to go in a layer without a bias."""
    with pytest.warns(UserWarning, match="no bias"):
        _, model = adapted(checkpoint, "tiny-llama", lora_bias=True)
    with pytest.raises(ValueError, match="no bias"):
        merge(model)
    reference = make_peft_adapter(checkpoint, "tiny-llama", OPTIONS["lora-bias"])
    with pytest.raises(RuntimeError, match="Impossible to merge"):
        reference.merge_and_unload()


# --- saving ---


@pytest.mark.parametrize("name", MODELS)
def test_adapted_models_cannot_be_saved_as_checkpoints(checkpoint, tmp_path, name):
    _, model = adapted(checkpoint, name)
    with pytest.raises(
        ValueError, match=r"eqx_zoo\.lora\.merge\(model\).*eqx_zoo\.lora\.save_adapter"
    ):
        model.save_pretrained(tmp_path / "out")
    assert not (tmp_path / "out").exists(), "nothing may be written before refusing"
    merge(model).save_pretrained(tmp_path / "out")  # the merged model saves as usual
    assert (tmp_path / "out" / "model.safetensors").exists()


@pytest.mark.parametrize("name", MODELS)
def test_one_adapted_layer_is_enough_to_refuse(checkpoint, tmp_path, name):
    target = {
        "tiny-llama": ["model.layers.0.self_attn.q_proj"],
        "tiny-qwen2": ["model.layers.1.mlp.up_proj"],
        "tiny-gpt2": ["transformer.h.1.mlp.c_fc"],
        "tiny-gpt-neox": ["gpt_neox.layers.1.attention.dense"],
    }[name]
    _, model = adapted(checkpoint, name, target_modules=target)
    assert len(adapter_layers(model)) == 1
    with pytest.raises(ValueError, match="LoRA adapter"):
        model.save_pretrained(tmp_path / "out")


def test_merging_a_plain_model_is_refused(checkpoint):
    with pytest.raises(ValueError, match="no adapter to merge"):
        merge(load(checkpoint, "tiny-llama"))
