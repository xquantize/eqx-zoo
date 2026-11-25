"""Loading and saving PEFT adapters, against PEFT itself, on the tiny models.

Every case has random, non-zero A and B (and `lora_B` biases where the option asks for them), so
that a wrong name, scaling or transposition changes the output. Tolerances come from measurement:
over 10 seeds, 4 models and 4 options (160 adapters) the largest disagreements were

- logits, PEFT's adapter loaded here: 2.1e-6 of the largest logit (median 4.3e-7);
- logits, our adapter loaded by PEFT: 1.4e-6 (median 4.1e-7);
- gradients of the adapter weights against PyTorch's: 1.1e-5 relative to the parameter's
  gradient norm, with the limit of 1e-4 that `support/gradients.py` sets for all gradients.

The logits tolerance below is about 5x the measurement; a wrong scaling, name or transposition
changes the logits by order 1.
"""

import json
import shutil
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors import safe_open
from support.gradients import check_gradients
from support.tiny import PROMPT_IDS

pytest.importorskip("peft")
import peft  # noqa: E402
import torch  # noqa: E402

from eqx_zoo.lora import add_adapter, load_adapter, save_adapter, trainable  # noqa: E402
from eqx_zoo.lora.adapt import adapter_layers  # noqa: E402
from lora.helpers import IDS, MODELS, adapted, load, randomise  # noqa: E402
from lora.peft_adapters import (  # noqa: E402
    OPTIONS,
    hf_model,
    make_peft_adapter,
    peft_logits,
    relative_error,
)

# Largest logit difference from PEFT, relative to the largest logit.
LOGITS_TOLERANCE = 1e-5

CASES = [
    pytest.param(name, options, id=f"{name}-{option}")
    for name in MODELS
    for option, options in OPTIONS.items()
]
# PEFT warns about lora_bias on layers without a bias; that is not what these tests are about.
pytestmark = pytest.mark.filterwarnings("ignore:lora_bias=True", "ignore:`lora_bias=True`")


def adapter_arrays(model) -> dict[str, np.ndarray]:
    return {
        f"{name}.{attribute}": np.asarray(getattr(layer, attribute))
        for name, layer in adapter_layers(model).items()
        for attribute in ("lora_A", "lora_B", "lora_B_bias")
        if getattr(layer, attribute) is not None
    }


@pytest.fixture(scope="module")
def llama_adapter(checkpoint, tmp_path_factory):
    """A PEFT adapter directory for tiny-llama, to edit in the refusal tests."""
    directory = tmp_path_factory.mktemp("llama-adapter")
    make_peft_adapter(checkpoint, "tiny-llama", OPTIONS["alpha-not-r"]).save_pretrained(directory)
    return directory


# --- PEFT to eqx-zoo, and back ---


@pytest.mark.parametrize(("name", "options"), CASES)
def test_loads_an_adapter_saved_by_peft(checkpoint, tmp_path, name, options):
    reference = make_peft_adapter(checkpoint, name, options)
    reference.save_pretrained(tmp_path)
    base = load(checkpoint, name)
    model = load_adapter(base, tmp_path)

    expected = peft_logits(reference)
    base_error = relative_error(base(IDS)[0], expected)
    assert base_error > 1e-2, "the adapter must change the output, or this test shows nothing"
    assert relative_error(model(IDS)[0], expected) <= LOGITS_TOLERANCE
    wrapped = {
        n.removeprefix("base_model.model.")
        for n, m in reference.named_modules()
        if isinstance(m, peft.tuners.lora.LoraLayer)
    }
    assert set(adapter_layers(model)) == wrapped


@pytest.mark.parametrize(("name", "options"), CASES)
def test_peft_loads_an_adapter_saved_here(checkpoint, tmp_path, name, options):
    model = randomise(adapted(checkpoint, name, **options)[1])
    save_adapter(model, tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reference = peft.PeftModel.from_pretrained(hf_model(checkpoint, name), tmp_path)
    # PEFT warns about missing or unexpected keys, and about a wrong `fan_in_fan_out`.
    messages = [str(w.message) for w in caught if "lora_bias" not in str(w.message)]
    assert not messages, messages
    # No adapter key is missing or unexpected (missing keys are the base model's, not loaded).
    fresh = peft.get_peft_model(
        hf_model(checkpoint, name), peft.LoraConfig.from_pretrained(tmp_path)
    )
    result = peft.set_peft_model_state_dict(fresh, peft.load_peft_weights(tmp_path))
    assert not [k for k in result.missing_keys if "lora_" in k]
    assert not result.unexpected_keys

    assert relative_error(model(IDS)[0], peft_logits(reference.eval())) <= LOGITS_TOLERANCE


@pytest.mark.parametrize(("name", "options"), CASES)
def test_save_then_load_is_exact(checkpoint, tmp_path, name, options):
    base = load(checkpoint, name)
    model = randomise(adapted(checkpoint, name, **options)[1])
    save_adapter(model, tmp_path)
    again = load_adapter(base, tmp_path)
    before, after = adapter_arrays(model), adapter_arrays(again)
    assert before.keys() == after.keys()
    for key in before:
        assert before[key].dtype == after[key].dtype == np.float32
        assert np.array_equal(before[key], after[key]), key
    for layer, other in zip(
        adapter_layers(model).values(), adapter_layers(again).values(), strict=True
    ):
        assert (layer.alpha, layer.rslora, layer.r) == (other.alpha, other.rslora, other.r)
    assert np.array_equal(np.asarray(model(IDS)[0]), np.asarray(again(IDS)[0]))


def loss(model):
    """Mean next-token cross-entropy, as transformers computes it with labels=ids."""
    logp = jax.nn.log_softmax(model(IDS)[0][:-1].astype(jnp.float32), axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, IDS[1:, None], axis=-1))


@pytest.mark.parametrize(("name", "options"), CASES)
def test_adapter_gradients_match_peft(checkpoint, tmp_path, name, options):
    reference = make_peft_adapter(checkpoint, name, options)
    reference.save_pretrained(tmp_path)
    model = load_adapter(load(checkpoint, name), tmp_path)

    ids = torch.tensor([PROMPT_IDS])
    reference(ids, labels=ids).loss.backward()
    expected = {
        n.removeprefix("base_model.model.").replace(".default", ""): p.grad.numpy()
        for n, p in reference.named_parameters()
        if "lora_" in n
    }
    train, frozen = eqx.partition(model, trainable(model))
    grads = jax.grad(lambda t: loss(eqx.combine(t, frozen)))(train)
    ours = {
        f"{layer_name}.{attribute}.{kind}": np.asarray(getattr(layer, attribute))
        for layer_name, layer in adapter_layers(grads).items()
        for attribute, kind in [("lora_A", "weight"), ("lora_B", "weight"), ("lora_B_bias", "bias")]
        if getattr(layer, attribute) is not None
    }
    # PEFT names the bias `lora_B.bias`, which the line above spells `lora_B_bias.bias`.
    ours = {k.replace("lora_B_bias.bias", "lora_B.bias"): v for k, v in ours.items()}
    check_gradients(f"{name}-{options}", ours, expected)


# --- what the saved files contain ---


@pytest.mark.parametrize("name", MODELS)
def test_saved_files(checkpoint, tmp_path, name):
    model = randomise(adapted(checkpoint, name, r=4, alpha=10, rslora=True, lora_bias=True)[1])
    save_adapter(model, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and config["task_type"] == "CAUSAL_LM"
    assert (config["r"], config["lora_alpha"]) == (4, 10)
    assert config["use_rslora"] is True and config["lora_bias"] is True
    assert config["bias"] == "none" and config["use_dora"] is False
    assert config["fan_in_fan_out"] is (name == "tiny-gpt2")
    assert config["target_modules"] == list(adapter_layers(model))
    with safe_open(tmp_path / "adapter_model.safetensors", "np") as file:
        assert file.metadata() == {"format": "pt"}
        keys = set(file.keys())
        assert all(file.get_tensor(k).dtype == np.float32 for k in keys)
    hf_names = {n.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for n in keys}
    assert hf_names == set(config["target_modules"])
    assert all(k.startswith("base_model.model.") and ".default" not in k for k in keys)
    assert all(k.endswith((".lora_A.weight", ".lora_B.weight", ".lora_B.bias")) for k in keys)


def test_bfloat16_base_model_saves_float32(checkpoint, tmp_path):
    base = load(checkpoint, "tiny-llama", dtype=jnp.bfloat16)
    model = add_adapter(base, r=4, alpha=8, target_modules=["q_proj"], key=jax.random.key(0))
    save_adapter(model, tmp_path)
    with safe_open(tmp_path / "adapter_model.safetensors", "np") as file:
        assert {file.get_tensor(k).dtype for k in file.keys()} == {np.dtype(np.float32)}  # noqa: SIM118


# --- accepted and refused configs ---


def edited(source, destination, **changes):
    """A copy of an adapter directory with `adapter_config.json` fields changed."""
    shutil.copytree(source, destination)
    path = destination / "adapter_config.json"
    config = json.loads(path.read_text())
    config.update(changes)
    path.write_text(json.dumps(config))
    return destination


def test_harmless_config_fields_are_accepted(checkpoint, llama_adapter, tmp_path):
    base = load(checkpoint, "tiny-llama")
    expected = np.asarray(load_adapter(base, llama_adapter)(IDS)[0])
    for i, changes in enumerate(
        [
            {"lora_dropout": 0.1},  # training only
            {"init_lora_weights": False},
            {"init_lora_weights": "gaussian"},
            {"fan_in_fan_out": True},  # decided by the layer type, as in PEFT
            {"exclude_modules": ["nothing"], "layers_to_transform": [0]},
            {"peft_version": "9.9.9", "revision": "main", "inference_mode": False},
        ]
    ):
        path = edited(llama_adapter, tmp_path / str(i), **changes)
        assert np.array_equal(np.asarray(load_adapter(base, path)(IDS)[0]), expected), changes


REFUSED = {
    "use_dora": ({"use_dora": True}, "DoRA"),
    "modules_to_save": ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
    "bias": ({"bias": "all"}, "bias='all'"),
    "bias-lora-only": ({"bias": "lora_only"}, "bias='lora_only'"),
    "rank_pattern": ({"rank_pattern": {"q_proj": 2}}, "rank_pattern"),
    "alpha_pattern": ({"alpha_pattern": {"q_proj": 2}}, "alpha_pattern"),
    "target_parameters": ({"target_parameters": ["experts.up"]}, "target_parameters"),
    "trainable_token_indices": ({"trainable_token_indices": [1]}, "trainable_token_indices"),
    "pissa": ({"init_lora_weights": "pissa"}, "modifies the base weights"),
    "pissa_niter": ({"init_lora_weights": "pissa_niter_4"}, "modifies the base weights"),
    "olora": ({"init_lora_weights": "olora"}, "modifies the base weights"),
    "loftq": ({"init_lora_weights": "loftq", "loftq_config": {"loftq_bits": 4}}, "modifies"),
    "loftq_config": ({"loftq_config": {"loftq_bits": 4}}, "LoftQ"),
    "use_qalora": ({"use_qalora": True}, "QALoRA"),
    "layer_replication": ({"layer_replication": [[0, 1]]}, "layer_replication"),
    "alora": ({"alora_invocation_tokens": [1, 2]}, "alora_invocation_tokens"),
    "ensure_weight_tying": ({"ensure_weight_tying": True}, "ensure_weight_tying"),
    "eva_config": ({"eva_config": {"rho": 2.0}}, "eva_config"),
    "unknown field": ({"brand_new_variant": {"x": 1}}, "brand_new_variant"),
    "other peft type": ({"peft_type": "ADALORA"}, "only LoRA"),
}


@pytest.mark.parametrize(("changes", "message"), REFUSED.values(), ids=REFUSED.keys())
def test_unsupported_config_is_refused(checkpoint, llama_adapter, tmp_path, changes, message):
    path = edited(llama_adapter, tmp_path / "adapter", **changes)
    with pytest.raises(NotImplementedError, match=message):
        load_adapter(load(checkpoint, "tiny-llama"), path)


# --- refused tensors, models and files ---


def retagged(source, destination, rename=None, drop=(), add=None, **config):
    """A copy of an adapter directory whose tensors are renamed, dropped or added."""
    from safetensors.numpy import load_file, save_file

    shutil.copytree(source, destination)
    file = destination / "adapter_model.safetensors"
    tensors = {k: v for k, v in load_file(file).items() if not k.endswith(tuple(drop))}
    tensors = {(rename or (lambda k: k))(k): v for k, v in tensors.items()}
    tensors |= add or {}
    save_file(tensors, file, metadata={"format": "pt"})
    if config:
        path = destination / "adapter_config.json"
        path.write_text(json.dumps(json.loads(path.read_text()) | config))
    return destination


def test_a_tensor_for_a_missing_module_is_named(checkpoint, llama_adapter, tmp_path):
    path = retagged(
        llama_adapter,
        tmp_path / "a",
        rename=lambda k: k.replace("layers.0.self_attn.q_proj", "layers.0.self_attn.query"),
    )
    with pytest.raises(ValueError, match=r"'base_model\.model\.model\.layers\.0\.self_attn\.query"):
        load_adapter(load(checkpoint, "tiny-llama"), path)


def test_a_tensor_for_a_module_that_is_not_linear_is_refused(checkpoint, llama_adapter, tmp_path):
    path = retagged(
        llama_adapter,
        tmp_path / "a",
        rename=lambda k: k.replace("layers.0.self_attn.q_proj", "layers.0.self_attn"),
    )
    with pytest.raises(ValueError, match="no linear layer 'model.layers.0.self_attn'"):
        load_adapter(load(checkpoint, "tiny-llama"), path)


def test_tied_output_head_adapters_are_refused(checkpoint, llama_adapter, tmp_path):
    """tiny-llama's head is tied to the embedding, so it has no layer to adapt."""
    path = retagged(
        llama_adapter,
        tmp_path / "a",
        rename=lambda k: k.replace("model.layers.0.self_attn.q_proj", "lm_head"),
    )
    with pytest.raises(ValueError, match="no linear layer 'lm_head'"):
        load_adapter(load(checkpoint, "tiny-llama"), path)


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.self_attn.q_proj.lora_magnitude_vector",  # DoRA
        "model.embed_tokens.lora_embedding_A",
        "lm_head.weight",  # modules_to_save
        "model.layers.0.self_attn.q_proj.lora_A.default.weight",  # an adapter-name segment
    ],
)
def test_other_tensors_are_refused(checkpoint, llama_adapter, tmp_path, key):
    path = retagged(llama_adapter, tmp_path / "a", add={f"base_model.model.{key}": np.zeros(3)})
    with pytest.raises(ValueError, match=f"{key.replace('.', '[.]')}.*is not a LoRA weight"):
        load_adapter(load(checkpoint, "tiny-llama"), path)


def test_tensors_without_the_peft_prefix_are_refused(checkpoint, llama_adapter, tmp_path):
    path = retagged(llama_adapter, tmp_path / "a", rename=lambda k: k.removeprefix("base_model."))
    with pytest.raises(ValueError, match="is not a LoRA weight"):
        load_adapter(load(checkpoint, "tiny-llama"), path)


def test_an_incomplete_pair_is_refused(checkpoint, llama_adapter, tmp_path):
    path = retagged(
        llama_adapter, tmp_path / "a", drop=("layers.0.self_attn.q_proj.lora_B.weight",)
    )
    with pytest.raises(ValueError, match="q_proj: the adapter has no lora_B.weight"):
        load_adapter(load(checkpoint, "tiny-llama"), path)


def test_the_rank_must_agree_with_the_config(checkpoint, llama_adapter, tmp_path):
    path = edited(llama_adapter, tmp_path / "a", r=8)
    with pytest.raises(ValueError, match="rank 4 in the tensors, but the config says r=8"):
        load_adapter(load(checkpoint, "tiny-llama"), path)


def test_lora_bias_must_agree_with_the_tensors(checkpoint, llama_adapter, tmp_path):
    with pytest.raises(ValueError, match="no lora_B.bias"):
        load_adapter(
            load(checkpoint, "tiny-llama"), edited(llama_adapter, tmp_path / "a", lora_bias=True)
        )
    biased = tmp_path / "b"
    make_peft_adapter(checkpoint, "tiny-llama", OPTIONS["lora-bias"]).save_pretrained(biased)
    path = edited(biased, tmp_path / "c", lora_bias=False)
    with pytest.raises(ValueError, match="lora_bias is false"):
        load_adapter(load(checkpoint, "tiny-llama"), path)


def test_mis_shaped_tensors_are_refused(checkpoint, llama_adapter, tmp_path):
    other = tmp_path / "other"
    make_peft_adapter(checkpoint, "tiny-gpt2", OPTIONS["plain"]).save_pretrained(other)
    path = retagged(
        other,
        tmp_path / "a",
        drop=("h.0.attn.c_proj.lora_A.weight", "h.0.attn.c_proj.lora_B.weight"),
        rename=lambda k: k.replace("transformer.h.0.attn.c_attn", "transformer.h.0.attn.c_proj"),
    )
    with pytest.raises(ValueError, match="c_proj: lora_B must be"):
        load_adapter(load(checkpoint, "tiny-gpt2"), path)


def test_a_missing_weights_file_is_reported(checkpoint, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        load_adapter(load(checkpoint, "tiny-llama"), tmp_path)


def test_loading_refuses_what_adding_refuses(checkpoint, llama_adapter):
    adapted_model = load_adapter(load(checkpoint, "tiny-llama"), llama_adapter)
    with pytest.raises(ValueError, match="already has an adapter"):
        load_adapter(adapted_model, llama_adapter)
    for name, message in [
        ("tiny-qwen35", "Qwen3_5.*prefix"),
        ("tiny-qwen3-moe", "mixture-of-experts"),
    ]:
        with pytest.raises(NotImplementedError, match=message):
            load_adapter(load(checkpoint, name), llama_adapter)
        with pytest.raises(NotImplementedError, match=message):
            adapted(checkpoint, name)


# --- saving ---


def test_saving_needs_an_adapter(checkpoint, tmp_path):
    with pytest.raises(ValueError, match="no adapter"):
        save_adapter(load(checkpoint, "tiny-llama"), tmp_path)


def test_saving_needs_one_rank_alpha_and_scaling(checkpoint, tmp_path):
    model = adapted(checkpoint, "tiny-llama", target_modules=["q_proj", "v_proj"])[1]
    v = model.model.layers[0].self_attn.v_proj

    def with_v(**changes):
        """The model with layer 0's `v_proj` rebuilt with some arguments changed."""
        arguments = {
            "lora_A": v.lora_A,
            "lora_B": v.lora_B,
            "alpha": v.alpha,
            "rslora": v.rslora,
            "lora_B_bias": v.lora_B_bias,
        } | changes
        layer = type(v)(v.base, arguments.pop("lora_A"), arguments.pop("lora_B"), **arguments)
        return eqx.tree_at(lambda m: m.model.layers[0].self_attn.v_proj, model, layer)

    with pytest.raises(ValueError, match="different alphas"):
        save_adapter(with_v(alpha=3.0), tmp_path)
    with pytest.raises(ValueError, match="different rslora"):
        save_adapter(with_v(rslora=True), tmp_path)
    with pytest.raises(ValueError, match="different ranks"):
        save_adapter(with_v(lora_A=v.lora_A[:2], lora_B=v.lora_B[:, :2]), tmp_path)
    with pytest.raises(ValueError, match="only some layers have a lora_B bias"):
        save_adapter(with_v(lora_B_bias=jnp.zeros(v.lora_B.shape[0])), tmp_path)


def test_gpt_neox_head_adapters_are_not_saved(checkpoint, tmp_path):
    model = adapted(checkpoint, "tiny-gpt-neox", target_modules=["embed_out"])[1]
    with pytest.raises(NotImplementedError, match="embed_out.*lm_head"):
        save_adapter(model, tmp_path)


def test_gpt_neox_head_adapters_load_under_either_name(checkpoint, tmp_path):
    """Adapters made with transformers 4 name the head `embed_out`; with 5, `lm_head`."""
    from safetensors.flax import save_file

    layer = randomise(
        adapted(checkpoint, "tiny-gpt-neox", target_modules=["embed_out"])[1]
    ).embed_out
    for head in ("embed_out", "lm_head"):
        directory = tmp_path / head
        directory.mkdir()
        save_file(
            {
                f"base_model.model.{head}.lora_A.weight": layer.lora_A,
                f"base_model.model.{head}.lora_B.weight": layer.lora_B,
            },
            str(directory / "adapter_model.safetensors"),
        )
        (directory / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
        loaded = load_adapter(load(checkpoint, "tiny-gpt-neox"), directory)
        assert np.array_equal(np.asarray(loaded.embed_out.lora_A), np.asarray(layer.lora_A))


# --- the lora_bias warning ---


def test_lora_bias_warns_for_layers_without_a_bias(checkpoint, tmp_path):
    with pytest.warns(UserWarning, match=r"lora_bias=True, but \d+ targeted layers have no bias"):
        model = adapted(checkpoint, "tiny-llama", lora_bias=True)[1]
    save_adapter(model, tmp_path)
    with pytest.warns(UserWarning, match="targeted layers have no bias"):
        load_adapter(load(checkpoint, "tiny-llama"), tmp_path)


@pytest.mark.parametrize("name", ["tiny-gpt2", "tiny-gpt-neox"])
def test_lora_bias_does_not_warn_when_every_layer_has_a_bias(checkpoint, name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapted(checkpoint, name, lora_bias=True, target_modules=["c_attn", "query_key_value"])
