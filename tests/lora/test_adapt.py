"""`add_adapter` and `trainable`, on the tiny models, against PEFT's targeting."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from support.tiny import build_encoder

pytest.importorskip("peft")

# Several tests ask for lora_bias on models whose layers have no bias; the warning is tested apart.
pytestmark = pytest.mark.filterwarnings("ignore:lora_bias=True")

import peft  # noqa: E402
import transformers  # noqa: E402

from eqx_zoo import Encoder  # noqa: E402
from eqx_zoo.lora import LoRAConv1D, LoRALinear, add_adapter, trainable  # noqa: E402
from eqx_zoo.lora.adapt import adapter_layers  # noqa: E402
from lora.helpers import IDS, MODELS, adapted, leaves, load  # noqa: E402

# Per model: target_modules for a list (suffix matching), a regex, and an exact full name.
TARGETS = {
    "tiny-llama": {
        "list": ["q_proj", "v_proj"],
        "regex": r"model\.layers\.0\..*_proj",
        "full-name": ["model.layers.1.self_attn.k_proj"],
    },
    "tiny-qwen2": {  # q/k/v have biases
        "list": ["k_proj", "down_proj"],
        "regex": r".*\.(self_attn\.o_proj|mlp\.up_proj)",
        "full-name": ["model.layers.0.mlp.gate_proj"],
    },
    "tiny-gpt2": {  # `c_proj` is in both the attention and the MLP
        "list": ["c_attn", "c_proj"],
        "regex": r"transformer\.h\.[0-9]+\.mlp\.c_(fc|proj)",
        "full-name": ["transformer.h.0.attn.c_attn"],
    },
    "tiny-gpt-neox": {  # `dense` must not match `dense_h_to_4h`
        "list": ["query_key_value", "dense"],
        "regex": r"gpt_neox\.layers\.1\.mlp\.dense_.*",  # not `mlp.act`
        "full-name": ["gpt_neox.layers.1.attention.dense"],
    },
}
CASES = [
    pytest.param(name, target, id=f"{name}-{kind}")
    for name in MODELS
    for kind in ("list", "regex", "full-name")
    for target in [TARGETS[name][kind]]
] + [
    pytest.param(name, spelling, id=f"{name}-{spelling}")
    for name in MODELS
    for spelling in ("all-linear", "ALL-LINEAR")
]


def peft_wrapped(checkpoint, name, target_modules) -> set[str]:
    """The module names PEFT wraps in a LoRA layer, or the error it raises."""
    hf = transformers.AutoModelForCausalLM.from_pretrained(checkpoint(name))
    wrapped = peft.get_peft_model(hf, peft.LoraConfig(r=4, target_modules=target_modules))
    return {
        n.removeprefix("base_model.model.")
        for n, m in wrapped.named_modules()
        if isinstance(m, peft.tuners.lora.LoraLayer)
    }


@pytest.mark.parametrize(("name", "target"), CASES)
def test_wraps_the_modules_peft_wraps(checkpoint, name, target):
    expected = peft_wrapped(checkpoint, name, target)
    assert expected, "the case must target something"
    _, model = adapted(checkpoint, name, target_modules=target)
    assert set(adapter_layers(model)) == expected


@pytest.mark.parametrize("name", MODELS)
def test_all_linear_leaves_out_the_output_head(checkpoint, name):
    """The untied heads (`lm_head`, `embed_out`) are Linear layers that all-linear skips."""
    base, model = adapted(checkpoint, name)
    wrapped = adapter_layers(model)
    assert not any(n in ("lm_head", "embed_out") for n in wrapped)
    if name in ("tiny-qwen2", "tiny-gpt-neox"):  # untied heads exist and are left unwrapped
        head = model.lm_head if name == "tiny-qwen2" else model.embed_out
        assert isinstance(head, eqx.nn.Linear)


@pytest.mark.parametrize("name", MODELS)
def test_wrong_boundary_matches_nothing(checkpoint, name):
    """ "proj" is a suffix of "q_proj" but not after a dot, so PEFT and we match nothing."""
    target = ["proj"] if name != "tiny-gpt-neox" else ["h_to_4h"]
    with pytest.raises(ValueError, match="not found"):
        peft_wrapped(checkpoint, name, target)
    with pytest.raises(ValueError, match="matched no module"):
        adapted(checkpoint, name, target_modules=target)


@pytest.mark.parametrize("name", MODELS)
def test_a_regex_must_match_the_whole_name(checkpoint, name):
    """A pattern that only matches the start of a name (`re.match`) selects nothing."""
    target = {
        "tiny-llama": r"model\.layers\.0\.self_attn\.[qk]_pro",
        "tiny-qwen2": r"model\.layers\.0\.self_attn\.[qk]_pro",
        "tiny-gpt2": r"transformer\.h\.0\.attn\.c_att",
        "tiny-gpt-neox": r"gpt_neox\.layers\.0\.attention\.query_key_valu",
    }[name]
    with pytest.raises(ValueError, match="not found"):
        peft_wrapped(checkpoint, name, target)
    with pytest.raises(ValueError, match="matched no module"):
        adapted(checkpoint, name, target_modules=target)


@pytest.mark.parametrize("name", MODELS)
def test_a_fresh_adapter_changes_nothing(checkpoint, name):
    base, model = adapted(checkpoint, name, lora_bias=True)
    expected, got = base(IDS)[0], model(IDS)[0]
    assert got.dtype == expected.dtype
    assert np.array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("name", MODELS)
def test_initialisation(checkpoint, name):
    _, model = adapted(checkpoint, name, r=8, alpha=16, lora_bias=True)
    layers = adapter_layers(model)
    first = None
    for layer in layers.values():
        in_features = layer.lora_A.shape[1]
        assert layer.lora_A.dtype == layer.lora_B.dtype == layer.lora_B_bias.dtype == jnp.float32
        assert layer.lora_A.shape[0] == 8 and layer.scaling == 2.0
        assert not np.any(np.asarray(layer.lora_B)) and not np.any(np.asarray(layer.lora_B_bias))
        bound = in_features**-0.5
        assert np.abs(np.asarray(layer.lora_A)).max() <= bound
        assert np.abs(np.asarray(layer.lora_A)).max() >= 0.9 * bound  # uniform up to the bound
        if first is not None and first.shape == layer.lora_A.shape:
            assert not np.array_equal(np.asarray(first), np.asarray(layer.lora_A))  # split keys
        first = layer.lora_A


def test_initialisation_matches_pepts_bound(checkpoint):
    """PEFT's A is Kaiming-uniform with a = sqrt(5): uniform on ±1/sqrt(in_features)."""
    hf = transformers.AutoModelForCausalLM.from_pretrained(checkpoint("tiny-llama"))
    wrapped = peft.get_peft_model(hf, peft.LoraConfig(r=8, target_modules=["q_proj"]))
    layer = wrapped.base_model.model.model.layers[0].self_attn.q_proj
    theirs = layer.lora_A["default"].weight.detach().numpy()
    _, model = adapted(checkpoint, "tiny-llama", r=8, target_modules=["q_proj"])
    ours = np.asarray(model.model.layers[0].self_attn.q_proj.lora_A)
    assert ours.shape == theirs.shape
    assert abs(np.abs(ours).max() - np.abs(theirs).max()) < 0.05 * ours.shape[1] ** -0.5
    assert abs(ours.std() - theirs.std()) < 0.1 * theirs.std()  # both ~ bound / sqrt(3)


def test_rslora_and_wrapper_types(checkpoint):
    _, llama = adapted(checkpoint, "tiny-llama", r=4, alpha=8, rslora=True)
    assert {layer.scaling for layer in adapter_layers(llama).values()} == {4.0}
    assert all(type(x) is LoRALinear for x in adapter_layers(llama).values())
    assert all(x.lora_B_bias is None for x in adapter_layers(llama).values())
    _, gpt2 = adapted(checkpoint, "tiny-gpt2")
    assert all(type(x) is LoRAConv1D for x in adapter_layers(gpt2).values())


def test_works_on_bfloat16_models(checkpoint):
    base = load(checkpoint, "tiny-llama", dtype=jnp.bfloat16)
    model = add_adapter(base, r=4, alpha=8, target_modules="all-linear", key=jax.random.key(0))
    layer = model.model.layers[0].self_attn.q_proj
    assert layer.base.weight.dtype == jnp.bfloat16 and layer.lora_A.dtype == jnp.float32
    assert np.array_equal(np.asarray(model(IDS)[0]), np.asarray(base(IDS)[0]))


# --- trainable ---


@pytest.mark.parametrize("name", MODELS)
def test_mask_is_true_exactly_at_the_adapter(checkpoint, name):
    base, model = adapted(checkpoint, name, lora_bias=True)
    mask = leaves(trainable(model))
    params = leaves(model)
    assert mask.keys() == params.keys()
    adapter = (".lora_A", ".lora_B", ".lora_B_bias")
    for path in params:
        assert mask[path] is path.endswith(adapter), path
    n_layers = len(adapter_layers(model))
    assert sum(mask.values()) == 3 * n_layers
    # Without a bias, only A and B.
    plain = adapted(checkpoint, name)[1]
    assert sum(leaves(trainable(plain)).values()) == 2 * n_layers
    # The base model has no adapter.
    assert not any(leaves(trainable(base)).values())


def step(model, lr=0.1):
    """One gradient step on the adapter, through `eqx.partition`, returning the grads too."""
    train, frozen = eqx.partition(model, trainable(model))

    def loss(train):
        logits = eqx.combine(train, frozen)(IDS)[0]
        return jnp.mean(logits.astype(jnp.float32) ** 2)

    grads = jax.grad(loss)(train)
    updated = jax.tree.map(lambda p, g: p - lr * g, train, grads)
    return eqx.combine(updated, frozen), grads


@pytest.mark.parametrize("name", MODELS)
def test_gradient_step_changes_only_the_adapter(checkpoint, name):
    _, model = adapted(checkpoint, name, lora_bias=True)
    new, grads = step(model)
    before, after = leaves(model), leaves(new)
    adapter = (".lora_A", ".lora_B", ".lora_B_bias")
    for path in before:
        same = np.array_equal(np.asarray(before[path]), np.asarray(after[path]))
        if path.endswith(".lora_B") or path.endswith(".lora_B_bias"):
            assert not same, f"{path} should have moved"
        else:
            # Everything else is bit-identical; so is A, whose gradient is zero while B is.
            assert same, f"{path} changed"
    # Only the adapter has gradients.
    assert all(p.endswith(adapter) for p in leaves(eqx.filter(grads, eqx.is_array)))


@pytest.mark.parametrize("name", MODELS)
def test_gradient_of_a_is_zero_while_b_is_zero(checkpoint, name):
    _, model = adapted(checkpoint, name)
    _, grads = step(model)
    for path, g in leaves(eqx.filter(grads, eqx.is_array)).items():
        if path.endswith(".lora_A"):
            assert not np.any(np.asarray(g)), path
        else:
            assert np.any(np.asarray(g)), f"{path} should have a gradient"
    # Once B is non-zero, A learns too.
    layers = adapter_layers(model)
    nonzero = eqx.tree_at(
        lambda m: [x.lora_B for x in adapter_layers(m).values()],
        model,
        [jax.random.normal(jax.random.key(1), x.lora_B.shape) * 0.1 for x in layers.values()],
    )
    _, grads = step(nonzero)
    for path, g in leaves(eqx.filter(grads, eqx.is_array)).items():
        assert np.any(np.asarray(g)), f"{path} should have a gradient"


# --- refusals ---


def test_dropout_is_refused(checkpoint):
    with pytest.raises(NotImplementedError, match="dropout=0.1.*PRNG key"):
        adapted(checkpoint, "tiny-llama", dropout=0.1)
    adapted(checkpoint, "tiny-llama", dropout=0.0)  # zero is fine


def test_a_second_adapter_is_refused(checkpoint):
    _, model = adapted(checkpoint, "tiny-llama")
    with pytest.raises(ValueError, match="already has an adapter"):
        add_adapter(model, r=4, alpha=8, target_modules="all-linear", key=jax.random.key(1))


def test_no_match_names_the_modules_that_exist(checkpoint):
    with pytest.raises(ValueError, match=r"matched no module.*q_proj.*model\.layers\.0"):
        adapted(checkpoint, "tiny-llama", target_modules=["query"])
    with pytest.raises(ValueError, match=r"matched no module.*c_attn.*transformer\.h\.0"):
        adapted(checkpoint, "tiny-gpt2", target_modules=r"model\..*")


def test_non_linear_targets_are_refused(checkpoint):
    with pytest.raises(ValueError, match="'model.layers.0.self_attn' is a Attention"):
        adapted(checkpoint, "tiny-llama", target_modules=["self_attn"])
    with pytest.raises(ValueError, match="RMSNorm"):
        adapted(checkpoint, "tiny-llama", target_modules=["input_layernorm"])
    with pytest.raises(NotImplementedError, match="embedding"):
        adapted(checkpoint, "tiny-llama", target_modules=["embed_tokens"])


def test_invalid_rank_is_refused(checkpoint):
    with pytest.raises(ValueError, match="positive"):
        adapted(checkpoint, "tiny-llama", r=0)


def test_qwen35_is_refused(checkpoint):
    with pytest.raises(NotImplementedError, match="Qwen3_5.*prefix"):
        adapted(checkpoint, "tiny-qwen35")


def test_other_model_types_are_refused(tmp_path):
    build_encoder("tiny-bert", tmp_path)
    encoder = Encoder.from_pretrained(tmp_path)
    with pytest.raises(NotImplementedError, match="CausalLM, GPT2 and GPTNeoX, not Encoder"):
        add_adapter(encoder, r=4, alpha=8, target_modules="all-linear", key=jax.random.key(0))
