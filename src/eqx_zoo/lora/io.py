"""Loading and saving adapters in PEFT's format."""

import json
import os
import re
from pathlib import Path

import jax.numpy as jnp
from huggingface_hub import snapshot_download
from safetensors import safe_open

from eqx_zoo.lora.adapt import (
    _check_model,
    _get,
    _insert,
    _linear_modules,
    _warn_lora_bias_without_bias,
    _wrap,
    adapter_layers,
)
from eqx_zoo.lora.layers import LoRAConv1D
from eqx_zoo.models.gpt_neox import GPTNeoX

CONFIG_FILE = "adapter_config.json"
WEIGHTS_FILE = "adapter_model.safetensors"
_PREFIX = "base_model.model."
_KEY = re.compile(r"(?P<module>.+)\.(?P<which>lora_A\.weight|lora_B\.weight|lora_B\.bias)")

# `adapter_config.json` fields that don't change what an adapter computes: bookkeeping, the
# choice of target layers (we adapt the layers that have weights), and options that only act
# during training.
_HARMLESS = {
    "peft_type",  # checked separately
    "task_type",
    "base_model_name_or_path",
    "revision",
    "auto_mapping",
    "peft_version",
    "inference_mode",
    "r",  # checked against the tensors
    "lora_alpha",
    "use_rslora",
    "lora_bias",
    "lora_dropout",  # training only
    "target_modules",
    "exclude_modules",
    "layers_to_transform",
    "layers_pattern",
    "fan_in_fan_out",  # we decide from the layer type, as PEFT does
    "init_lora_weights",  # checked separately
    "bias",  # checked separately
    "megatron_core",
    "qalora_group_size",
    "runtime_config",
}
# Fields whose non-default values mean something we don't implement.
_REFUSED = {
    "use_dora": "DoRA",
    "modules_to_save": "modules_to_save (fully trained modules stored in the adapter)",
    "rank_pattern": "per-module ranks (rank_pattern)",
    "alpha_pattern": "per-module alphas (alpha_pattern)",
    "target_parameters": "target_parameters (adapted nn.Parameters, such as MoE experts)",
    "trainable_token_indices": "trainable_token_indices",
    "loftq_config": "LoftQ (loftq_config)",
    "use_qalora": "QALoRA",
    "layer_replication": "layer_replication",
    "alora_invocation_tokens": "activated LoRA (alora_invocation_tokens)",
    "ensure_weight_tying": "ensure_weight_tying",
}
# Initialisations that leave the base weights alone: only the starting point of training.
_SAFE_INITIALISATIONS = {True, False, "gaussian", "eva", "orthogonal", "nora"}


def _empty(value) -> bool:
    return value is None or value is False or value == {} or value == [] or value == ""


def _validate_config(config: dict) -> tuple[float, bool, bool]:
    """Check an `adapter_config.json` for features we don't implement.

    Returns:
        `lora_alpha`, `use_rslora` and `lora_bias`.

    Raises:
        NotImplementedError: If the adapter uses a feature we don't implement, or a field
            this version doesn't know about is set.
    """
    if str(config.get("peft_type", "LORA")).upper() != "LORA":
        raise NotImplementedError(f"only LoRA adapters are supported, not {config['peft_type']}")
    if config.get("bias", "none") != "none":
        raise NotImplementedError(
            f"bias={config['bias']!r}: adapters that also train the base layers' biases are "
            "not supported"
        )
    init = config.get("init_lora_weights", True)
    if not (isinstance(init, bool | str) and init in _SAFE_INITIALISATIONS):
        raise NotImplementedError(
            f"init_lora_weights={init!r} also modifies the base weights, which this adapter "
            "was trained against; only initialisations that leave them alone are supported"
        )
    for field, what in _REFUSED.items():
        if not _empty(config.get(field)):
            raise NotImplementedError(f"{what} is not supported (adapter_config.json: {field})")
    for field, value in config.items():
        if field not in _HARMLESS and field not in _REFUSED and not _empty(value):
            raise NotImplementedError(
                f"adapter_config.json sets {field!r}, which this version doesn't support; "
                "it may change what the adapter computes"
            )
    return (
        float(config.get("lora_alpha", 8)),
        bool(config.get("use_rslora", False)),
        bool(config.get("lora_bias", False)),
    )


def _resolve(repo_or_path: str | os.PathLike[str], revision: str | None) -> Path:
    if Path(repo_or_path).is_dir():
        path = Path(repo_or_path)
    else:
        path = Path(
            snapshot_download(
                str(repo_or_path), revision=revision, allow_patterns=[CONFIG_FILE, WEIGHTS_FILE]
            )
        )
    for file in (CONFIG_FILE, WEIGHTS_FILE):
        if not (path / file).exists():
            raise FileNotFoundError(f"{repo_or_path}: no {file}")
    return path


def load_adapter(model, repo_or_path: str | os.PathLike[str], *, revision: str | None = None):
    """Add a LoRA adapter saved by PEFT (or by `save_adapter`) to a model.

    Layers are injected from the adapter's tensors: each module that has `lora_A` and `lora_B`
    weights is wrapped, with its rank taken from their shapes and `lora_alpha`, `use_rslora` and
    `lora_bias` from `adapter_config.json`. The adapter is held in float32, as PEFT holds it.
    `lora_dropout` is accepted: it doesn't affect inference. Whether a layer is transposed
    (`fan_in_fan_out`) follows from its type, as in PEFT, not from the config.

    Args:
        model: A `CausalLM`, `GPT2` or `GPTNeoX` without an adapter.
        repo_or_path: Hub repository, or a local directory containing `adapter_config.json`
            and `adapter_model.safetensors`; only those two files are downloaded.
        revision: Optional git revision of a Hub repository; ignored for directories.

    Returns:
        The model with the adapter.

    Raises:
        NotImplementedError: If the model can't take an adapter (see `add_adapter`), or the
            adapter uses DoRA, `modules_to_save`, trained biases, per-module ranks or alphas,
            `target_parameters`, `trainable_token_indices`, an initialisation that modified
            the base weights, or any other field that this version doesn't know.
        ValueError: If the model already has an adapter, a tensor is not a LoRA weight or
            names a module that is not a linear layer of the model, a pair of weights is
            incomplete or mis-shaped, or the tensors disagree with the config (`r`, `lora_bias`).
        FileNotFoundError: If a file is missing (for example an adapter saved only as `.bin`).
    """
    _check_model(model)
    path = _resolve(repo_or_path, revision)
    config = json.loads((path / CONFIG_FILE).read_text())
    alpha, rslora, lora_bias = _validate_config(config)

    modules = _linear_modules(model)
    if isinstance(model, GPTNeoX):  # the output head is `embed_out`, or `lm_head` in transformers 5
        modules["lm_head"] = modules["embed_out"]

    weights: dict[str, dict] = {}
    with safe_open(path / WEIGHTS_FILE, framework="flax") as file:
        for key in file.keys():  # noqa: SIM118
            match = _KEY.fullmatch(key.removeprefix(_PREFIX)) if key.startswith(_PREFIX) else None
            if match is None:
                raise ValueError(
                    f"adapter tensor {key!r} is not a LoRA weight: expected "
                    f"'{_PREFIX}<module>.lora_A.weight' (or lora_B.weight, lora_B.bias); other "
                    "adapter types, DoRA magnitudes and saved modules are not supported"
                )
            if match["module"] not in modules:
                raise ValueError(
                    f"adapter tensor {key!r}: the model has no linear layer {match['module']!r}"
                )
            tensor = file.get_tensor(key)
            weights.setdefault(match["module"], {})[match["which"]] = tensor.astype(jnp.float32)

    wrapped = {}
    for name, tensors in weights.items():
        for needed in ("lora_A.weight", "lora_B.weight"):
            if needed not in tensors:
                raise ValueError(f"{name}: the adapter has no {needed}")
        bias = tensors.get("lora_B.bias")
        if lora_bias and bias is None:
            raise ValueError(f"{name}: lora_bias is set in the config, but there is no lora_B.bias")
        if bias is not None and not lora_bias:
            raise ValueError(f"{name}: has a lora_B.bias, but the config's lora_bias is false")
        r = tensors["lora_A.weight"].shape[0]
        if "r" in config and r != config["r"]:
            raise ValueError(
                f"{name}: rank {r} in the tensors, but the config says r={config['r']}"
            )
        try:
            wrapped[modules[name]] = _wrap(
                _get(model, modules[name]),
                tensors["lora_A.weight"],
                tensors["lora_B.weight"],
                bias,
                alpha=alpha,
                rslora=rslora,
            )
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    if not wrapped:
        raise ValueError(f"{path / WEIGHTS_FILE} contains no LoRA weights")
    _warn_lora_bias_without_bias(model, wrapped)
    return _insert(model, wrapped)


def save_adapter(model, directory: str | os.PathLike[str]) -> None:
    """Save a model's adapter in PEFT's format, so that PEFT can load it.

    Writes `adapter_model.safetensors` (float32, with PEFT's tensor names: Hugging Face module
    names under `base_model.model.`, without an adapter-name segment) and `adapter_config.json`
    (`peft_type` `LORA`, `task_type` `CAUSAL_LM`, `target_modules` as the full module names).
    Only one `r` and one `lora_alpha` are written, so every layer must share them.

    Args:
        model: A model with an adapter.
        directory: Destination directory; created if needed.

    Raises:
        ValueError: If the model has no adapter, or its layers differ in rank, alpha, scaling
            kind (`rslora`) or whether they have a `lora_B` bias.
        NotImplementedError: If the adapter targets GPT-NeoX's output head, whose Hugging Face
            name differs between transformers versions (`embed_out` or `lm_head`).
    """
    from safetensors.flax import save_file

    layers = adapter_layers(model)
    if not layers:
        raise ValueError("the model has no adapter to save")
    if "embed_out" in layers:
        raise NotImplementedError(
            "the adapter targets GPT-NeoX's output head, whose name in Hugging Face is "
            "'embed_out' in transformers 4 but 'lm_head' in transformers 5, so PEFT might not "
            "find it; adapt another layer"
        )
    first = next(iter(layers.values()))
    for attribute, what in [
        ("r", "ranks"),
        ("alpha", "alphas"),
        ("rslora", "rslora settings"),
    ]:
        values = {getattr(layer, attribute) for layer in layers.values()}
        if len(values) > 1:
            raise ValueError(
                f"layers have different {what} ({sorted(values)}), but the PEFT config this "
                "writes has a single value"
            )
    if len({layer.lora_B_bias is None for layer in layers.values()}) > 1:
        raise ValueError("only some layers have a lora_B bias, but PEFT's lora_bias is global")

    tensors = {}
    for name, layer in layers.items():
        tensors[f"{_PREFIX}{name}.lora_A.weight"] = layer.lora_A.astype(jnp.float32)
        tensors[f"{_PREFIX}{name}.lora_B.weight"] = layer.lora_B.astype(jnp.float32)
        if layer.lora_B_bias is not None:
            tensors[f"{_PREFIX}{name}.lora_B.bias"] = layer.lora_B_bias.astype(jnp.float32)

    hf_config = getattr(model, "hf_config", None)
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": (json.loads(hf_config).get("_name_or_path") or None)
        if hf_config
        else None,
        "revision": None,
        "inference_mode": True,
        "r": first.r,
        "lora_alpha": first.alpha,
        "lora_dropout": 0.0,
        "use_rslora": first.rslora,
        "lora_bias": first.lora_B_bias is not None,
        "bias": "none",
        "use_dora": False,
        "init_lora_weights": True,
        "fan_in_fan_out": isinstance(first, LoRAConv1D),
        "target_modules": list(layers),
        "exclude_modules": None,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
        "trainable_token_indices": None,
        # Not `target_parameters` (PEFT 0.17): older PEFT warns about fields it doesn't know, and
        # the default (none) applies when it is missing.
    }
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / WEIGHTS_FILE), metadata={"format": "pt"})
    (path / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
