"""Read sentence-transformers pipelines, and apply their pooling and normalisation."""

import json
from pathlib import Path

from jaxtyping import Array, Bool, Float

from eqx_zoo.layers import cls_pool, l2_normalize, last_token_pool, mean_pool

# sentence-transformers pooling modes -> our names for them.
_MODES = {"cls_token": "cls", "mean_tokens": "mean", "lasttoken": "last_token"}
_POOL = {"cls": cls_pool, "mean": mean_pool, "last_token": last_token_pool}
_FILES = ("modules.json", "config_sentence_transformers.json", "sentence_bert_config.json")


def read_files(path: Path) -> tuple[tuple[str, str], ...]:
    """A checkpoint's sentence-transformers configuration files, as (relative path, text)."""
    names = [name for name in _FILES if (path / name).exists()]
    names += sorted(str(p.relative_to(path)) for p in path.glob("*/config.json"))
    return tuple((name, (path / name).read_text()) for name in names)


def write_files(path: Path, files: tuple[tuple[str, str], ...]) -> None:
    """Write files read by `read_files` back into a checkpoint directory."""
    for name, text in files:
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)


def read_pipeline(path: Path) -> tuple[str | None, bool]:
    """Read the pooling and normalisation of a sentence-transformers checkpoint.

    Returns `(None, False)` for checkpoints without a `modules.json`.

    Raises:
        NotImplementedError: If the pipeline uses a pooling mode or module not supported yet.
    """
    modules_file = path / "modules.json"
    if not modules_file.exists():
        return None, False

    pooling, normalize = None, False
    for module in json.loads(modules_file.read_text()):
        kind = module["type"].rsplit(".", 1)[-1]
        if kind == "Transformer":
            continue
        if kind == "Pooling":
            config = json.loads((path / module["path"] / "config.json").read_text())
            modes = [
                k.removeprefix("pooling_mode_")
                for k, v in config.items()
                if k.startswith("pooling_mode_") and v
            ]
            if len(modes) != 1 or modes[0] not in _MODES:
                raise NotImplementedError(
                    f"sentence-transformers pooling {modes} is not supported yet"
                )
            pooling = _MODES[modes[0]]
        elif kind == "Normalize":
            normalize = True
        else:
            raise NotImplementedError(f"sentence-transformers module {kind!r} is not supported yet")
    return pooling, normalize


def pool(
    hidden: Float[Array, "seq dim"], mask: Bool[Array, " seq"], pooling: str, normalize: bool
) -> Float[Array, " dim"]:
    """Pool token states into one embedding, normalising it if requested."""
    pooled = _POOL[pooling](hidden, mask)
    return l2_normalize(pooled) if normalize else pooled


def _pipeline_files(pooling: str, normalize: bool, dim: int) -> tuple[tuple[str, str], ...]:
    """Sentence-transformers files describing a pooling and normalisation, for saving."""
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]
    if normalize:
        modules.append(
            {
                "idx": 2,
                "name": "2",
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            }
        )
    mode = {ours: theirs for theirs, ours in _MODES.items()}[pooling]
    config = {"word_embedding_dimension": dim} | {
        f"pooling_mode_{m}": m == mode
        for m in ("cls_token", "mean_tokens", "max_tokens", "mean_sqrt_len_tokens", "lasttoken")
    }
    return (
        ("modules.json", json.dumps(modules, indent=2) + "\n"),
        ("1_Pooling/config.json", json.dumps(config, indent=2) + "\n"),
    )


def load_pipeline(
    path: Path, pooling: str | None, normalize: bool | None, dim: int
) -> tuple[str | None, bool, tuple[tuple[str, str], ...]]:
    """The pooling and normalisation to use, and the sentence-transformers files to keep.

    A checkpoint's own sentence-transformers pipeline takes precedence; `pooling` and
    `normalize` may restate it but not contradict it. For a checkpoint without one, they
    supply it, and matching sentence-transformers files are generated so that a saved copy
    loads with the same pooling.

    Args:
        path: Checkpoint directory.
        pooling: `"cls"`, `"mean"` or `"last_token"`, or `None` if not specified.
        normalize: Whether to scale embeddings to unit length, or `None` if not specified.
        dim: Hidden size, recorded in a generated pooling configuration.

    Returns:
        The pooling (or `None` if there is none), whether to normalise, and the files.

    Raises:
        ValueError: If `pooling` is unknown, or the arguments contradict the checkpoint's
            pipeline.
    """
    if pooling is not None and pooling not in _POOL:
        raise ValueError(f"pooling must be one of {sorted(_POOL)}, not {pooling!r}")
    files = read_files(path)
    if (path / "modules.json").exists():
        stored_pooling, stored_normalize = read_pipeline(path)
        if pooling is not None and pooling != stored_pooling:
            raise ValueError(
                f"pooling={pooling!r} contradicts the checkpoint's pooling {stored_pooling!r}"
            )
        if normalize is not None and normalize != stored_normalize:
            raise ValueError(
                f"normalize={normalize} contradicts the checkpoint's normalize={stored_normalize}"
            )
        return stored_pooling, stored_normalize, files
    if pooling is None:
        return None, bool(normalize), files
    return pooling, bool(normalize), files + _pipeline_files(pooling, bool(normalize), dim)
