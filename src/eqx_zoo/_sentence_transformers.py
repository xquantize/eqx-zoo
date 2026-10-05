"""Read sentence-transformers pipelines, and apply their pooling and normalisation."""

import json
from pathlib import Path

from jaxtyping import Array, Bool, Float

from eqx_zoo.layers import cls_pool, l2_normalize, last_token_pool, mean_pool

# sentence-transformers pooling modes -> our names for them.
_MODES = {"cls_token": "cls", "mean_tokens": "mean", "lasttoken": "last_token"}
_POOL = {"cls": cls_pool, "mean": mean_pool, "last_token": last_token_pool}


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
