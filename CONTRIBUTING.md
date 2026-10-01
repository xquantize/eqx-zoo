# Contributing to eqx-zoo

Thanks for your interest in contributing. eqx-zoo provides pretrained models as Equinox
modules, numerically verified against their Hugging Face reference implementations, so
every change comes with tests that prove it.

## Development setup

```bash
git clone https://github.com/xquantize/eqx-zoo && cd eqx-zoo
uv sync
uv run pytest -m "not checkpoint"   # fast: tiny random models, no downloads
uv run pytest                       # full: also downloads and verifies checkpoints
uv run ruff format && uv run ruff check
```

Reference activations are captured from Hugging Face on first use and cached in
`reference/`; `uv run pytest --regenerate-reference` refreshes them.

## Conventions

- Modules act on a single unbatched sequence; batch with `jax.vmap`.
- Attribute names mirror Hugging Face checkpoint parameter names, so weights load by name.
- Match the reference numerics, including where it computes in float32 (norms, softmax, RoPE).
- Building blocks live in `src/eqx_zoo/layers/`, one module per kind of block. Each model
  class has a sub-package in `src/eqx_zoo/models/` (e.g. `models/causal_lm/`) holding its model
  and its configuration, including the table of supported architectures.
- Tests for each model class live in `tests/<model class>/`; shared building blocks are
  tested directly in `tests/`.
- Google-style docstrings on all public modules, classes and functions, with jaxtyping shape
  annotations.
- The README uses absolute URLs for images and links, because PyPI renders it without the
  repository.

## Adding a model

- **A checkpoint of a supported architecture:** add it to `CHECKPOINTS` in
  `tests/support/registry.py` and run the full test suite.
- **A new architecture:** add it to the architecture table in its model class's `config.py`
  (e.g. `src/eqx_zoo/models/causal_lm/config.py`), a tiny config to `TINY_MODELS` in
  `tests/support/registry.py` (commented with the code paths it covers), and at least one
  verified checkpoint to `CHECKPOINTS`.

## Pull requests

- Keep each pull request to one change, with tests, and keep diffs minimal.
- Pull requests that claim a performance change should include before/after numbers from
  `uv run python benchmarks/generation.py <repo_id> [--dtype bfloat16]`.

## Versioning

eqx-zoo follows semantic versioning, adapted for releases before 1.0.

- **Public API:** the names importable from `eqx_zoo` and `eqx_zoo.layers`. Everything else,
  including `eqx_zoo.models` and modules whose names start with `_`, is internal and may change
  in any release.
- **Patch releases (`0.x.Z`)** contain backwards-compatible changes: new architectures, new
  features, fixes and documentation. Most releases are patch releases.
- **Minor releases (`0.X.0`)** are reserved for changes that break the public API, such as a
  removed name or an incompatible signature. Their release notes list every break and how to
  migrate.

## Releasing (maintainers)

1. From an up-to-date `main`: `git switch -c release/vX.Y.Z`, then `uv version --bump patch`
   (or `minor` for a breaking change; see Versioning), and update `version` and
   `date-released` in `CITATION.cff`.
2. Commit, push, open a pull request, wait for CI, and squash-merge.
3. `gh release create vX.Y.Z --target main --title vX.Y.Z --notes-file <notes>`. The Release
   workflow checks that the tag matches the package version, then publishes to PyPI.
4. Wait a few seconds for the run to register, then find it with
   `gh run list --workflow release.yml` and follow it with `gh run watch <id>`.
5. Verify the published package:
   `uv run --isolated --no-project --refresh --with "eqx-zoo==X.Y.Z" python -c "import eqx_zoo; print(eqx_zoo.__version__)"`
6. Close the milestone, and move any unfinished issues to the next one.
