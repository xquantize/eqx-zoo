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
- Building blocks live in `src/eqx_zoo/layers/`, one module per kind of block; the decoder
  model is `src/eqx_zoo/causal_lm.py`; per-architecture settings are in `src/eqx_zoo/config.py`.
- Google-style docstrings on all public modules, classes and functions, with jaxtyping shape
  annotations.
- The README uses absolute URLs for images and links, because PyPI renders it without the
  repository.

## Adding a model

- **A checkpoint of a supported architecture:** add it to `CHECKPOINTS` in
  `tests/support/registry.py` and run the full test suite.
- **A new architecture:** add its settings to `config.py`, a tiny config to `TINY_MODELS` in
  `tests/support/registry.py` (commented with the code paths it covers), and at least one
  verified checkpoint to `CHECKPOINTS`.

## Pull requests

- Keep each pull request to one change, with tests, and keep diffs minimal.
- Pull requests that claim a performance change should include before/after numbers from
  `uv run python benchmarks/generation.py <repo_id> [--dtype bfloat16]`.

## Releasing (maintainers)

1. From an up-to-date `main`: `git switch -c release/vX.Y.Z`, then `uv version --bump patch`
   (or `minor`).
2. Commit, push, open a pull request, wait for CI, and squash-merge.
3. `gh release create vX.Y.Z --target main --title vX.Y.Z --notes-file <notes>`. The Release
   workflow checks that the tag matches the package version, then publishes to PyPI.
4. Wait a few seconds for the run to register, then find it with
   `gh run list --workflow release.yml` and follow it with `gh run watch <id>`.
5. Verify the published package:
   `uv run --isolated --no-project --refresh --with "eqx-zoo==X.Y.Z" python -c "import eqx_zoo; print(eqx_zoo.__version__)"`
6. Close the milestone, and move any unfinished issues to the next one.
