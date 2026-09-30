
## Releasing
1. From an up-to-date `main`: `git switch -c release/vX.Y.Z`, then `uv version --bump patch` (or `minor`).
2. Commit, push, open a PR, wait for CI, and squash-merge.
3. `gh release create vX.Y.Z --target main --title vX.Y.Z --notes-file <notes>`. The Release
   workflow checks that the tag matches the package version, then publishes to PyPI.
4. Wait a few seconds for the run to register, then find it with
   `gh run list --workflow release.yml` and follow it with `gh run watch <id>`.
5. Verify the published package:
   `uv run --isolated --no-project --refresh --with "eqx-zoo==X.Y.Z" python -c "import eqx_zoo; print(eqx_zoo.__version__)"`
6. Close the milestone, and move any unfinished issues to the next one.

The README must use absolute URLs for images and links: PyPI renders it without the repository.
