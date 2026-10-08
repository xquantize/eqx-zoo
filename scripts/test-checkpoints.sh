#!/usr/bin/env bash
# Run every checkpoint's tests, each in its own process, so memory is freed between models.
# The peak memory is that of the largest single model, not of all models together.
# Extra arguments are passed to pytest, e.g. scripts/test-checkpoints.sh -x
set -eo pipefail
cd "$(dirname "$0")/.."

names=$(python3 -c "import sys; sys.path.insert(0, 'tests'); from support.registry import ALL_CHECKPOINTS; print(*ALL_CHECKPOINTS)")
failed=()
for name in $names; do
  echo "=== $name"
  uv run pytest -q --checkpoint "$name" "$@" || failed+=("$name")
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}"
  exit 1
fi
echo "all checkpoints passed"
