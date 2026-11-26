#!/usr/bin/env bash
# Install the wheel built in dist/ for every interpreter of $BUILD_PYTHONS (or $PYTHONS)
# into a fresh virtual environment, and run the test suite against it.
set -euo pipefail

pythons="${BUILD_PYTHONS:-$PYTHONS}"
failed=""
for version in $pythons; do
  echo "::group::Python $version"
  rm -rf .venv-test
  uv venv -q --managed-python --python "$version" .venv-test
  if ! uv pip install -q --python .venv-test --only-binary numpy,shapely,scipy numpy shapely scipy pytest; then
    echo "::warning::No NumPy / Shapely / SciPy wheel for Python $version on this platform: wheel not tested"
    echo "::endgroup::"
    continue
  fi
  # optional: the plotting tests are skipped where matplotlib has no wheel
  uv pip install -q --python .venv-test --only-binary matplotlib matplotlib ||
    echo "::notice::No matplotlib wheel for Python $version: plotting tests skipped"
  # --no-index: install the wheel just built, never one from PyPI.
  uv pip install -q --python .venv-test --no-cache --no-index --no-deps --find-links dist quickpaver
  if [[ -x .venv-test/bin/python ]]; then py=.venv-test/bin/python; else py=.venv-test/Scripts/python.exe; fi
  "$py" -m pytest -q -p no:cacheprovider tests || failed="$failed $version"
  echo "::endgroup::"
done
rm -rf .venv-test

if [[ -n "$failed" ]]; then
  echo "::error::Tests failed for Python$failed"
  exit 1
fi
