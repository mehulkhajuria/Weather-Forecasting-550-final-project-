#!/usr/bin/env bash
# Reproducible per-machine setup for Rutgers iLab / Amarel / Linux workstations.
# Use --install-only on Amarel login nodes; submit amarel_verify.sbatch for tests.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_TESTS=1
if [[ "${1:-}" == "--install-only" ]]; then RUN_TESTS=0; fi
if [[ ! -f pyproject.toml ]]; then echo 'Missing pyproject.toml' >&2; exit 1; fi

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "PYTHON_BIN not found: $PYTHON_BIN" >&2; exit 1; }
    printf '%s\n' "$PYTHON_BIN"
    return
  fi
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,15)' 2>/dev/null; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo 'No supported Python 3.11-3.14 found. Load a suitable Python module first.' >&2
  exit 1
}
PY="$(pick_python)"
"$PY" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,15), "Requires Python 3.11 through 3.14"'

echo "Creating environment using $(command -v "$PY")"
if [[ ! -x .venv/bin/python ]]; then "$PY" -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[all]'
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > .venv/requirements-installed.txt
printf 'Environment: %s\n' "$(readlink -f .venv/bin/python)"
.venv/bin/python --version
if (( RUN_TESTS == 1 )); then
  .venv/bin/python -m pytest -q
  .venv/bin/weather550 demo --out data/demo
  echo 'Environment installed; offline tests and demo completed.'
else
  echo 'Environment installed. Compute tests skipped on shared login node.'
  echo 'On Amarel: submit "sbatch scripts/amarel_verify.sbatch".'
fi
