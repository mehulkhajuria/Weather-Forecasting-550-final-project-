#!/usr/bin/env bash
# Run on a workstation, iLab permitted interactive session, or an allocated SLURM compute node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -x .venv/bin/python ]] || { echo 'Missing .venv. Run scripts/setup_linux.sh first.' >&2; exit 1; }
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/weather550 demo --out data/demo
.venv/bin/python - <<'PY'
import numpy as np
with np.load('data/demo/SYNTHETIC_numeric.npz') as dat:
    print('Synthetic tensor:', dat['X_numerical'].shape)
    print('Validity mask:   ', dat['X_mask'].shape)
PY
