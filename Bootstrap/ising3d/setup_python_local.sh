#!/usr/bin/env bash

# Create/update local Python environment for this workflow.
#
# Installs dependencies required by vendored PyCFTBoot import:
# - symengine
# - sympy
# - mpmath

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$THIS_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install symengine sympy mpmath

export PYCFTBOOT_DIR="${PYCFTBOOT_DIR:-$THIS_DIR/vendor/pycftboot}"
if command -v sdpb >/dev/null 2>&1; then
  # Optional import check when SDPB is already available on PATH.
  python - <<'PY'
import os, sys
from pathlib import Path
p = Path(os.environ.get("PYCFTBOOT_DIR")).resolve()
if not p.is_dir():
    raise SystemExit(f"PYCFTBOOT_DIR not found: {p}")
old = Path.cwd()
try:
    os.chdir(p)
    sys.path.insert(0, str(p))
    import bootstrap  # noqa: F401
finally:
    os.chdir(old)
print("PyCFTBoot import check: OK")
PY
else
  echo "Skipping PyCFTBoot import check: sdpb is not yet on PATH."
  echo "Run 'source setup_sdpb_local.sh' first if you want the import check."
fi

echo "Python environment ready: $VENV_DIR"
