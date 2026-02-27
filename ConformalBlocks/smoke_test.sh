#!/usr/bin/env bash

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-$THIS_DIR/../Bootstrap/ising3d}"
SMOKE_ROOT="${SMOKE_ROOT:-$THIS_DIR/_smoke_ci}"
SMOKE_SKIP_3D="${SMOKE_SKIP_3D:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$SMOKE_ROOT"

echo "[1/5] Python syntax checks"
"$PYTHON_BIN" -m py_compile \
  "$THIS_DIR/conformal_blocks_and_bootstrap_demo_streamlined.py" \
  "$THIS_DIR/2D_conformal_blocks_demo_streamlined.py" \
  "$BOOTSTRAP_DIR/3D_conformal_bootstrap_demo_streamlined.py"

echo "[2/5] 2D smoke run"
"$THIS_DIR/conformal_blocks_and_bootstrap_demo_streamlined.py" 2d \
  --x-min 0.10 --x-max 0.12 --x-step 0.01 \
  --weight-min 0 --weight-max 1 --weight-step 1 \
  --gap-scan-weight-max 1 --gap-scan-step 1 \
  --out-dir "$SMOKE_ROOT/2d"

[[ -f "$SMOKE_ROOT/2d/summary_2d.json" ]]

echo "[3/5] 4D smoke run"
"$THIS_DIR/conformal_blocks_and_bootstrap_demo_streamlined.py" 4d \
  --x-min 0.10 --x-max 0.12 --x-step 0.01 \
  --derivative-degree 1 \
  --out-dir "$SMOKE_ROOT/4d"

[[ -f "$SMOKE_ROOT/4d/summary_4d.json" ]]

if [[ "$SMOKE_SKIP_3D" == "1" ]]; then
  echo "[4/5] 3D smoke run skipped (SMOKE_SKIP_3D=1)"
else
  echo "[4/5] 3D smoke run (requires Docker + SDPB image)"

  if [[ ! -d "$BOOTSTRAP_DIR" ]]; then
    echo "Bootstrap directory not found: $BOOTSTRAP_DIR" >&2
    exit 1
  fi

  bash "$BOOTSTRAP_DIR/setup_python_local.sh"

  RUN_DIR="$SMOKE_ROOT/3d_work" \
  BOOTSTRAP_DIR="$BOOTSTRAP_DIR" \
  "$THIS_DIR/run_3d_demo_local.sh" \
    --delta-sigma 0.518 \
    --k-max 6 --l-max 4 --m-max 1 --n-max 2 \
    --lower 1.1 --upper 1.3 --tol 0.05 \
    --name smoke_ci \
    --sdpb-options "$BOOTSTRAP_DIR/sdpb_options.json" \
    --out "$SMOKE_ROOT/3d/demo_result.json"

  [[ -f "$SMOKE_ROOT/3d/demo_result.json" ]]
fi

echo "[5/5] Smoke summary"
echo "  2D output: $SMOKE_ROOT/2d/summary_2d.json"
echo "  4D output: $SMOKE_ROOT/4d/summary_4d.json"
if [[ "$SMOKE_SKIP_3D" == "1" ]]; then
  echo "  3D output: skipped"
else
  echo "  3D output: $SMOKE_ROOT/3d/demo_result.json"
fi

echo "Smoke tests passed."
