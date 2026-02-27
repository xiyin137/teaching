#!/usr/bin/env bash

# Run a local mixed-correlator 3D Ising island scan.
#
# Pipeline:
# 1) Builds mixed conformal block tables at each (Delta_sigma, Delta_epsilon).
# 2) Constructs the mixed-vector SDP and runs SDPB feasibility (`iterate`).
# 3) Aggregates results to scan_results.csv/json and renders an island plot.
#
# Typical use:
#   source ../ising3d/setup_sdpb_local.sh
#   bash run_island_local.sh
#
# Optional quick run:
#   SIGMA_START=0.517 SIGMA_END=0.519 SIGMA_STEP=0.001 \
#   EPSILON_START=1.410 EPSILON_END=1.430 EPSILON_STEP=0.005 \
#   K_MAX=14 L_MAX=14 M_MAX=1 N_MAX=3 CUTOFF=0.20 bash run_island_local.sh

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$THIS_DIR/results/local_mixed_island}"
SDPB_OPTIONS="${SDPB_OPTIONS:-$THIS_DIR/../ising3d/sdpb_options.json}"

export PYCFTBOOT_DIR="${PYCFTBOOT_DIR:-$THIS_DIR/../ising3d/vendor/pycftboot}"

SIGMA_START="${SIGMA_START:-0.517}"
SIGMA_END="${SIGMA_END:-0.521}"
SIGMA_STEP="${SIGMA_STEP:-0.001}"

EPSILON_START="${EPSILON_START:-1.410}"
EPSILON_END="${EPSILON_END:-1.435}"
EPSILON_STEP="${EPSILON_STEP:-0.0025}"

cmd=(
  "$PYTHON_BIN" "$THIS_DIR/ising3d_mixed_island_scan.py"
  --sigma-start "$SIGMA_START"
  --sigma-end "$SIGMA_END"
  --sigma-step "$SIGMA_STEP"
  --epsilon-start "$EPSILON_START"
  --epsilon-end "$EPSILON_END"
  --epsilon-step "$EPSILON_STEP"
  --dim "${DIM:-3}"
  --k-max "${K_MAX:-25}"
  --l-max "${L_MAX:-20}"
  --m-max "${M_MAX:-2}"
  --n-max "${N_MAX:-6}"
  --cutoff "${CUTOFF:-0.15}"
  --dual-error-threshold "${DUAL_ERROR_THRESHOLD:-1e-15}"
  --max-points "${MAX_POINTS:-1200}"
  --name-prefix "${NAME_PREFIX:-ising3d_mixed}"
  --out-dir "$OUTPUT_ROOT"
)

if [[ -f "$SDPB_OPTIONS" ]]; then
  cmd+=(--sdpb-options "$SDPB_OPTIONS")
fi
if [[ -n "${SDPB_PATH:-}" ]]; then
  cmd+=(--sdpb-path "$SDPB_PATH")
fi
if [[ -n "${MPIRUN_PATH:-}" ]]; then
  cmd+=(--mpirun-path "$MPIRUN_PATH")
fi
if [[ -n "${MPIRUN_NP:-}" ]]; then
  cmd+=(--mpirun-np "$MPIRUN_NP")
fi

echo "Running mixed island scan locally..."
"${cmd[@]}"

CSV_PATH="$OUTPUT_ROOT/scan_results.csv"
JSON_PATH="$OUTPUT_ROOT/scan_results.json"
if [[ -f "$CSV_PATH" ]]; then
  if ! "$PYTHON_BIN" "$THIS_DIR/plot_ising3d_mixed_island.py" \
    --csv "$CSV_PATH" \
    --json "$JSON_PATH" \
    --out-prefix "$OUTPUT_ROOT/ising3d_mixed_island"; then
    echo "Warning: island plot generation failed (is matplotlib installed?)."
  fi
fi
