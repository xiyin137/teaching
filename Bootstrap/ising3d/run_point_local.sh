#!/usr/bin/env bash

# Run one 3D Ising sigma point locally.
#
# What this wrapper does:
# 1) Builds command-line args/environment for `ising3d_even_scan.py`.
# 2) Python script generates conformal blocks via PyCFTBoot
#    (`ConformalBlockTable` + `ConvolvedBlockTable`).
# 3) PyCFTBoot prepares PMP/SDP artifacts and calls `pmp2sdp` + `sdpb`
#    (through local wrappers when `setup_sdpb_local.sh` has been sourced).
#
# Usage:
#   ./run_point_local.sh 0.51800
# or:
#   DEL_SIGMA=0.51800 ./run_point_local.sh

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

DEL_SIGMA="${DEL_SIGMA:-${1:-}}"
if [[ -z "$DEL_SIGMA" ]]; then
  echo "Usage: ./run_point_local.sh <DEL_SIGMA>"
  echo "or set DEL_SIGMA in environment."
  exit 1
fi
if ! [[ "$DEL_SIGMA" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
  echo "DEL_SIGMA must be numeric, got: $DEL_SIGMA"
  exit 1
fi

# Execution/config defaults (override with env vars).
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$THIS_DIR/results/local}"
SDPB_OPTIONS="${SDPB_OPTIONS:-$THIS_DIR/sdpb_options.json}"
export PYCFTBOOT_DIR="${PYCFTBOOT_DIR:-$THIS_DIR/vendor/pycftboot}"

mkdir -p "$OUTPUT_ROOT"

cmd=(
  "$PYTHON_BIN" "$THIS_DIR/ising3d_even_scan.py"
  --sigma-start "$DEL_SIGMA"
  --sigma-end "$DEL_SIGMA"
  --dim "${DIM:-3}"
  --k-max "${K_MAX:-25}"
  --l-max "${L_MAX:-20}"
  --m-max "${M_MAX:-1}"
  --n-max "${N_MAX:-4}"
  --cutoff "${CUTOFF:-0.20}"
  --lower "${BISECT_LOWER:-1.0}"
  --upper "${BISECT_UPPER:-2.0}"
  --tol "${BISECT_TOL:-1e-4}"
  --out-dir "$OUTPUT_ROOT"
  --name-prefix "${NAME_PREFIX:-ising3d_even}"
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
if [[ "${SINGLE_RELEVANT_CHECK:-0}" == "1" ]]; then
  cmd+=(--single-relevant-check)
fi

echo "Running local point: sigma=$DEL_SIGMA"
"${cmd[@]}"
