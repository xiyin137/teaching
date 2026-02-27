#!/usr/bin/env bash

#SBATCH --job-name=ising3d
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

# Single-point Harvard RC runner.
#
# Flow is the same as local:
# - Python driver builds conformal block tables
# - PyCFTBoot prepares SDP files (via pmp2sdp)
# - SDPB is called using configured paths/module environment
#
# Submit via:
#   sbatch --export=ALL,DEL_SIGMA=0.51800 run_point_harvard.slurm

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "SLURM_SUBMIT_DIR is not set. Submit with sbatch."
  exit 1
fi
: "${DEL_SIGMA:?Set DEL_SIGMA via sbatch --export}"
if ! [[ "$DEL_SIGMA" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
  echo "DEL_SIGMA must be numeric, got: $DEL_SIGMA"
  exit 1
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"
mkdir -p logs

module purge
module load python
if [[ -n "${HARVARD_MODULE_LOAD:-}" ]]; then
  eval "$HARVARD_MODULE_LOAD"
fi

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$THIS_DIR/results/harvard}"
SDPB_OPTIONS="${SDPB_OPTIONS:-$THIS_DIR/sdpb_options.json}"
export PYCFTBOOT_DIR="${PYCFTBOOT_DIR:-$THIS_DIR/vendor/pycftboot}"

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

echo "Harvard RC point: sigma=$DEL_SIGMA"
"${cmd[@]}"
