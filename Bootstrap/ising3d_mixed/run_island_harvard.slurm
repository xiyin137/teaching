#!/usr/bin/env bash

#SBATCH --job-name=ising3d_mixed
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

# Single-job Harvard RC mixed-correlator island scan.
#
# This launches the full (Delta_sigma, Delta_epsilon) grid in one SLURM job.
# For fan-out submission by sigma slices, use submit_island_harvard.sh.

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "SLURM_SUBMIT_DIR is not set. Submit with sbatch."
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
OUTPUT_ROOT="${OUTPUT_ROOT:-$THIS_DIR/results/harvard_mixed_island}"
SDPB_OPTIONS="${SDPB_OPTIONS:-$THIS_DIR/../ising3d/sdpb_options.json}"
export PYCFTBOOT_DIR="${PYCFTBOOT_DIR:-$THIS_DIR/../ising3d/vendor/pycftboot}"

cmd=(
  "$PYTHON_BIN" "$THIS_DIR/ising3d_mixed_island_scan.py"
  --sigma-start "${SIGMA_START:-0.517}"
  --sigma-end "${SIGMA_END:-0.521}"
  --sigma-step "${SIGMA_STEP:-0.001}"
  --epsilon-start "${EPSILON_START:-1.410}"
  --epsilon-end "${EPSILON_END:-1.435}"
  --epsilon-step "${EPSILON_STEP:-0.0025}"
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

echo "Harvard RC mixed-island scan starting"
"${cmd[@]}"

CSV_PATH="$OUTPUT_ROOT/scan_results.csv"
JSON_PATH="$OUTPUT_ROOT/scan_results.json"
if [[ -f "$CSV_PATH" ]]; then
  if ! "$PYTHON_BIN" "$THIS_DIR/plot_ising3d_mixed_island.py" \
    --csv "$CSV_PATH" \
    --json "$JSON_PATH" \
    --out-prefix "$OUTPUT_ROOT/ising3d_mixed_island"; then
    echo "Warning: island plot generation failed (is matplotlib available on this node?)."
  fi
fi
