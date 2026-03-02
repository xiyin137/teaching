#!/usr/bin/env bash
# Submit six mixed-correlator regression points on Harvard RC shared partition.
# This is intended to validate expected allowed/excluded classifications against
# known reference points.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/.venvs/ising3d_mixed/bin/activate}"
export PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/ising3d_mixed/bin/python3}"
export SDPB_PATH="${SDPB_PATH:-$HOME/bin/sdpb_singularity/sdpb}"
export MPIRUN_PATH="${MPIRUN_PATH:-$HOME/teaching/teaching/Bootstrap/ising3d/bin/mpirun}"
export SDPB_SIF_IMAGE="${SDPB_SIF_IMAGE:-$HOME/software/sdpb_master.sif}"

export TEST_ROOT="${TEST_ROOT:-$HOME/ising3d_mixed_runs/mixed_equation_regression_k20_l20_m1_n6_c015}"
mkdir -p "$TEST_ROOT/slurm_logs"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Missing VENV_ACTIVATE: $VENV_ACTIVATE"
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing PYTHON_BIN: $PYTHON_BIN"
  exit 1
fi
if [[ ! -x "$SDPB_PATH" ]]; then
  echo "Missing SDPB_PATH executable: $SDPB_PATH"
  exit 1
fi
if [[ ! -x "$MPIRUN_PATH" ]]; then
  echo "Missing MPIRUN_PATH executable: $MPIRUN_PATH"
  exit 1
fi

submit_pt () {
  local s="$1" e="$2" tag="$3"
  sbatch --partition=shared --time=03:00:00 \
    --output="$TEST_ROOT/slurm_logs/${tag}_%j.out" \
    --export=ALL,VENV_ACTIVATE="$VENV_ACTIVATE",PYTHON_BIN="$PYTHON_BIN",SDPB_PATH="$SDPB_PATH",MPIRUN_PATH="$MPIRUN_PATH",SDPB_SIF_IMAGE="$SDPB_SIF_IMAGE",\
OUTPUT_ROOT="$TEST_ROOT",NAME_PREFIX="$tag",\
SIGMA_START="$s",SIGMA_END="$s",SIGMA_STEP=0.0001,\
EPSILON_START="$e",EPSILON_END="$e",EPSILON_STEP=0.0001,\
K_MAX=20,L_MAX=20,M_MAX=1,N_MAX=6,CUTOFF=0.15,DUAL_ERROR_THRESHOLD=1e-15 \
    "$THIS_DIR/run_island_harvard.slurm"
}

submit_pt 0.51810 1.41260 exp_allowed_1
submit_pt 0.51820 1.41270 exp_allowed_2
submit_pt 0.51830 1.41280 exp_allowed_3
submit_pt 0.51800 1.41250 exp_excluded_1
submit_pt 0.51790 1.41265 exp_excluded_2
submit_pt 0.51780 1.41255 exp_excluded_3

echo "Submitted 6 regression jobs to shared."
echo "Check: squeue -u \"\$USER\" -p shared,yin"
echo "Logs:  $TEST_ROOT/slurm_logs"
