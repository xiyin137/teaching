#!/usr/bin/env bash
# Submit the exact PyCFTBoot tutorial two-point mixed-correlator check.
#
# Target behavior from tutorial.py:
#   pair1 = (0.518, 1.412) -> expected allowed
#   pair2 = (0.530, 1.412) -> expected excluded
#
# Truncation/settings follow tutorial section:
#   k=20, l=20, m=2, n=4, cutoff=0, dualErrorThreshold=1e-15.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/.venvs/ising3d_mixed/bin/activate}"
export PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/ising3d_mixed/bin/python3}"
export SDPB_PATH="${SDPB_PATH:-$HOME/bin/sdpb_singularity/sdpb}"
export MPIRUN_PATH="${MPIRUN_PATH:-$HOME/teaching/teaching/Bootstrap/ising3d/bin/mpirun}"
export SDPB_SIF_IMAGE="${SDPB_SIF_IMAGE:-$HOME/software/sdpb_master.sif}"

export TEST_ROOT="${TEST_ROOT:-$HOME/ising3d_mixed_runs/mixed_tutorial_pair_k20_l20_m2_n4_c0}"
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
SIGMA_START="$s",SIGMA_END="$s",SIGMA_STEP=0.001,\
EPSILON_START="$e",EPSILON_END="$e",EPSILON_STEP=0.001,\
K_MAX=20,L_MAX=20,M_MAX=2,N_MAX=4,CUTOFF=0,DUAL_ERROR_THRESHOLD=1e-15 \
    "$THIS_DIR/run_island_harvard.slurm"
}

submit_pt 0.518 1.412 tutorial_expected_allowed
submit_pt 0.530 1.412 tutorial_expected_excluded

echo "Submitted 2 tutorial-pair jobs on shared."
echo "Expected labels: tutorial_expected_allowed -> allowed, tutorial_expected_excluded -> excluded."
echo "Check status: sacct -X --name ising3d_mixed --format=JobID,State,Elapsed,ExitCode | tail -n 20"
echo "Decision lines: grep -R \"status=ok decision=\" \"$TEST_ROOT/slurm_logs\" | sort"
