#!/usr/bin/env bash

#SBATCH --job-name=ym3d_shard
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "SLURM_SUBMIT_DIR is not set. Submit with sbatch."
  exit 1
fi

# Use submit directory, not BASH_SOURCE, because Slurm executes a spool copy.
THIS_DIR="$SLURM_SUBMIT_DIR"
cd "$THIS_DIR"
mkdir -p logs

if ! command -v module >/dev/null 2>&1; then
  # In non-login batch shells, Environment Modules may need manual init.
  for modules_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [[ -r "$modules_init" ]]; then
      # shellcheck disable=SC1090
      source "$modules_init"
      break
    fi
  done
fi

if command -v module >/dev/null 2>&1; then
  module purge
  module load python
  if [[ -n "${HARVARD_MODULE_LOAD:-}" ]]; then
    eval "$HARVARD_MODULE_LOAD"
  fi
else
  echo "Warning: 'module' command not found; using python from current PATH."
fi
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN"
  exit 1
fi
echo "Using python: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" - <<'PY'
import sys
import numpy
print("Python:", sys.version.split()[0], "NumPy:", numpy.__version__)
PY

# Prefer one BLAS/OpenMP thread per task to avoid oversubscription in arrays.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CHAIN_ID="${CHAIN_ID:-${SLURM_ARRAY_TASK_ID:?Set CHAIN_ID or submit as array}}"
TOTAL_CHAINS="${TOTAL_CHAINS:-64}"
TOTAL_MEAS="${TOTAL_MEAS:-3000}"
SEED_BASE="${SEED_BASE:-12345}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$THIS_DIR/../results/harvard_shards}"
if [[ -z "${SIM_SCRIPT:-}" ]]; then
  # Support both layouts:
  # 1) .../MonteCarlo/cluster/slurm (preferred)
  # 2) .../MonteCarlo/slurm (legacy flattened sync)
  if [[ -f "$THIS_DIR/../../latticeYM3d-simulation.py" ]]; then
    SIM_SCRIPT="$THIS_DIR/../../latticeYM3d-simulation.py"
  elif [[ -f "$THIS_DIR/../latticeYM3d-simulation.py" ]]; then
    SIM_SCRIPT="$THIS_DIR/../latticeYM3d-simulation.py"
  else
    echo "Simulation script not found. Tried:"
    echo "  $THIS_DIR/../../latticeYM3d-simulation.py"
    echo "  $THIS_DIR/../latticeYM3d-simulation.py"
    exit 1
  fi
fi
if [[ ! -f "$SIM_SCRIPT" ]]; then
  echo "Simulation script not found: $SIM_SCRIPT"
  exit 1
fi

RUNNER_SCRIPT="${RUNNER_SCRIPT:-$THIS_DIR/../scripts/run_latticeYM3d_shard.py}"
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "Runner script not found: $RUNNER_SCRIPT"
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

cmd=(
  "$PYTHON_BIN" "$RUNNER_SCRIPT"
  --simulation-script "$SIM_SCRIPT"
  --chain-id "$CHAIN_ID"
  --total-chains "$TOTAL_CHAINS"
  --total-meas "$TOTAL_MEAS"
  --seed-base "$SEED_BASE"
  --out-dir "$OUTPUT_ROOT"
)

if [[ -n "${LATTICE_L:-}" ]]; then
  cmd+=(--lattice-size "$LATTICE_L")
fi
if [[ -n "${BETA:-}" ]]; then
  cmd+=(--beta "$BETA")
fi
if [[ -n "${N_THERM:-}" ]]; then
  cmd+=(--n-therm "$N_THERM")
fi
if [[ -n "${N_SKIP:-}" ]]; then
  cmd+=(--n-skip "$N_SKIP")
fi
if [[ -n "${EPSILON_MET:-}" ]]; then
  cmd+=(--epsilon-met "$EPSILON_MET")
fi

echo "Harvard RC YM shard: chain=$CHAIN_ID/$TOTAL_CHAINS total_meas=$TOTAL_MEAS"
"${cmd[@]}"
