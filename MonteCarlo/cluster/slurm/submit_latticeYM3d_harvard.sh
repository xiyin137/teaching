#!/usr/bin/env bash

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH."
  exit 1
fi

TOTAL_CHAINS="${TOTAL_CHAINS:-64}"
TOTAL_MEAS="${TOTAL_MEAS:-3000}"
SEED_BASE="${SEED_BASE:-12345}"
JOB_PREFIX="${JOB_PREFIX:-ym3d}"
PARTITION="${PARTITION:-shared}"   # set PARTITION=yin for dedicated partition
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

if (( TOTAL_CHAINS < 1 )); then
  echo "TOTAL_CHAINS must be >= 1"
  exit 1
fi
if (( TOTAL_MEAS < 1 )); then
  echo "TOTAL_MEAS must be >= 1"
  exit 1
fi

read -r -a extra_args <<< "$SBATCH_EXTRA_ARGS"
array_end=$((TOTAL_CHAINS - 1))

sbatch "${extra_args[@]}" \
  --chdir="$THIS_DIR" \
  --partition="$PARTITION" \
  --job-name="$JOB_PREFIX" \
  --array="0-${array_end}" \
  --export=ALL,TOTAL_CHAINS="$TOTAL_CHAINS",TOTAL_MEAS="$TOTAL_MEAS",SEED_BASE="$SEED_BASE" \
  "$THIS_DIR/run_latticeYM3d_shard_harvard.slurm"

echo "Submitted Harvard RC array job:"
echo "  partition=$PARTITION"
echo "  total_chains=$TOTAL_CHAINS"
echo "  total_meas=$TOTAL_MEAS"
echo "  seed_base=$SEED_BASE"
echo "  array=0-${array_end}"
