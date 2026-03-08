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
JOB_PREFIX="${JOB_PREFIX:-ym3d_scan}"
PARTITION="${PARTITION:-shared}"   # set PARTITION=yin for dedicated partition
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
BETA_LIST="${BETA_LIST:-6.0}"
SCAN_TAG="${SCAN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SCAN_ROOT="${SCAN_ROOT:-$THIS_DIR/../results/beta_scan/$SCAN_TAG}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-harvard_shards}"

if (( TOTAL_CHAINS < 1 )); then
  echo "TOTAL_CHAINS must be >= 1"
  exit 1
fi
if (( TOTAL_MEAS < 1 )); then
  echo "TOTAL_MEAS must be >= 1"
  exit 1
fi
if [[ -z "$BETA_LIST" ]]; then
  echo "BETA_LIST must not be empty."
  exit 1
fi

read -r -a extra_args <<< "$SBATCH_EXTRA_ARGS"
array_end=$((TOTAL_CHAINS - 1))

mkdir -p "$SCAN_ROOT"
manifest="$SCAN_ROOT/submit_manifest.tsv"
printf "job_id\tbeta\toutput_root\n" > "$manifest"

beta_tokens="${BETA_LIST//,/ }"
submitted=0
for beta in $beta_tokens; do
  # Make a filesystem-safe tag from beta (e.g. 6.0 -> 6p0, -5.5 -> m5p5).
  beta_tag="${beta//-/m}"
  beta_tag="${beta_tag//./p}"

  beta_root="$SCAN_ROOT/beta_${beta_tag}"
  shard_output="$beta_root/$OUTPUT_SUBDIR"
  mkdir -p "$shard_output"

  submit_out="$(
    sbatch "${extra_args[@]}" \
      --chdir="$THIS_DIR" \
      --partition="$PARTITION" \
      --job-name="${JOB_PREFIX}_b${beta_tag}" \
      --array="0-${array_end}" \
      --export=ALL,TOTAL_CHAINS="$TOTAL_CHAINS",TOTAL_MEAS="$TOTAL_MEAS",SEED_BASE="$SEED_BASE",BETA="$beta",OUTPUT_ROOT="$shard_output" \
      "$THIS_DIR/run_latticeYM3d_shard_harvard.slurm"
  )"

  job_id="$(awk '/Submitted batch job/ {print $4}' <<< "$submit_out")"
  if [[ -z "$job_id" ]]; then
    echo "Failed to parse job id from sbatch output:"
    echo "$submit_out"
    exit 1
  fi

  printf "%s\t%s\t%s\n" "$job_id" "$beta" "$shard_output" >> "$manifest"
  echo "Submitted beta=$beta as job_id=$job_id output=$shard_output"
  submitted=$((submitted + 1))
done

echo "Submitted $submitted beta points."
echo "Manifest: $manifest"
