#!/usr/bin/env bash

# Fan out a mixed-island scan on Harvard RC by submitting one sigma slice per job.
#
# Usage:
#   bash submit_island_harvard.sh
#   bash submit_island_harvard.sh --time=12:00:00 --partition=shared
#
# Any CLI arguments are forwarded directly to sbatch.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
JOB_PREFIX="${JOB_PREFIX:-ising3d_mixed}"

SIGMA_START="${SIGMA_START:-0.517}"
SIGMA_END="${SIGMA_END:-0.521}"
SIGMA_STEP="${SIGMA_STEP:-0.001}"

mapfile -t sigma_values < <(
  "$PYTHON_BIN" - "$SIGMA_START" "$SIGMA_END" "$SIGMA_STEP" <<'PY'
from decimal import Decimal, InvalidOperation
import sys

try:
    start = Decimal(sys.argv[1])
    end = Decimal(sys.argv[2])
    step = Decimal(sys.argv[3])
except (IndexError, InvalidOperation):
    raise SystemExit("Invalid SIGMA_START/SIGMA_END/SIGMA_STEP")

if step <= 0:
    raise SystemExit("SIGMA_STEP must be > 0")
if start > end:
    raise SystemExit("SIGMA_START must be <= SIGMA_END")

x = start
guard = 0
while x <= end + abs(step) / Decimal("1000000"):
    print(format(x, ".5f"))
    x += step
    guard += 1
    if guard > 1000000:
        raise SystemExit("Too many sigma points; check inputs")
PY
)

if [[ "${#sigma_values[@]}" -eq 0 ]]; then
  echo "No sigma points generated."
  exit 1
fi

for sigma in "${sigma_values[@]}"; do
  tag="${sigma/./}"
  job_name="${JOB_PREFIX}_${tag}"
  sbatch "$@" \
    --job-name "$job_name" \
    --export "ALL,SIGMA_START=${sigma},SIGMA_END=${sigma}" \
    run_island_harvard.slurm
done

echo "Submitted ${#sigma_values[@]} mixed-island jobs."
