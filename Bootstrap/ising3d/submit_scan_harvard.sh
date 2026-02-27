#!/usr/bin/env bash

# Submit one Harvard RC job per sigma value.
#
# This wrapper fans out the scan range into independent cluster jobs,
# which is typically easier to manage than one long monolithic run.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

SIGMA_START="${SIGMA_START:-0.518}"
SIGMA_END="${SIGMA_END:-0.521}"
SIGMA_STEP="${SIGMA_STEP:-0.001}"
JOB_PREFIX="${JOB_PREFIX:-ising3d}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH."
  exit 1
fi

# Parse optional sbatch arguments safely from shell-like syntax.
# This preserves quoted chunks (for example: --comment "long note")
# without executing the content.
extra_args=()
if [[ -n "$SBATCH_EXTRA_ARGS" ]]; then
  while IFS= read -r arg; do
    extra_args+=("$arg")
  done < <(
    "$PYTHON_BIN" - "$SBATCH_EXTRA_ARGS" <<'PY'
import shlex
import sys

try:
    tokens = shlex.split(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"Invalid SBATCH_EXTRA_ARGS quoting: {exc}")

for token in tokens:
    print(token)
PY
  )
fi

generate_sigma_values() {
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
eps = abs(step) / Decimal("1000000")
while x <= end + eps:
    print(format(x, ".5f"))
    x += step
    guard += 1
    if guard > 1000000:
        raise SystemExit("Too many scan points; check sigma range inputs")
PY
}

submitted=0
while IFS= read -r sigma_fmt; do
  sigma_id=${sigma_fmt/./}
  job_name="${JOB_PREFIX}_${sigma_id}"

  sbatch "${extra_args[@]}" \
    --job-name="$job_name" \
    --export=ALL,DEL_SIGMA="$sigma_fmt" \
    run_point_harvard.slurm

  submitted=$((submitted + 1))
done < <(generate_sigma_values)

echo "Submitted $submitted Harvard RC jobs."
