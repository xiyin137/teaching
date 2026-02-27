#!/usr/bin/env bash

# Run a local sigma-range scan by repeatedly invoking run_point_local.sh.
#
# Notes:
# - Conformal blocks are rebuilt per call (per-point process model).
# - This simple loop is robust and easy to checkpoint/restart.
# - For fast sanity checks, reduce K/L/M/N and widen BISECT_TOL.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

SIGMA_START="${SIGMA_START:-0.518}"
SIGMA_END="${SIGMA_END:-0.521}"
SIGMA_STEP="${SIGMA_STEP:-0.001}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

count=0
while IFS= read -r sigma_fmt; do
  DEL_SIGMA="$sigma_fmt" ./run_point_local.sh
  count=$((count + 1))
done < <(generate_sigma_values)

echo "Completed $count local points."
