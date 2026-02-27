#!/usr/bin/env bash

# Harvard RC preflight checks for the 3D Ising bootstrap workflow.
#
# This script validates the same pieces that the production run needs:
# - scheduler availability (`sbatch`)
# - Python interpreter and core package imports
# - key files/directories in this workflow
# - output directory write access
# - SDPB/mpirun overrides when explicitly set
# - visibility of `pmp2sdp` in current shell
#
# Typical usage:
#   source env.harvard.example
#   bash preflight_harvard.sh
#
# If you want the script to source an env file itself:
#   bash preflight_harvard.sh --source-env --env-file env.harvard.example

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

SOURCE_ENV=0
ENV_FILE="$THIS_DIR/env.harvard.example"

usage() {
  cat <<'EOF'
Usage: bash preflight_harvard.sh [--source-env] [--env-file PATH]

Options:
  --source-env       Source environment variables before checks.
  --env-file PATH    Environment file to source with --source-env.
  -h, --help         Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-env)
      SOURCE_ENV=1
      shift
      ;;
    --env-file)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --env-file"
        exit 2
      fi
      ENV_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ "$SOURCE_ENV" -eq 1 ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "[FAIL] env file not found: $ENV_FILE"
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

errors=0
warnings=0

pass() {
  echo "[PASS] $1"
}

warn() {
  echo "[WARN] $1"
  warnings=$((warnings + 1))
}

fail() {
  echo "[FAIL] $1"
  errors=$((errors + 1))
}

check_command() {
  local cmd="$1"
  local label="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "$label: $(command -v "$cmd")"
  else
    fail "$label: command '$cmd' not found"
  fi
}

check_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    pass "$label: $path"
  else
    fail "$label missing: $path"
  fi
}

check_command "sbatch" "Scheduler"
if command -v squeue >/dev/null 2>&1; then
  pass "Queue tool: $(command -v squeue)"
else
  warn "Queue tool 'squeue' not found in current shell"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  pass "Python executable: $(command -v "$PYTHON_BIN")"
  if "$PYTHON_BIN" --version >/dev/null 2>&1; then
    echo "       $("$PYTHON_BIN" --version 2>&1)"
  fi
else
  fail "Python executable not found: $PYTHON_BIN"
fi

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
required = ["symengine", "sympy", "mpmath"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
  then
    pass "Python packages available: symengine, sympy, mpmath"
  else
    warn "Missing one or more Python packages: symengine, sympy, mpmath"
  fi
fi

check_file "$THIS_DIR/ising3d_even_scan.py" "Python driver"
check_file "$THIS_DIR/run_point_harvard.slurm" "Harvard run script"
check_file "$THIS_DIR/submit_scan_harvard.sh" "Harvard submit script"

PYCFTBOOT_DIR="${PYCFTBOOT_DIR:-$THIS_DIR/vendor/pycftboot}"
if [[ -d "$PYCFTBOOT_DIR" && -f "$PYCFTBOOT_DIR/bootstrap.py" ]]; then
  pass "PYCFTBOOT_DIR: $PYCFTBOOT_DIR"
else
  fail "PYCFTBOOT_DIR invalid or missing bootstrap.py: $PYCFTBOOT_DIR"
fi

SDPB_OPTIONS="${SDPB_OPTIONS:-$THIS_DIR/sdpb_options.json}"
if [[ -f "$SDPB_OPTIONS" ]]; then
  pass "SDPB options file: $SDPB_OPTIONS"
else
  warn "SDPB options file not found (optional): $SDPB_OPTIONS"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$THIS_DIR/results/harvard}"
if mkdir -p "$OUTPUT_ROOT" && [[ -w "$OUTPUT_ROOT" ]]; then
  pass "OUTPUT_ROOT writable: $OUTPUT_ROOT"
else
  fail "Cannot create/write OUTPUT_ROOT: $OUTPUT_ROOT"
fi

if [[ -n "${SDPB_PATH:-}" ]]; then
  if [[ -x "$SDPB_PATH" ]]; then
    pass "SDPB_PATH executable: $SDPB_PATH"
  else
    fail "SDPB_PATH set but not executable: $SDPB_PATH"
  fi
else
  warn "SDPB_PATH not set (PyCFTBoot default path resolution will be used)"
fi

if [[ -n "${MPIRUN_PATH:-}" ]]; then
  if [[ -x "$MPIRUN_PATH" ]]; then
    pass "MPIRUN_PATH executable: $MPIRUN_PATH"
  else
    fail "MPIRUN_PATH set but not executable: $MPIRUN_PATH"
  fi
fi

if [[ -n "${MPIRUN_NP:-}" ]]; then
  if [[ "$MPIRUN_NP" =~ ^[1-9][0-9]*$ ]]; then
    pass "MPIRUN_NP: $MPIRUN_NP"
  else
    fail "MPIRUN_NP must be a positive integer: $MPIRUN_NP"
  fi
fi

if command -v pmp2sdp >/dev/null 2>&1; then
  pass "pmp2sdp visible in current shell: $(command -v pmp2sdp)"
else
  warn "pmp2sdp not visible in current shell (may still appear inside job module environment)"
fi

echo
echo "Preflight summary: $errors fail, $warnings warn"
if [[ "$errors" -gt 0 ]]; then
  echo "Fix failures before submitting production scans."
  exit 1
fi

echo "Preflight passed. Suggested one-point validation:"
echo "  sbatch --export=ALL,DEL_SIGMA=0.51800 run_point_harvard.slurm"

