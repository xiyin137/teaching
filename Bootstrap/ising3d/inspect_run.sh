#!/usr/bin/env bash

# Inspect the latest Harvard/local bootstrap run state.
#
# This helper summarizes:
# - current queue entries (when `squeue` is available),
# - latest log file and trailing output lines,
# - aggregate result files under OUTPUT_ROOT,
# - optional per-sigma artifact snapshot.
#
# Typical usage:
#   bash inspect_run.sh
#   bash inspect_run.sh --sigma 0.51800
#   OUTPUT_ROOT=/n/homeXX/you/ising3d_runs bash inspect_run.sh --lines 120

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-$THIS_DIR/results/harvard}"
LOG_DIR="${LOG_DIR:-$THIS_DIR/logs}"
TAIL_LINES="${TAIL_LINES:-80}"
SIGMA_INPUT=""
SHOW_QUEUE=1

usage() {
  cat <<'EOF'
Usage: bash inspect_run.sh [options]

Options:
  --output-root PATH   Override output root (default: $OUTPUT_ROOT or results/harvard).
  --log-dir PATH       Override log directory (default: $LOG_DIR or logs/).
  --lines N            Tail N lines from latest log (default: 80).
  --sigma VALUE        Show details for sigma point (e.g. 0.51800 or 051800).
  --no-queue           Skip queue summary.
  -h, --help           Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --output-root"
        exit 2
      fi
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --log-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --log-dir"
        exit 2
      fi
      LOG_DIR="$2"
      shift 2
      ;;
    --lines)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --lines"
        exit 2
      fi
      TAIL_LINES="$2"
      shift 2
      ;;
    --sigma)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --sigma"
        exit 2
      fi
      SIGMA_INPUT="$2"
      shift 2
      ;;
    --no-queue)
      SHOW_QUEUE=0
      shift
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

if [[ ! "$TAIL_LINES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --lines value: $TAIL_LINES"
  exit 2
fi

sigma_to_tag() {
  local raw="$1"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    echo "$raw"
    return
  fi
  if [[ "$raw" =~ ^[0-9]*\.[0-9]+$ ]]; then
    printf "%.5f" "$raw" | tr -d '.'
    return
  fi
  echo ""
}

print_header() {
  local title="$1"
  echo
  echo "=== $title ==="
}

if [[ "$SHOW_QUEUE" -eq 1 ]]; then
  print_header "Queue"
  if command -v squeue >/dev/null 2>&1; then
    squeue -u "${USER:-}" | sed -n '1,12p'
  else
    echo "squeue not found in PATH"
  fi
fi

print_header "Latest Log"
if [[ -d "$LOG_DIR" ]] && compgen -G "$LOG_DIR/*.out" >/dev/null; then
  latest_log="$(ls -1t "$LOG_DIR"/*.out | head -n 1)"
  echo "File: $latest_log"
  echo "--- tail -n $TAIL_LINES ---"
  tail -n "$TAIL_LINES" "$latest_log"

  if grep -q "status=ok" "$latest_log"; then
    echo
    echo "Signal: found at least one 'status=ok' line in latest log."
  fi
  if grep -Eiq "status=failed|traceback|exception|error" "$latest_log"; then
    echo "Signal: latest log contains failure/error keywords."
  fi
else
  echo "No log files found at $LOG_DIR/*.out"
fi

print_header "Output Root"
echo "Path: $OUTPUT_ROOT"
if [[ ! -d "$OUTPUT_ROOT" ]]; then
  echo "Output root does not exist yet."
  exit 0
fi

sigma_count="$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name 'sigma_*' | wc -l | tr -d ' ')"
echo "Sigma directories: $sigma_count"

csv_path="$OUTPUT_ROOT/scan_results.csv"
json_path="$OUTPUT_ROOT/scan_results.json"

if [[ -f "$csv_path" ]]; then
  echo
  echo "scan_results.csv (last 5 lines):"
  tail -n 5 "$csv_path"
else
  echo "scan_results.csv not found"
fi

if [[ -f "$json_path" ]]; then
  echo
  echo "scan_results.json size: $(wc -c < "$json_path" | tr -d ' ') bytes"
else
  echo "scan_results.json not found"
fi

if [[ -n "$SIGMA_INPUT" ]]; then
  tag="$(sigma_to_tag "$SIGMA_INPUT")"
  if [[ -z "$tag" ]]; then
    echo
    echo "Invalid --sigma value: $SIGMA_INPUT"
    exit 2
  fi

  point_dir="$OUTPUT_ROOT/sigma_$tag"
  print_header "Sigma $SIGMA_INPUT (tag: $tag)"
  if [[ ! -d "$point_dir" ]]; then
    echo "Point directory not found: $point_dir"
    exit 0
  fi

  echo "Directory: $point_dir"
  echo "File count: $(find "$point_dir" -maxdepth 1 -type f | wc -l | tr -d ' ')"

  echo
  echo "Top files:"
  find "$point_dir" -maxdepth 1 -type f | sed "s|$point_dir/||" | sort | head -n 40

  my_sdp_count="$(find "$point_dir" -maxdepth 1 -type f -name 'mySDP*' | wc -l | tr -d ' ')"
  ckpt_count="$(find "$point_dir" -maxdepth 1 -type f \( -name '*.ck' -o -name '*checkpoint*' \) | wc -l | tr -d ' ')"
  out_count="$(find "$point_dir" -maxdepth 1 -type f \( -name '*.out' -o -name '*.log' \) | wc -l | tr -d ' ')"

  echo
  echo "Artifact counters:"
  echo "  mySDP*: $my_sdp_count"
  echo "  checkpoint-like: $ckpt_count"
  echo "  log/out files: $out_count"
fi

