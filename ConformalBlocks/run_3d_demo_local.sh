#!/usr/bin/env bash

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-$DEMO_DIR/../Bootstrap/ising3d}"
RUN_DIR="${RUN_DIR:-$DEMO_DIR}"

if [[ ! -d "$BOOTSTRAP_DIR" ]]; then
  echo "Bootstrap directory not found: $BOOTSTRAP_DIR" >&2
  exit 1
fi

# Provides Docker-backed sdpb/pmp2sdp wrappers and default PYTHON_BIN.
# shellcheck source=/dev/null
source "$BOOTSTRAP_DIR/setup_sdpb_local.sh"

export PYCFTBOOT_DIR="${PYCFTBOOT_DIR:-$BOOTSTRAP_DIR/vendor/pycftboot}"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
exec "$BOOTSTRAP_DIR/3D_conformal_bootstrap_demo_streamlined.py" "$@"
