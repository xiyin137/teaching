#!/usr/bin/env bash

# Configure local Docker-backed SDPB wrappers.
#
# Must be sourced (not executed) if you want exports in current shell:
#   source setup_sdpb_local.sh
#
# After sourcing, PyCFTBoot will typically call:
#   pmp2sdp -> sdp (via wrapper) [and optionally mpirun wrapper]

set -euo pipefail

fail_setup() {
  echo "ERROR: $*" >&2
  return 1 2>/dev/null || exit 1
}

if [[ -n "${BASH_SOURCE:-}" ]]; then
  SCRIPT_SOURCE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  SCRIPT_SOURCE="${(%):-%N}"
else
  SCRIPT_SOURCE="$0"
fi

THIS_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
BIN_DIR="$THIS_DIR/bin"
VENV_PY="$THIS_DIR/.venv/bin/python"
PYCFTBOOT_DIR="$THIS_DIR/vendor/pycftboot"

export PATH="$BIN_DIR:$PATH"
export SDPB_PATH="$BIN_DIR/sdpb"
export MPIRUN_PATH="$BIN_DIR/mpirun"
export MPIRUN_NP="${MPIRUN_NP:-1}"
export SDPB_DOCKER_IMAGE="${SDPB_DOCKER_IMAGE:-bootstrapcollaboration/sdpb:master}"
export PYCFTBOOT_DIR

if [[ -x "$VENV_PY" ]]; then
  export PATH="$THIS_DIR/.venv/bin:$PATH"
  export PYTHON_BIN="${PYTHON_BIN:-$VENV_PY}"
fi

echo "Configured local SDPB wrappers:"
echo "  SDPB_PATH=$SDPB_PATH"
echo "  MPIRUN_PATH=$MPIRUN_PATH"
echo "  PYCFTBOOT_DIR=$PYCFTBOOT_DIR"
echo "  PYTHON_BIN=${PYTHON_BIN:-python3}"
echo "  PATH starts with $BIN_DIR"

if ! command -v docker >/dev/null 2>&1; then
  fail_setup "docker CLI not found in PATH. Install Docker Desktop/Engine first."
fi
if ! docker info >/dev/null 2>&1; then
  fail_setup "docker daemon is not reachable. Start Docker Desktop/daemon and retry."
fi

echo "Checking docker image..."
docker image inspect "$SDPB_DOCKER_IMAGE" >/dev/null 2>&1 || {
  echo "Image $SDPB_DOCKER_IMAGE not found locally. Pulling..."
  docker pull "$SDPB_DOCKER_IMAGE" || fail_setup "failed to pull image: $SDPB_DOCKER_IMAGE"
}

echo "Verifying wrappers..."
"$BIN_DIR/sdpb" --help >/dev/null || fail_setup "sdpb wrapper health check failed"
"$BIN_DIR/pmp2sdp" --help >/dev/null || fail_setup "pmp2sdp wrapper health check failed"

echo "SDPB local setup is ready."
