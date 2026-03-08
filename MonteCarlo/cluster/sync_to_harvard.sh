#!/usr/bin/env bash

set -euo pipefail

# Required:
#   RC_LOGIN='user@login.rc.fas.harvard.edu'
#   RC_BASE_DIR='/n/home00/user/projects'
# Optional:
#   LOCAL_ROOT='/Users/xiyin/teaching'

if [[ -z "${RC_LOGIN:-}" ]]; then
  echo "Set RC_LOGIN first, e.g. user@login.rc.fas.harvard.edu"
  exit 1
fi
if [[ -z "${RC_BASE_DIR:-}" ]]; then
  echo "Set RC_BASE_DIR first, e.g. /n/home00/user/projects"
  exit 1
fi
if [[ "$RC_LOGIN" == *"your_user"* || "$RC_BASE_DIR" == *"your_user"* ]]; then
  echo "Replace placeholder 'your_user' with your actual Harvard RC username."
  exit 1
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$THIS_DIR/../.." && pwd)}"
LOCAL_MC_DIR="$LOCAL_ROOT/MonteCarlo"
REMOTE_MC_DIR="$RC_BASE_DIR/MonteCarlo"

if [[ ! -d "$LOCAL_MC_DIR" ]]; then
  echo "Local MonteCarlo directory not found: $LOCAL_MC_DIR"
  exit 1
fi

ssh "$RC_LOGIN" "mkdir -p '$REMOTE_MC_DIR/cluster'"

rsync -av --delete \
  "$LOCAL_MC_DIR/latticeYM3d-simulation.py" \
  "$LOCAL_MC_DIR/latticeYM3d-analysis.py" \
  "$LOCAL_MC_DIR/latticeYM3d-errors.py" \
  "$LOCAL_MC_DIR/cluster" \
  "$RC_LOGIN:$REMOTE_MC_DIR/"

echo "Synced MonteCarlo to $RC_LOGIN:$REMOTE_MC_DIR"
