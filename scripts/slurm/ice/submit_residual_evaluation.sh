#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [sbatch options...]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

cd "$REPO_DIR"
mkdir -p logs/ice
export REPO_DIR CHECKPOINT

sbatch "$@" "$SCRIPT_DIR/evaluate_residual.sbatch"
