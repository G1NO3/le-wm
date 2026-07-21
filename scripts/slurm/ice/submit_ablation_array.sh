#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ABLATION_FILE="${ABLATION_FILE:-$REPO_DIR/config/ablations/pusht_residual_flow.tsv}"
ABLATION_SEEDS="${ABLATION_SEEDS:-3072}"

if [[ ! -f "$ABLATION_FILE" ]]; then
  echo "Ablation matrix not found: $ABLATION_FILE" >&2
  exit 2
fi

NUM_ABLATIONS="$(grep -Evc '^[[:space:]]*(#|$)' "$ABLATION_FILE")"
IFS=',' read -r -a SEEDS <<< "$ABLATION_SEEDS"
NUM_JOBS=$((NUM_ABLATIONS * ${#SEEDS[@]}))

if (( NUM_JOBS == 0 )); then
  echo "No ablation jobs were configured." >&2
  exit 2
fi

cd "$REPO_DIR"
mkdir -p logs/ice
export REPO_DIR ABLATION_FILE ABLATION_SEEDS

echo "Submitting $NUM_JOBS jobs: $NUM_ABLATIONS ablations x ${#SEEDS[@]} seeds"
sbatch --array="0-$((NUM_JOBS - 1))" "$@" \
  "$SCRIPT_DIR/train_ablation_array.sbatch"
