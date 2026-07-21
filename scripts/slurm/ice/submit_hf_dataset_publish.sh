#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

cd "$REPO_DIR"
mkdir -p logs/ice
export REPO_DIR

sbatch "$@" "$SCRIPT_DIR/publish_hf_dataset.sbatch"
