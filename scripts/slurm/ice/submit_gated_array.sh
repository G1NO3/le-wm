#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GATE_FILE="${GATE_FILE:?Set GATE_FILE to the preceding gate JSON}"

"${PIXI_BIN:-pixi}" run python "$REPO_DIR/scripts/verify_gate.py" "$GATE_FILE"
exec "$SCRIPT_DIR/submit_ablation_array.sh" "$@"
