#!/usr/bin/env bash

set -euo pipefail

REPO_ID="${REPO_ID:-quentinll/lewm-cube}"
REVISION="${REVISION:-b0747c5002e86d2ce8f3cd8178004b97524c587d}"
STABLEWM_HOME="${STABLEWM_HOME:?Set STABLEWM_HOME to persistent storage}"
PIXI_BIN="${PIXI_BIN:-pixi}"
NAME="${NAME:-lewm_cube}"
SOURCE_DIR="$STABLEWM_HOME/checkpoints/official/hf/$NAME/$REVISION"
OUTPUT="$STABLEWM_HOME/checkpoints/official/${NAME}_object.ckpt"

if [[ -f "$OUTPUT" && -f "${OUTPUT%.ckpt}.json" ]]; then
  echo "Official checkpoint already prepared: $OUTPUT"
  exit 0
fi

mkdir -p "$SOURCE_DIR" "$(dirname "$OUTPUT")"
"$PIXI_BIN" run --locked hf download "$REPO_ID" \
  --revision "$REVISION" \
  --local-dir "$SOURCE_DIR"
"$PIXI_BIN" run --locked python scripts/data/convert_hf_lewm_checkpoint.py \
  --source "$SOURCE_DIR" \
  --output "$OUTPUT" \
  --repo-id "$REPO_ID" \
  --revision "$REVISION"
