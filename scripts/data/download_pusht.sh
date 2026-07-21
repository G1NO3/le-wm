#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="${STABLEWM_HOME:-$REPO_DIR/data}"
RAW_DIR="$DATA_DIR/raw"
ARCHIVE="$RAW_DIR/pusht_expert_train.h5.zst"
OUTPUT="$DATA_DIR/pusht_expert_train.h5"
export RAW_DIR ARCHIVE OUTPUT

cd "$REPO_DIR"
mkdir -p "$RAW_DIR" "$DATA_DIR"

if [[ -f "$OUTPUT" ]]; then
  echo "Dataset already exists: $OUTPUT"
  exit 0
fi

source "$REPO_DIR/scripts/pixi_env.sh"
resolve_pixi_bin

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DIR/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME"

"$PIXI_BIN" run --locked python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import shutil

repo_id = "quentinll/lewm-pusht"
filename = "pusht_expert_train.h5.zst"
raw_dir = Path(os.environ["RAW_DIR"])
archive = Path(os.environ["ARCHIVE"])

path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    local_dir=raw_dir,
    local_dir_use_symlinks=False,
)

if Path(path) != archive:
    shutil.copyfile(path, archive)

print(f"Downloaded {repo_id}/{filename} -> {archive}")
PY

"$PIXI_BIN" run --locked python - <<'PY'
from pathlib import Path
import os
import zstandard as zstd

archive = Path(os.environ["ARCHIVE"])
output = Path(os.environ["OUTPUT"])
tmp = output.with_suffix(output.suffix + ".tmp")

print(f"Decompressing {archive} -> {output}")
with archive.open("rb") as src, tmp.open("wb") as dst:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(src, dst)

tmp.replace(output)
print(f"Ready: {output}")
PY
