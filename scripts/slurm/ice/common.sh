#!/usr/bin/env bash

set -euo pipefail

prepare_lewm_runtime() {
  REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
  cd "$REPO_DIR"

  if [[ -n "${PIXI_BIN:-}" ]]; then
    :
  elif command -v pixi >/dev/null 2>&1; then
    PIXI_BIN="$(command -v pixi)"
  elif [[ -x "$HOME/.pixi/bin/pixi" ]]; then
    PIXI_BIN="$HOME/.pixi/bin/pixi"
  else
    echo "pixi was not found. Install it on the ICE login node first." >&2
    echo "See docs/ice-setup.md for the supported bootstrap command." >&2
    return 1
  fi

  export REPO_DIR PIXI_BIN
  export STABLEWM_HOME="${STABLEWM_HOME:-$REPO_DIR/data}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STABLEWM_HOME/.cache}"
  export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-6}}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

  mkdir -p \
    "$STABLEWM_HOME" \
    "$XDG_CACHE_HOME" \
    "$HF_HOME" \
    "$MPLCONFIGDIR" \
    "$REPO_DIR/logs/ice"
}

print_run_metadata() {
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "repo=$REPO_DIR"
  echo "commit=$(git rev-parse HEAD)"
  echo "stablewm_home=$STABLEWM_HOME"
  echo "pixi=$($PIXI_BIN --version)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total \
      --format=csv,noheader
  fi
}
