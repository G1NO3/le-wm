#!/usr/bin/env bash

resolve_pixi_bin() {
  if [[ -n "${PIXI_BIN:-}" ]]; then
    :
  elif command -v pixi >/dev/null 2>&1; then
    PIXI_BIN="$(command -v pixi)"
  elif [[ -x "$HOME/.pixi/bin/pixi" ]]; then
    PIXI_BIN="$HOME/.pixi/bin/pixi"
  else
    echo "pixi was not found; install it before running this script." >&2
    return 1
  fi
  export PIXI_BIN
}
