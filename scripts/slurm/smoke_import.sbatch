#!/usr/bin/env bash
#SBATCH --job-name=lewm-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=wu-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:10:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/flash/Research/WM}"
cd "$REPO_DIR"

export STABLEWM_HOME="${STABLEWM_HOME:-$REPO_DIR/data}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STABLEWM_HOME/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

mkdir -p "$STABLEWM_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR" logs

source "$REPO_DIR/scripts/pixi_env.sh"
resolve_pixi_bin

hostname
nvidia-smi -L || true
SLURM_BIN="${SLURM_BIN:-/opt/slurm/Ubuntu-20.04/current/bin}"
"$SLURM_BIN/srun" -u "$PIXI_BIN" run --locked smoke-import
"$SLURM_BIN/srun" -u "$PIXI_BIN" run --locked smoke-cuda
"$SLURM_BIN/srun" -u "$PIXI_BIN" run --locked smoke-residual
