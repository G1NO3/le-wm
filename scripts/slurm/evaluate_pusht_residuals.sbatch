#!/usr/bin/env bash
#SBATCH --job-name=lewm-eval-resid
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=wu-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/flash/Research/WM}"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export STABLEWM_HOME="${STABLEWM_HOME:-$REPO_DIR/data}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STABLEWM_HOME/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

mkdir -p "$STABLEWM_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR" logs data/eval

source "$REPO_DIR/scripts/pixi_env.sh"
resolve_pixi_bin

CHECKPOINT="${CHECKPOINT:-data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_epoch_1_object.ckpt}"
OUTPUT="${OUTPUT:-data/eval/pusht_rflow_1epoch_residual_eval.json}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_BATCHES="${MAX_BATCHES:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
FLOW_STEPS="${FLOW_STEPS:-8}"

SLURM_BIN="${SLURM_BIN:-/opt/slurm/Ubuntu-20.04/current/bin}"
"$SLURM_BIN/srun" -u "$PIXI_BIN" run --locked python -u scripts/eval/evaluate_latent_residuals.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-batches "$MAX_BATCHES" \
  --num-samples "$NUM_SAMPLES" \
  --flow-steps "$FLOW_STEPS"
