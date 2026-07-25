#!/usr/bin/env bash
#SBATCH --job-name=fetch-state-dynamics
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$REPO_DIR/scripts/slurm/ice/common.sh"
prepare_lewm_runtime
: "${DATASET_PATH:?Set DATASET_PATH}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST}"
: "${OUTPUT_CHECKPOINT:?Set OUTPUT_CHECKPOINT ending in _object.ckpt}"
SEED="${SEED:-23041}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NOMINAL_EPOCHS="${NOMINAL_EPOCHS:-80}"
FLOW_EPOCHS="${FLOW_EPOCHS:-80}"
MODE_EPOCHS="${MODE_EPOCHS:-80}"
print_run_metadata
srun "$PIXI_BIN" run --locked python -u scripts/train_fetch_state_dynamics.py \
  --dataset "$DATASET_PATH" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output "$OUTPUT_CHECKPOINT" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --nominal-epochs "$NOMINAL_EPOCHS" \
  --flow-epochs "$FLOW_EPOCHS" \
  --mode-epochs "$MODE_EPOCHS" \
  --device cuda
