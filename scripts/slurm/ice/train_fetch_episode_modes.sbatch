#!/usr/bin/env bash
#SBATCH --job-name=fetch-episode-modes
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$REPO_DIR/scripts/slurm/ice/common.sh"
prepare_lewm_runtime

: "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT}"
: "${DATASET_PATH:?Set DATASET_PATH}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST}"
: "${OUTPUT_CHECKPOINT:?Set OUTPUT_CHECKPOINT ending in _object.ckpt}"
SEED="${SEED:-25041}"
EPOCHS="${EPOCHS:-120}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
ASSIGNMENT="${ASSIGNMENT:-balanced_episode}"
OBJECT_MOTION_THRESHOLD="${OBJECT_MOTION_THRESHOLD:-1e-4}"
CENTER_MODE_RESIDUALS="${CENTER_MODE_RESIDUALS:-false}"

if [[ -e "$OUTPUT_CHECKPOINT" || -e "${OUTPUT_CHECKPOINT%.*}.metadata.json" ]]; then
  echo "Refusing to overwrite episode-mode output: $OUTPUT_CHECKPOINT" >&2
  exit 2
fi

print_run_metadata
echo "base_checkpoint=$BASE_CHECKPOINT output=$OUTPUT_CHECKPOINT seed=$SEED"
echo "assignment=$ASSIGNMENT object_motion_threshold=$OBJECT_MOTION_THRESHOLD"
echo "center_mode_residuals=$CENTER_MODE_RESIDUALS"
center_args=()
if [[ "$CENTER_MODE_RESIDUALS" == "true" ]]; then
  center_args+=(--center-mode-residuals)
elif [[ "$CENTER_MODE_RESIDUALS" != "false" ]]; then
  echo "CENTER_MODE_RESIDUALS must be true or false" >&2
  exit 2
fi
srun "$PIXI_BIN" run --locked python -u scripts/train_fetch_episode_modes.py \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --dataset "$DATASET_PATH" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output "$OUTPUT_CHECKPOINT" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --assignment "$ASSIGNMENT" \
  --object-motion-threshold "$OBJECT_MOTION_THRESHOLD" \
  "${center_args[@]}" \
  --device cuda
