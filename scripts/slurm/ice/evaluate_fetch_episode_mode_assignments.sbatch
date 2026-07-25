#!/usr/bin/env bash
#SBATCH --job-name=fetch-mode-diagnose
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$REPO_DIR/scripts/slurm/ice/common.sh"
prepare_lewm_runtime

: "${CHECKPOINT:?Set CHECKPOINT}"
: "${DATASET_PATH:?Set DATASET_PATH}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST}"
: "${OUTPUT:?Set OUTPUT}"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite assignment diagnostic: $OUTPUT" >&2
  exit 2
fi
print_run_metadata
srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/evaluate_fetch_episode_mode_assignments.py \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET_PATH" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output "$OUTPUT" \
  --device cuda
