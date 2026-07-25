#!/usr/bin/env bash
#SBATCH --job-name=lewm-resid-eval
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "CHECKPOINT must point to an object checkpoint." >&2
  exit 2
fi

# Training configs deliberately keep these paths as oc.env interpolations so
# checkpoints can move between machines.  Export every pinned input here as
# well; otherwise OmegaConf fails while loading the checkpoint config before
# evaluation can verify any hashes or construct the held-out loader.
: "${DATASET_PATH:?Set DATASET_PATH to the immutable HDF5 collection}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the grouped episode split}"
: "${NOMINAL_CHECKPOINT:?Set NOMINAL_CHECKPOINT to the frozen nominal checkpoint}"
: "${RESIDUAL_SCALE_PATH:?Set RESIDUAL_SCALE_PATH to the centered train-only statistics}"

OUTPUT="${OUTPUT:-$STABLEWM_HOME/ice/evaluations/$(basename "${CHECKPOINT%_object.ckpt}").json}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_BATCHES="${MAX_BATCHES:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
FLOW_STEPS="${FLOW_STEPS:-8}"
ALLOW_LEGACY_WINDOW_SPLIT="${ALLOW_LEGACY_WINDOW_SPLIT:-false}"

mkdir -p "$(dirname "$OUTPUT")"
print_run_metadata
echo "checkpoint=$CHECKPOINT"
echo "output=$OUTPUT"

EXTRA_ARGS=()
if [[ "$ALLOW_LEGACY_WINDOW_SPLIT" == "true" ]]; then
  EXTRA_ARGS+=(--allow-legacy-window-split)
fi

srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/evaluate_latent_residuals.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-batches "$MAX_BATCHES" \
  --num-samples "$NUM_SAMPLES" \
  --flow-steps "$FLOW_STEPS" \
  "${EXTRA_ARGS[@]}"
