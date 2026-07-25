#!/usr/bin/env bash
#SBATCH --job-name=lewm-slide-collect
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

PAIRS="${PAIRS:-100}"
NUM_ENVS="${NUM_ENVS:-1}"
COLLECTION_SEED="${COLLECTION_SEED:-4072}"
PHYSICS_PROFILE="${PHYSICS_PROFILE:-slide_strong}"
IMAGE_SIZE="${IMAGE_SIZE:-112}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-130}"
OUTPUT="${OUTPUT:-$STABLEWM_HOME/pilots/fetch_slide_friction_${PAIRS}pairs_seed${COLLECTION_SEED}.h5}"
SPLIT_OUTPUT="${SPLIT_OUTPUT:-${OUTPUT%.h5}.split.json}"

mkdir -p "$(dirname "$OUTPUT")"
print_run_metadata
echo "pairs=$PAIRS"
echo "num_envs=$NUM_ENVS"
echo "profile=$PHYSICS_PROFILE"
echo "seed=$COLLECTION_SEED"
echo "image_size=$IMAGE_SIZE"
echo "max_episode_steps=$MAX_EPISODE_STEPS"
echo "output=$OUTPUT"
echo "split=$SPLIT_OUTPUT"

EXTRA_ARGS=()
if (( PAIRS > 100 )); then
  EXTRA_ARGS+=(--confirm-large)
fi

srun "$PIXI_BIN" run --locked python -u scripts/data/collect_fetch_slide_friction.py \
  --profile "$PHYSICS_PROFILE" \
  --pairs "$PAIRS" \
  --num-envs "$NUM_ENVS" \
  --seed "$COLLECTION_SEED" \
  --image-size "$IMAGE_SIZE" \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --output "$OUTPUT" \
  "${EXTRA_ARGS[@]}"

srun "$PIXI_BIN" run --locked python -u scripts/data/validate_fetch_slide_friction.py \
  --input "$OUTPUT" \
  --pairs "$PAIRS"

srun "$PIXI_BIN" run --locked python -u scripts/data/create_episode_split_manifest.py \
  --input "$OUTPUT" \
  --output "$SPLIT_OUTPUT" \
  --seed "$COLLECTION_SEED" \
  --group-key privileged/pair_id
