#!/usr/bin/env bash
#SBATCH --job-name=lewm-fetch-collect
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

PAIRS="${PAIRS:-500}"
NUM_ENVS="${NUM_ENVS:-1}"
COLLECTION_SEED="${COLLECTION_SEED:-3072}"
PHYSICS_PROFILE="${PHYSICS_PROFILE:-push_strong}"
OUTPUT="${OUTPUT:-$STABLEWM_HOME/pilots/fetch_push_friction_${PAIRS}pairs_seed${COLLECTION_SEED}.h5}"
SPLIT_OUTPUT="${SPLIT_OUTPUT:-${OUTPUT%.h5}.split.json}"

mkdir -p "$(dirname "$OUTPUT")"
print_run_metadata
echo "pairs=$PAIRS"
echo "num_envs=$NUM_ENVS"
echo "profile=$PHYSICS_PROFILE"
echo "seed=$COLLECTION_SEED"
echo "output=$OUTPUT"
echo "split=$SPLIT_OUTPUT"

srun "$PIXI_BIN" run --locked python -u scripts/data/collect_fetch_push_friction.py \
  --profile "$PHYSICS_PROFILE" \
  --pairs "$PAIRS" \
  --num-envs "$NUM_ENVS" \
  --seed "$COLLECTION_SEED" \
  --output "$OUTPUT"

srun "$PIXI_BIN" run --locked python -u scripts/data/validate_fetch_push_friction.py \
  --input "$OUTPUT" \
  --pairs "$PAIRS"

srun "$PIXI_BIN" run --locked python -u scripts/data/create_episode_split_manifest.py \
  --input "$OUTPUT" \
  --output "$SPLIT_OUTPUT" \
  --seed "$COLLECTION_SEED" \
  --group-key privileged/pair_id
