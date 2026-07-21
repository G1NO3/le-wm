#!/usr/bin/env bash
#SBATCH --job-name=lewm-ogb-collect
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

COLLECTION_TASK="${COLLECTION_TASK:-double_stack}"
PHYSICS_PROFILE="${PHYSICS_PROFILE:-strong}"
EPISODES="${EPISODES:-1000}"
NUM_ENVS="${NUM_ENVS:-8}"
COLLECTION_SEED="${COLLECTION_SEED:-3072}"
OUTPUT="${OUTPUT:-$STABLEWM_HOME/pilots/${COLLECTION_TASK}_${PHYSICS_PROFILE}_${EPISODES}_seed${COLLECTION_SEED}.h5}"
VALIDATE_ONLY="${VALIDATE_ONLY:-false}"

mkdir -p "$(dirname "$OUTPUT")"
print_run_metadata
echo "collection_task=$COLLECTION_TASK"
echo "physics_profile=$PHYSICS_PROFILE"
echo "episodes=$EPISODES"
echo "num_envs=$NUM_ENVS"
echo "seed=$COLLECTION_SEED"
echo "output=$OUTPUT"
echo "validate_only=$VALIDATE_ONLY"

if [[ "$VALIDATE_ONLY" == "true" ]]; then
  if [[ ! -f "$OUTPUT" ]]; then
    echo "validate-only input does not exist: $OUTPUT" >&2
    exit 2
  fi
else
  srun "$PIXI_BIN" run --locked python -u scripts/data/collect_stochastic_ogbench.py \
    --task "$COLLECTION_TASK" \
    --profile "$PHYSICS_PROFILE" \
    --episodes "$EPISODES" \
    --num-envs "$NUM_ENVS" \
    --seed "$COLLECTION_SEED" \
    --output "$OUTPUT"
fi

srun "$PIXI_BIN" run --locked python -u scripts/data/validate_stochastic_ogbench.py \
  --input "$OUTPUT" \
  --episodes "$EPISODES"
