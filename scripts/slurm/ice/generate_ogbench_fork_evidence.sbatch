#!/usr/bin/env bash
#SBATCH --job-name=lewm-ogb-forks
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

COLLECTION_TASK="${COLLECTION_TASK:-double_stack}"
PHYSICS_PROFILE="${PHYSICS_PROFILE:-strong}"
EPISODES="${EPISODES:-1000}"
COLLECTION_SEED="${COLLECTION_SEED:-3072}"
FORK_SEED="${FORK_SEED:-9182}"
CONTEXTS="${CONTEXTS:-64}"
REALIZATIONS="${REALIZATIONS:-32}"
HORIZON="${HORIZON:-5}"
INPUT="${INPUT:-$STABLEWM_HOME/pilots/${COLLECTION_TASK}_${PHYSICS_PROFILE}_${EPISODES}_seed${COLLECTION_SEED}.h5}"
OUTPUT="${OUTPUT:-${INPUT%.h5}.h${HORIZON}_forks.pt}"

print_run_metadata
echo "input=$INPUT"
echo "output=$OUTPUT"
echo "task=$COLLECTION_TASK"
echo "profile=$PHYSICS_PROFILE"
echo "contexts=$CONTEXTS"
echo "realizations=$REALIZATIONS"
echo "horizon=$HORIZON"
echo "fork_seed=$FORK_SEED"

srun "$PIXI_BIN" run --locked python -u scripts/data/generate_pilot_fork_evidence.py \
  --input "$INPUT" \
  --task "$COLLECTION_TASK" \
  --profile "$PHYSICS_PROFILE" \
  --output "$OUTPUT" \
  --contexts "$CONTEXTS" \
  --realizations "$REALIZATIONS" \
  --horizon "$HORIZON" \
  --seed "$FORK_SEED"
