#!/usr/bin/env bash
#SBATCH --job-name=lewm-mem-filter
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "CHECKPOINT must point to a GRU-flow object checkpoint." >&2
  exit 2
fi

OUTPUT="${OUTPUT:-$STABLEWM_HOME/ice/evaluations/$(basename "${CHECKPOINT%_object.ckpt}")_memory_filter.json}"
STATE_OUTPUT="${STATE_OUTPUT:-${OUTPUT%.json}_states.pt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_EPISODES="${MAX_EPISODES:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
FLOW_STEPS="${FLOW_STEPS:-8}"
UPDATE_CHECKPOINTS="${UPDATE_CHECKPOINTS:-1,2,3,5,8,12,20,30,50,75,100,150,200}"

mkdir -p "$(dirname "$OUTPUT")"
print_run_metadata
echo "checkpoint=$CHECKPOINT"
echo "output=$OUTPUT"
echo "state_output=$STATE_OUTPUT"

srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/evaluate_memory_filter.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --state-output "$STATE_OUTPUT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-episodes "$MAX_EPISODES" \
  --num-samples "$NUM_SAMPLES" \
  --flow-steps "$FLOW_STEPS" \
  --update-checkpoints "$UPDATE_CHECKPOINTS"
