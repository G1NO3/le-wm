#!/usr/bin/env bash
#SBATCH --job-name=lewm-fetch-forks
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

: "${DATASET_PATH:?Set DATASET_PATH}"
: "${NOMINAL_CHECKPOINT:?Set NOMINAL_CHECKPOINT}"
: "${CONDITIONAL_GAUSSIAN_CHECKPOINT:?Set CONDITIONAL_GAUSSIAN_CHECKPOINT}"
: "${FLOW_CHECKPOINT:?Set FLOW_CHECKPOINT}"

FORK_PAYLOAD="${FORK_PAYLOAD:-$STABLEWM_HOME/ice/fetch_push_prediction/fork_samples.pt}"
FORK_RESULT="${FORK_RESULT:-$STABLEWM_HOME/ice/fetch_push_prediction/fork_metrics.json}"
CONTEXTS="${CONTEXTS:-32}"
SAMPLES="${SAMPLES:-32}"
HORIZON="${HORIZON:-10}"
FORK_SEED="${FORK_SEED:-103072}"

mkdir -p "$(dirname "$FORK_PAYLOAD")" "$(dirname "$FORK_RESULT")"
print_run_metadata
echo "dataset=$DATASET_PATH"
echo "nominal=$NOMINAL_CHECKPOINT"
echo "fork_payload=$FORK_PAYLOAD"
echo "fork_result=$FORK_RESULT"
echo "fork_seed=$FORK_SEED"

CHECKPOINT_ARGS=(
  --checkpoint "conditional_gaussian=$CONDITIONAL_GAUSSIAN_CHECKPOINT"
  --checkpoint "flow=$FLOW_CHECKPOINT"
)
if [[ -n "${FLOW_GRU_CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint "flow_gru_memory=$FLOW_GRU_CHECKPOINT")
fi

srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/generate_fetch_push_fork_samples.py \
  --nominal-checkpoint "$NOMINAL_CHECKPOINT" \
  "${CHECKPOINT_ARGS[@]}" \
  --dataset "$DATASET_PATH" \
  --output "$FORK_PAYLOAD" \
  --contexts "$CONTEXTS" \
  --samples "$SAMPLES" \
  --horizon "$HORIZON" \
  --flow-steps 8 \
  --seed "$FORK_SEED" \
  --device cuda

srun "$PIXI_BIN" run --locked python -u scripts/eval/evaluate_fork_samples.py \
  --input "$FORK_PAYLOAD" \
  --output "$FORK_RESULT" \
  --primary-horizon "$HORIZON"
