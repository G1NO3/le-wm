#!/usr/bin/env bash
#SBATCH --job-name=lewm-fetch-verdict
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

RESULT_DIR="${RESULT_DIR:-$STABLEWM_HOME/ice/fetch_push_prediction_a100}"
HELDOUT_DIR="${HELDOUT_DIR:-$RESULT_DIR}"
OUTPUT="${OUTPUT:-$RESULT_DIR/verdict.json}"
INCLUDE_GRU="${INCLUDE_GRU:-true}"

print_run_metadata
echo "result_dir=$RESULT_DIR"
echo "heldout_dir=$HELDOUT_DIR"
echo "output=$OUTPUT"

HELDOUT_ARGS=(
  --heldout "conditional_gaussian=$HELDOUT_DIR/eval_conditional_gaussian.json"
  --heldout "flow=$HELDOUT_DIR/eval_flow.json"
)
if [[ "$INCLUDE_GRU" == "true" ]]; then
  HELDOUT_ARGS+=(
    --heldout "flow_gru_memory=$HELDOUT_DIR/eval_flow_gru_memory.json"
  )
fi

srun "$PIXI_BIN" run --locked python -u scripts/eval/summarize_fetch_push_results.py \
  "${HELDOUT_ARGS[@]}" \
  --fork "$RESULT_DIR/fork_metrics.json" \
  --output "$OUTPUT" \
  --horizon 10
