#!/usr/bin/env bash
#SBATCH --job-name=lewm-fetch-posterior
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
: "${PRIOR_PAYLOAD:?Set PRIOR_PAYLOAD to the matched pre-contact payload}"
: "${POSTERIOR_PAYLOAD:?Set POSTERIOR_PAYLOAD}"
: "${POSTERIOR_RESULT:?Set POSTERIOR_RESULT}"

CONTEXTS="${CONTEXTS:-128}"
SAMPLES="${SAMPLES:-32}"
HORIZON="${HORIZON:-8}"
OBSERVATION_MODEL_STEPS="${OBSERVATION_MODEL_STEPS:-2}"
POSTERIOR_SEED="${POSTERIOR_SEED:-203072}"

mkdir -p "$(dirname "$POSTERIOR_PAYLOAD")" "$(dirname "$POSTERIOR_RESULT")"
print_run_metadata
echo "prior_payload=$PRIOR_PAYLOAD"
echo "posterior_payload=$POSTERIOR_PAYLOAD"
echo "posterior_result=$POSTERIOR_RESULT"
echo "posterior_seed=$POSTERIOR_SEED"
echo "observation_model_steps=$OBSERVATION_MODEL_STEPS"

CHECKPOINT_ARGS=(
  --checkpoint "conditional_gaussian=$CONDITIONAL_GAUSSIAN_CHECKPOINT"
  --checkpoint "flow=$FLOW_CHECKPOINT"
)
if [[ -n "${FLOW_GRU_CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint "flow_gru_memory=$FLOW_GRU_CHECKPOINT")
fi

srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/generate_fetch_push_posterior_samples.py \
  --nominal-checkpoint "$NOMINAL_CHECKPOINT" \
  "${CHECKPOINT_ARGS[@]}" \
  --dataset "$DATASET_PATH" \
  --output "$POSTERIOR_PAYLOAD" \
  --contexts "$CONTEXTS" \
  --samples "$SAMPLES" \
  --horizon "$HORIZON" \
  --observation-model-steps "$OBSERVATION_MODEL_STEPS" \
  --flow-steps 8 \
  --seed "$POSTERIOR_SEED" \
  --device cuda

srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/evaluate_fetch_push_posterior.py \
  --posterior-input "$POSTERIOR_PAYLOAD" \
  --prior-input "$PRIOR_PAYLOAD" \
  --output "$POSTERIOR_RESULT" \
  --primary-horizon "$HORIZON" \
  --seed 33072

if [[ -n "${MODE_PROBE_RESULT:-}" ]]; then
  srun "$PIXI_BIN" run --locked python -u \
    scripts/eval/evaluate_fetch_push_mode_probe.py \
    --input "$POSTERIOR_PAYLOAD" \
    --output "$MODE_PROBE_RESULT" \
    --seed 43072
fi
