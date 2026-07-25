#!/usr/bin/env bash
#SBATCH --job-name=lewm-fetch-control
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

: "${DATASET_NAME:?Set the absolute HDF5 path without .h5}"
: "${POLICY:?Set the absolute object checkpoint path or basename}"
: "${OUTPUT_FILENAME:?Set an immutable output filename}"

NUM_EVAL="${NUM_EVAL:-40}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_SEED="${EVAL_SEED:-303072}"
STOCHASTIC="${STOCHASTIC:-false}"
PARTICLES="${PARTICLES:-8}"
OBJECTIVE="${OBJECTIVE:-cvar}"
FLOW_STEPS="${FLOW_STEPS:-8}"
CVAR_TAIL_FRACTION="${CVAR_TAIL_FRACTION:-0.25}"
RISK_WEIGHT="${RISK_WEIGHT:-0.25}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.05}"
SUCCESS_TEMPERATURE="${SUCCESS_TEMPERATURE:-0.01}"
TEMPORAL_COMMON_NOISE="${TEMPORAL_COMMON_NOISE:-false}"
MINIMUM_DISTANCE="${MINIMUM_DISTANCE:-true}"
KERNEL="${KERNEL:-flow}"

if (( NUM_EVAL % 2 )); then
  echo "NUM_EVAL must be even for paired low/high scenes" >&2
  exit 2
fi
if [[ -e "$OUTPUT_FILENAME" || -e "$OUTPUT_FILENAME.summary.json" ]]; then
  echo "Refusing to overwrite control output: $OUTPUT_FILENAME" >&2
  exit 2
fi
POLICY="${POLICY%_object.ckpt}"

mkdir -p "$(dirname "$OUTPUT_FILENAME")"
print_run_metadata
echo "policy=$POLICY dataset=$DATASET_NAME output=$OUTPUT_FILENAME"
echo "episodes=$NUM_EVAL paired_scenes=$((NUM_EVAL / 2)) seed=$EVAL_SEED"
echo "stochastic=$STOCHASTIC particles=$PARTICLES objective=$OBJECTIVE"

EXTRA_ARGS=()
if [[ "$STOCHASTIC" == "true" ]]; then
  EXTRA_ARGS+=(
    "eval.stochastic_planning.enabled=true"
    "eval.stochastic_planning.particles=$PARTICLES"
    "eval.stochastic_planning.objective=$OBJECTIVE"
    "eval.stochastic_planning.cvar_tail_fraction=$CVAR_TAIL_FRACTION"
    "eval.stochastic_planning.risk_weight=$RISK_WEIGHT"
    "eval.stochastic_planning.success_threshold=$SUCCESS_THRESHOLD"
    "eval.stochastic_planning.success_temperature=$SUCCESS_TEMPERATURE"
    "eval.stochastic_planning.temporal_common_noise=$TEMPORAL_COMMON_NOISE"
    "eval.stochastic_planning.minimum_distance=$MINIMUM_DISTANCE"
    "eval.stochastic_planning.kernel=$KERNEL"
    "eval.stochastic_planning.flow_steps=$FLOW_STEPS"
  )
fi
read -r -a EXTRA_OVERRIDE_ARGS <<< "${EXTRA_OVERRIDES:-}"

srun "$PIXI_BIN" run --locked python -u eval.py \
  --config-name=fetch_push_friction.yaml \
  "policy=$POLICY" \
  "eval.dataset_name=$DATASET_NAME" \
  "eval.num_eval=$NUM_EVAL" \
  "eval.batch_size=$BATCH_SIZE" \
  "seed=$EVAL_SEED" \
  "output.filename=$OUTPUT_FILENAME" \
  "${EXTRA_ARGS[@]}" \
  "${EXTRA_OVERRIDE_ARGS[@]}"
