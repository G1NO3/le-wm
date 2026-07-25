#!/usr/bin/env bash
#SBATCH --job-name=lewm-control
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

: "${DATASET_NAME:?Set DATASET_NAME to the absolute HDF5 path without .h5}"
: "${POLICY:?Set POLICY to the absolute checkpoint base without _object.ckpt}"
NUM_EVAL="${NUM_EVAL:-100}"
NUM_CONTEXTS="${NUM_CONTEXTS:-20}"
STOCHASTIC_REPEATS="${STOCHASTIC_REPEATS:-5}"
BATCH_SIZE="${BATCH_SIZE:-5}"
EVAL_SEED="${EVAL_SEED:-12041}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:?Set OUTPUT_FILENAME for this control run}"

# Stochastic particle-MPC settings; STOCHASTIC=false runs deterministic CEM.
STOCHASTIC="${STOCHASTIC:-false}"
PARTICLES="${PARTICLES:-8}"
OBJECTIVE="${OBJECTIVE:-mean}"
FLOW_STEPS="${FLOW_STEPS:-4}"
PARTICLE_CHUNK="${PARTICLE_CHUNK:-4}"

if (( NUM_CONTEXTS * STOCHASTIC_REPEATS != NUM_EVAL )); then
  echo "NUM_CONTEXTS * STOCHASTIC_REPEATS must equal NUM_EVAL" >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "$STOCHASTIC" == "true" ]]; then
  EXTRA_ARGS+=(
    "++eval.stochastic_planning.enabled=true"
    "++eval.stochastic_planning.particles=$PARTICLES"
    "++eval.stochastic_planning.objective=$OBJECTIVE"
    "++eval.stochastic_planning.flow_steps=$FLOW_STEPS"
    "++eval.stochastic_planning.particle_chunk_size=$PARTICLE_CHUNK"
    "++eval.stochastic_planning.common_random_numbers=true"
    # No fitted physical probe yet; the collateral MPC cost stays off until
    # fit_physical_probes.py produces a validated probe checkpoint.
    "++eval.stochastic_planning.collateral_penalty_weight=0"
  )
fi

# Optional extra Hydra overrides (space-separated), e.g. selecting a stage:
#   EXTRA_OVERRIDES="++eval.contact_stages=[4]"
read -r -a EXTRA_OVERRIDE_ARGS <<< "${EXTRA_OVERRIDES:-}"

print_run_metadata
echo "dataset_name=$DATASET_NAME"
echo "policy=$POLICY"
echo "num_eval=$NUM_EVAL num_contexts=$NUM_CONTEXTS repeats=$STOCHASTIC_REPEATS"
echo "stochastic=$STOCHASTIC particles=$PARTICLES objective=$OBJECTIVE nfe=$FLOW_STEPS chunk=$PARTICLE_CHUNK"
echo "extra_overrides=${EXTRA_OVERRIDES:-}"
echo "output=$OUTPUT_FILENAME"

srun "$PIXI_BIN" run --locked python -u eval.py \
  --config-name=ogb_quadruple_stack.yaml \
  policy="$POLICY" \
  eval.dataset_name="$DATASET_NAME" \
  eval.num_eval="$NUM_EVAL" \
  eval.num_contexts="$NUM_CONTEXTS" \
  eval.stochastic_repeats="$STOCHASTIC_REPEATS" \
  eval.batch_size="$BATCH_SIZE" \
  seed="$EVAL_SEED" \
  output.filename="$OUTPUT_FILENAME" \
  "${EXTRA_ARGS[@]}" \
  "${EXTRA_OVERRIDE_ARGS[@]}"
