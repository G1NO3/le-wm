#!/usr/bin/env bash
#SBATCH --job-name=fetch-flow-commit
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
source "$REPO_DIR/scripts/slurm/ice/common.sh"
prepare_lewm_runtime

: "${CHECKPOINT:?Set the label-free residual-flow object checkpoint}"
: "${OUTPUT:?Set an immutable JSON output path}"
CONTEXTS="${CONTEXTS:-64}"
SEED="${SEED:-1083072}"
FLOW_SEED="${FLOW_SEED:-24041}"
FLOW_PARTICLES="${FLOW_PARTICLES:-32}"
FLOW_STEPS="${FLOW_STEPS:-4}"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite flow commitment result: $OUTPUT" >&2
  exit 2
fi

print_run_metadata
echo "checkpoint=$CHECKPOINT output=$OUTPUT contexts=$CONTEXTS seed=$SEED"
echo "flow_seed=$FLOW_SEED flow_particles=$FLOW_PARTICLES flow_steps=$FLOW_STEPS"
srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/evaluate_fetch_push_flow_commitment.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --contexts "$CONTEXTS" \
  --seed "$SEED" \
  --flow-seed "$FLOW_SEED" \
  --flow-particles "$FLOW_PARTICLES" \
  --flow-steps "$FLOW_STEPS" \
  --device cuda
