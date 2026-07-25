#!/usr/bin/env bash
#SBATCH --job-name=fetch-commitment
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

: "${CHECKPOINT:?Set the mode-residual object checkpoint}"
: "${OUTPUT:?Set an immutable JSON output path}"
CONTEXTS="${CONTEXTS:-64}"
SEED="${SEED:-903072}"
RAW_HORIZON="${RAW_HORIZON:-20}"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite commitment result: $OUTPUT" >&2
  exit 2
fi

print_run_metadata
echo "checkpoint=$CHECKPOINT output=$OUTPUT contexts=$CONTEXTS seed=$SEED"
srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/evaluate_fetch_push_commitment.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --contexts "$CONTEXTS" \
  --seed "$SEED" \
  --raw-horizon "$RAW_HORIZON" \
  --device cuda
