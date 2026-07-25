#!/usr/bin/env bash
#SBATCH --job-name=slide-commitment
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$REPO_DIR/scripts/slurm/ice/common.sh"
prepare_lewm_runtime
: "${CHECKPOINT:?Set CHECKPOINT}"
: "${OUTPUT:?Set OUTPUT}"
CONTEXTS="${CONTEXTS:-64}"
RAW_HORIZON="${RAW_HORIZON:-100}"
SEED="${SEED:-703072}"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite commitment output: $OUTPUT" >&2
  exit 2
fi
print_run_metadata
srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/evaluate_fetch_slide_commitment.py \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" \
  --contexts "$CONTEXTS" --raw-horizon "$RAW_HORIZON" \
  --seed "$SEED" --device cuda
