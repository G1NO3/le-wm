#!/usr/bin/env bash
#SBATCH --job-name=slide-lewm-forks
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
source "$REPO_DIR/scripts/slurm/ice/common.sh"
prepare_lewm_runtime
: "${NOMINAL_CHECKPOINT:?Set NOMINAL_CHECKPOINT}"
: "${DATASET_PATH:?Set DATASET_PATH}"
: "${FORK_PAYLOAD:?Set FORK_PAYLOAD}"
: "${FORK_RESULT:?Set FORK_RESULT}"
CONTEXTS="${CONTEXTS:-64}"
SAMPLES="${SAMPLES:-32}"
RAW_HORIZON="${RAW_HORIZON:-100}"
SEED="${SEED:-603072}"
if [[ -e "$FORK_PAYLOAD" || -e "$FORK_RESULT" ]]; then
  echo "Refusing to overwrite vanilla LeWM fork output" >&2
  exit 2
fi
print_run_metadata
srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/generate_fetch_slide_lewm_fork_samples.py \
  --checkpoint "$NOMINAL_CHECKPOINT" \
  --dataset "$DATASET_PATH" \
  --output "$FORK_PAYLOAD" \
  --contexts "$CONTEXTS" \
  --samples "$SAMPLES" \
  --raw-horizon "$RAW_HORIZON" \
  --seed "$SEED" \
  --device cuda
srun "$PIXI_BIN" run --locked python -u scripts/eval/evaluate_fork_samples.py \
  --input "$FORK_PAYLOAD" \
  --output "$FORK_RESULT" \
  --seed "$SEED" \
  --primary-horizon "$((RAW_HORIZON / 2))"
