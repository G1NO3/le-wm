#!/usr/bin/env bash
#SBATCH --job-name=slide-state-forks
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
: "${FORK_PAYLOAD:?Set FORK_PAYLOAD}"
: "${FORK_RESULT:?Set FORK_RESULT}"
CONTEXTS="${CONTEXTS:-64}"
SAMPLES="${SAMPLES:-32}"
RAW_HORIZON="${RAW_HORIZON:-100}"
SEED="${SEED:-603072}"
KERNEL="${KERNEL:-mode_mixture}"
if [[ -e "$FORK_PAYLOAD" || -e "$FORK_RESULT" ]]; then
  echo "Refusing to overwrite fork output" >&2
  exit 2
fi
print_run_metadata
srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/generate_fetch_slide_state_fork_samples.py \
  --checkpoint "$CHECKPOINT" --output "$FORK_PAYLOAD" \
  --contexts "$CONTEXTS" --samples "$SAMPLES" \
  --raw-horizon "$RAW_HORIZON" --seed "$SEED" --device cuda \
  --kernel "$KERNEL"
srun "$PIXI_BIN" run --locked python -u scripts/eval/evaluate_fork_samples.py \
  --input "$FORK_PAYLOAD" --output "$FORK_RESULT" \
  --seed "$SEED" --primary-horizon "$((RAW_HORIZON / 2))"
