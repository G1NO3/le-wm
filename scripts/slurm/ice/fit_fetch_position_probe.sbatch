#!/usr/bin/env bash
#SBATCH --job-name=lewm-fetch-probe
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$REPO_DIR/scripts/slurm/ice/common.sh"
prepare_lewm_runtime
: "${NOMINAL_CHECKPOINT:?Set NOMINAL_CHECKPOINT}"
: "${DATASET_PATH:?Set DATASET_PATH}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST}"
: "${PROBE_OUTPUT:?Set PROBE_OUTPUT}"
print_run_metadata
srun "$PIXI_BIN" run --locked python -u scripts/eval/fit_fetch_position_probe.py \
  --checkpoint "$NOMINAL_CHECKPOINT" \
  --dataset "$DATASET_PATH" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output "$PROBE_OUTPUT" \
  --device cuda
