#!/usr/bin/env bash
#SBATCH --job-name=lewm-fetch-multiseed
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

: "${CONFIRM_DIR:?Set CONFIRM_DIR to the multi-seed result root}"
OUTPUT="${OUTPUT:-$CONFIRM_DIR/multiseed_verdict.json}"

print_run_metadata
echo "confirm_dir=$CONFIRM_DIR"
echo "output=$OUTPUT"

srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/summarize_fetch_push_multiseed.py \
  --run "13041=$CONFIRM_DIR/seed_13041" \
  --run "13042=$CONFIRM_DIR/seed_13042" \
  --run "13043=$CONFIRM_DIR/seed_13043" \
  --output "$OUTPUT" \
  --bootstrap-draws 5000 \
  --seed 23072
