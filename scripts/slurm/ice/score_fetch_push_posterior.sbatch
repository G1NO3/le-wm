#!/usr/bin/env bash
#SBATCH --job-name=lewm-fetch-posterior-score
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

: "${POSTERIOR_PAYLOAD:?Set POSTERIOR_PAYLOAD}"
: "${PRIOR_PAYLOAD:?Set PRIOR_PAYLOAD}"
: "${POSTERIOR_RESULT:?Set POSTERIOR_RESULT}"
PRIMARY_HORIZON="${PRIMARY_HORIZON:-8}"

print_run_metadata
echo "posterior_payload=$POSTERIOR_PAYLOAD"
echo "prior_payload=$PRIOR_PAYLOAD"
echo "posterior_result=$POSTERIOR_RESULT"

srun "$PIXI_BIN" run --locked python -u \
  scripts/eval/evaluate_fetch_push_posterior.py \
  --posterior-input "$POSTERIOR_PAYLOAD" \
  --prior-input "$PRIOR_PAYLOAD" \
  --output "$POSTERIOR_RESULT" \
  --primary-horizon "$PRIMARY_HORIZON" \
  --seed 33072
