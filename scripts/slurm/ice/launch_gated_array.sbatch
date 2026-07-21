#!/usr/bin/env bash
#SBATCH --job-name=lewm-gated-launch
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

: "${GATE_FILE:?Set GATE_FILE to the signed execution gate}"
: "${ABLATION_FILE:?Set ABLATION_FILE to the approved full experiment matrix}"

TARGET_ACCOUNT="${TARGET_ACCOUNT:-cse}"
TARGET_PARTITION="${TARGET_PARTITION:-ice-gpu}"
TARGET_QOS="${TARGET_QOS:-coe-ice}"
TARGET_GRES="${TARGET_GRES:-gpu:a100:1}"

print_run_metadata
echo "gate_file=$GATE_FILE"
echo "ablation_file=$ABLATION_FILE"
echo "target_account=$TARGET_ACCOUNT"
echo "target_partition=$TARGET_PARTITION"
echo "target_qos=$TARGET_QOS"
echo "target_gres=$TARGET_GRES"

# submit_gated_array.sh verifies the signed execution gate immediately before
# creating the full array.  This launcher itself is dependency-bound to the
# smoke array, so a failed smoke can never reach this line.
exec "$SCRIPT_DIR/submit_gated_array.sh" \
  --parsable \
  --account="$TARGET_ACCOUNT" \
  --partition="$TARGET_PARTITION" \
  --qos="$TARGET_QOS" \
  --gres="$TARGET_GRES"
