#!/usr/bin/env bash
#SBATCH --job-name=lewm-ablate
#SBATCH --output=logs/ice/%x-%A_%a.out
#SBATCH --error=logs/ice/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

ABLATION_FILE="${ABLATION_FILE:-$REPO_DIR/config/ablations/pusht_residual_flow.tsv}"
ABLATION_SEEDS="${ABLATION_SEEDS:-3072}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-6}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
DATA_CONFIG="${DATA_CONFIG:-pusht}"

mapfile -t ABLATIONS < <(grep -Ev '^[[:space:]]*(#|$)' "$ABLATION_FILE")
IFS=',' read -r -a SEEDS <<< "$ABLATION_SEEDS"

if [[ ${#ABLATIONS[@]} -eq 0 || ${#SEEDS[@]} -eq 0 ]]; then
  echo "The ablation matrix and seed list must both be non-empty." >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SPEC_INDEX=$((TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))

if (( SPEC_INDEX >= ${#ABLATIONS[@]} )); then
  echo "Array task $TASK_ID exceeds the configured ablation matrix." >&2
  exit 2
fi

IFS='|' read -r RUN_NAME HYDRA_OVERRIDES <<< "${ABLATIONS[$SPEC_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
read -r -a EXTRA_ARGS <<< "$HYDRA_OVERRIDES"

RUN_SUBDIR="${RUN_SUBDIR_PREFIX:-ice/ablations}/$RUN_NAME/seed_$SEED"
MODEL_NAME="lewm_${RUN_NAME}_seed_${SEED}"

print_run_metadata
echo "array_task=$TASK_ID"
echo "ablation=$RUN_NAME"
echo "seed=$SEED"
echo "hydra_overrides=$HYDRA_OVERRIDES"
echo "output=$STABLEWM_HOME/$RUN_SUBDIR"

srun "$PIXI_BIN" run --locked python -u train.py \
  "data=$DATA_CONFIG" \
  "seed=$SEED" \
  "trainer.max_epochs=$MAX_EPOCHS" \
  "loader.batch_size=$BATCH_SIZE" \
  "loader.num_workers=$NUM_WORKERS" \
  "wandb.enabled=$WANDB_ENABLED" \
  "subdir=$RUN_SUBDIR" \
  "output_model_name=$MODEL_NAME" \
  "${EXTRA_ARGS[@]}"
