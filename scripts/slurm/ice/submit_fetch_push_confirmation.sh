#!/usr/bin/env bash
# Submit the three-seed FetchPush confirmatory prediction experiment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STABLEWM_HOME="${STABLEWM_HOME:-$REPO_DIR/data}"
DATASET_PATH="${DATASET_PATH:-$STABLEWM_HOME/pilots/fetch_push_friction_500pairs_seed3072_serial.h5}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${DATASET_PATH%.h5}.split.json}"
VALIDATION_FILE="${VALIDATION_FILE:-${DATASET_PATH%.h5}.validation.json}"
CONFIRM_DIR="${CONFIRM_DIR:-$STABLEWM_HOME/ice/fetch_push_confirmation}"

ACCOUNT="${ACCOUNT:-cse}"
PARTITION="${PARTITION:-ice-gpu}"
QOS="${QOS:-coe-ice}"
GPU_GRES="${GPU_GRES:-gpu:l40s:1}"
FORK_SEED="${FORK_SEED:-203072}"
CONTEXTS="${CONTEXTS:-128}"
SAMPLES="${SAMPLES:-32}"
HORIZON="${HORIZON:-10}"

for required in "$DATASET_PATH" "$SPLIT_MANIFEST" "$VALIDATION_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "Required immutable input is missing: $required" >&2
    exit 2
  fi
done
if [[ -e "$CONFIRM_DIR/multiseed_verdict.json" ]]; then
  echo "Refusing to overwrite completed confirmation: $CONFIRM_DIR" >&2
  exit 2
fi

mkdir -p "$REPO_DIR/logs/ice" "$CONFIRM_DIR"
cd "$REPO_DIR"

GPU_ARGS=(
  --account="$ACCOUNT"
  --partition="$PARTITION"
  --qos="$QOS"
  --gres="$GPU_GRES"
)
CPU_ARGS=(
  --account="$ACCOUNT"
  --partition="$PARTITION"
  --qos="$QOS"
)
NOMINAL_MATRIX="$REPO_DIR/config/ablations/fetch_push_nominal.tsv"
PRIMARY_MATRIX="$REPO_DIR/config/ablations/fetch_push_prediction_primary.tsv"

nominal_job="$(
  env \
    REPO_DIR="$REPO_DIR" \
    STABLEWM_HOME="$STABLEWM_HOME" \
    DATASET_PATH="$DATASET_PATH" \
    SPLIT_MANIFEST="$SPLIT_MANIFEST" \
    ABLATION_FILE="$NOMINAL_MATRIX" \
    ABLATION_SEEDS="13042,13043" \
    DATA_CONFIG="fetch_push_friction" \
    RUN_SUBDIR_PREFIX="ice/fetch_push_confirmation/nominal" \
    MAX_EPOCHS=10 \
    BATCH_SIZE=64 \
    NUM_WORKERS=6 \
    WANDB_ENABLED=false \
    sbatch --parsable "${GPU_ARGS[@]}" --array=0-1 \
      --job-name=lewm-fetch-nominal-confirm --export=ALL \
      "$SCRIPT_DIR/train_ablation_array.sbatch"
)"

declare -a verdict_jobs=()
for seed_index in 0 1; do
  seed=$((13042 + seed_index))
  seed_dir="$CONFIRM_DIR/seed_$seed"
  nominal_checkpoint="$CONFIRM_DIR/nominal/nominal_task_adapted/seed_$seed/lewm_nominal_task_adapted_seed_${seed}_epoch_10_object.ckpt"
  stats_path="$seed_dir/residual_stats.pt"

  stats_job="$(
    env \
      REPO_DIR="$REPO_DIR" \
      STABLEWM_HOME="$STABLEWM_HOME" \
      DATASET_PATH="$DATASET_PATH" \
      SPLIT_MANIFEST="$SPLIT_MANIFEST" \
      VALIDATION_FILE="$VALIDATION_FILE" \
      NOMINAL_CHECKPOINT="$nominal_checkpoint" \
      RESIDUAL_STATS_OUTPUT="$stats_path" \
      MODEL_SEED="$seed" \
      sbatch --parsable "${GPU_ARGS[@]}" \
        --dependency="afterok:${nominal_job}_${seed_index}" \
        --job-name="lewm-fetch-stats-$seed" --export=ALL \
        "$SCRIPT_DIR/prepare_fetch_push_residual_stats.sbatch"
  )"

  residual_job="$(
    env \
      REPO_DIR="$REPO_DIR" \
      STABLEWM_HOME="$STABLEWM_HOME" \
      DATASET_PATH="$DATASET_PATH" \
      SPLIT_MANIFEST="$SPLIT_MANIFEST" \
      NOMINAL_CHECKPOINT="$nominal_checkpoint" \
      RESIDUAL_SCALE_PATH="$stats_path" \
      ABLATION_FILE="$PRIMARY_MATRIX" \
      ABLATION_SEEDS="$seed" \
      DATA_CONFIG="fetch_push_friction" \
      RUN_SUBDIR_PREFIX="ice/fetch_push_confirmation/seed_$seed/residual" \
      MAX_EPOCHS=10 \
      BATCH_SIZE=64 \
      NUM_WORKERS=6 \
      WANDB_ENABLED=false \
      sbatch --parsable "${GPU_ARGS[@]}" --array=0-1 \
        --dependency="afterok:$stats_job" \
        --job-name="lewm-fetch-residual-$seed" --export=ALL \
        "$SCRIPT_DIR/train_ablation_array.sbatch"
  )"

  gaussian_checkpoint="$seed_dir/residual/conditional_gaussian/seed_$seed/lewm_conditional_gaussian_seed_${seed}_epoch_10_object.ckpt"
  flow_checkpoint="$seed_dir/residual/flow/seed_$seed/lewm_flow_seed_${seed}_epoch_10_object.ckpt"
  eval_common=(
    REPO_DIR="$REPO_DIR"
    STABLEWM_HOME="$STABLEWM_HOME"
    DATASET_PATH="$DATASET_PATH"
    SPLIT_MANIFEST="$SPLIT_MANIFEST"
    NOMINAL_CHECKPOINT="$nominal_checkpoint"
    RESIDUAL_SCALE_PATH="$stats_path"
  )
  gaussian_eval_job="$(
    env "${eval_common[@]}" \
      CHECKPOINT="$gaussian_checkpoint" \
      OUTPUT="$seed_dir/eval_conditional_gaussian.json" \
      sbatch --parsable "${GPU_ARGS[@]}" \
        --dependency="afterok:${residual_job}_0" \
        --job-name="lewm-fetch-gaussian-eval-$seed" --export=ALL \
        "$SCRIPT_DIR/evaluate_residual.sbatch"
  )"
  flow_eval_job="$(
    env "${eval_common[@]}" \
      CHECKPOINT="$flow_checkpoint" \
      OUTPUT="$seed_dir/eval_flow.json" \
      sbatch --parsable "${GPU_ARGS[@]}" \
        --dependency="afterok:${residual_job}_1" \
        --job-name="lewm-fetch-flow-eval-$seed" --export=ALL \
        "$SCRIPT_DIR/evaluate_residual.sbatch"
  )"
  fork_job="$(
    env \
      REPO_DIR="$REPO_DIR" \
      STABLEWM_HOME="$STABLEWM_HOME" \
      DATASET_PATH="$DATASET_PATH" \
      NOMINAL_CHECKPOINT="$nominal_checkpoint" \
      CONDITIONAL_GAUSSIAN_CHECKPOINT="$gaussian_checkpoint" \
      FLOW_CHECKPOINT="$flow_checkpoint" \
      FORK_PAYLOAD="$seed_dir/fork_samples.pt" \
      FORK_RESULT="$seed_dir/fork_metrics.json" \
      FORK_SEED="$FORK_SEED" \
      CONTEXTS="$CONTEXTS" \
      SAMPLES="$SAMPLES" \
      HORIZON="$HORIZON" \
      sbatch --parsable "${GPU_ARGS[@]}" \
        --dependency="afterok:$residual_job" \
        --job-name="lewm-fetch-forks-$seed" --export=ALL \
        "$SCRIPT_DIR/evaluate_fetch_push_forks.sbatch"
  )"
  verdict_job="$(
    env \
      REPO_DIR="$REPO_DIR" \
      STABLEWM_HOME="$STABLEWM_HOME" \
      RESULT_DIR="$seed_dir" \
      HELDOUT_DIR="$seed_dir" \
      INCLUDE_GRU=false \
      OUTPUT="$seed_dir/verdict.json" \
      sbatch --parsable "${CPU_ARGS[@]}" \
        --dependency="afterok:$gaussian_eval_job:$flow_eval_job:$fork_job" \
        --job-name="lewm-fetch-verdict-$seed" --export=ALL \
        "$SCRIPT_DIR/summarize_fetch_push.sbatch"
  )"
  verdict_jobs+=("$verdict_job")
  printf 'seed=%s stats=%s residual=%s gaussian_eval=%s flow_eval=%s fork=%s verdict=%s\n' \
    "$seed" "$stats_job" "$residual_job" "$gaussian_eval_job" \
    "$flow_eval_job" "$fork_job" "$verdict_job"
done

seed_13041_dir="$CONFIRM_DIR/seed_13041"
exploratory_dir="$STABLEWM_HOME/ice/fetch_push_prediction_a100"
seed_13041_fork_job="$(
  env \
    REPO_DIR="$REPO_DIR" \
    STABLEWM_HOME="$STABLEWM_HOME" \
    DATASET_PATH="$DATASET_PATH" \
    NOMINAL_CHECKPOINT="$STABLEWM_HOME/ice/fetch_push_nominal_a100/nominal_task_adapted/seed_13041/lewm_nominal_task_adapted_seed_13041_epoch_10_object.ckpt" \
    CONDITIONAL_GAUSSIAN_CHECKPOINT="$exploratory_dir/conditional_gaussian/seed_13041/lewm_conditional_gaussian_seed_13041_epoch_10_object.ckpt" \
    FLOW_CHECKPOINT="$exploratory_dir/flow/seed_13041/lewm_flow_seed_13041_epoch_10_object.ckpt" \
    FORK_PAYLOAD="$seed_13041_dir/fork_samples.pt" \
    FORK_RESULT="$seed_13041_dir/fork_metrics.json" \
    FORK_SEED="$FORK_SEED" \
    CONTEXTS="$CONTEXTS" \
    SAMPLES="$SAMPLES" \
    HORIZON="$HORIZON" \
    sbatch --parsable "${GPU_ARGS[@]}" \
      --job-name=lewm-fetch-forks-13041 --export=ALL \
      "$SCRIPT_DIR/evaluate_fetch_push_forks.sbatch"
)"
seed_13041_verdict_job="$(
  env \
    REPO_DIR="$REPO_DIR" \
    STABLEWM_HOME="$STABLEWM_HOME" \
    RESULT_DIR="$seed_13041_dir" \
    HELDOUT_DIR="$exploratory_dir" \
    INCLUDE_GRU=false \
    OUTPUT="$seed_13041_dir/verdict.json" \
    sbatch --parsable "${CPU_ARGS[@]}" \
      --dependency="afterok:$seed_13041_fork_job" \
      --job-name=lewm-fetch-verdict-13041 --export=ALL \
      "$SCRIPT_DIR/summarize_fetch_push.sbatch"
)"
verdict_jobs=("$seed_13041_verdict_job" "${verdict_jobs[@]}")

verdict_dependency="$(IFS=:; echo "${verdict_jobs[*]}")"
multiseed_job="$(
  env \
    REPO_DIR="$REPO_DIR" \
    STABLEWM_HOME="$STABLEWM_HOME" \
    CONFIRM_DIR="$CONFIRM_DIR" \
    OUTPUT="$CONFIRM_DIR/multiseed_verdict.json" \
    sbatch --parsable "${CPU_ARGS[@]}" \
      --dependency="afterok:$verdict_dependency" \
      --job-name=lewm-fetch-multiseed --export=ALL \
      "$SCRIPT_DIR/summarize_fetch_push_multiseed.sbatch"
)"

printf 'nominal=%s seed_13041_fork=%s seed_13041_verdict=%s multiseed=%s\n' \
  "$nominal_job" "$seed_13041_fork_job" "$seed_13041_verdict_job" "$multiseed_job"
