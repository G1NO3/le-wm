#!/usr/bin/env bash
#SBATCH --job-name=lewm-hard-screen
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

: "${DATASET_NAME:?Set DATASET_NAME to the absolute HDF5 path without .h5}"
: "${POLICY:?Set POLICY to the absolute checkpoint base without _object.ckpt}"
NUM_EVAL="${NUM_EVAL:-100}"
NUM_CONTEXTS="${NUM_CONTEXTS:-20}"
STOCHASTIC_REPEATS="${STOCHASTIC_REPEATS:-5}"
BATCH_SIZE="${BATCH_SIZE:-10}"
EVAL_SEED="${EVAL_SEED:-12041}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-quadruple_stack_official_lewm_reachable_screen.txt}"

if (( NUM_CONTEXTS * STOCHASTIC_REPEATS != NUM_EVAL )); then
  echo "NUM_CONTEXTS * STOCHASTIC_REPEATS must equal NUM_EVAL" >&2
  exit 2
fi

print_run_metadata
echo "dataset_name=$DATASET_NAME"
echo "policy=$POLICY"
echo "num_eval=$NUM_EVAL"
echo "num_contexts=$NUM_CONTEXTS"
echo "stochastic_repeats=$STOCHASTIC_REPEATS"
echo "batch_size=$BATCH_SIZE"
echo "eval_seed=$EVAL_SEED"

srun "$PIXI_BIN" run --locked python -u eval.py \
  --config-name=ogb_quadruple_stack.yaml \
  policy="$POLICY" \
  eval.dataset_name="$DATASET_NAME" \
  eval.num_eval="$NUM_EVAL" \
  eval.num_contexts="$NUM_CONTEXTS" \
  eval.stochastic_repeats="$STOCHASTIC_REPEATS" \
  eval.batch_size="$BATCH_SIZE" \
  seed="$EVAL_SEED" \
  output.filename="$OUTPUT_FILENAME"
