#!/usr/bin/env bash
#SBATCH --job-name=lewm-fetch-stats
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

: "${DATASET_PATH:?Set DATASET_PATH to the validated paired HDF5 dataset}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the grouped split}"
: "${NOMINAL_CHECKPOINT:?Set NOMINAL_CHECKPOINT to the frozen nominal model}"
: "${RESIDUAL_STATS_OUTPUT:?Set RESIDUAL_STATS_OUTPUT}"

VALIDATION_FILE="${VALIDATION_FILE:-${DATASET_PATH%.h5}.validation.json}"
RESOLVED_CONFIG="${RESOLVED_CONFIG:-$(dirname "$RESIDUAL_STATS_OUTPUT")/resolved_config.yaml}"
MODEL_SEED="${MODEL_SEED:-13041}"

"$PIXI_BIN" run --locked python - "$VALIDATION_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
if not result.get("pairing_valid") or not result.get("commanded_actions_match"):
    raise SystemExit("FetchPush collection integrity gate failed")
success = float(result["success_rate"])
if not 0.4 <= success <= 0.8:
    raise SystemExit(f"FetchPush task gate failed: balanced success={success:.3f}")
if len(result.get("friction_multipliers", [])) != 2:
    raise SystemExit("FetchPush collection does not contain both friction modes")
print(f"fetch_push_gate=pass success={success:.3f}")
PY

mkdir -p "$(dirname "$RESIDUAL_STATS_OUTPUT")"
"$PIXI_BIN" run --locked python scripts/eval/resolve_train_config.py \
  --output "$RESOLVED_CONFIG" \
  data=fetch_push_friction \
  "+data.dataset.path=$DATASET_PATH" \
  "data.split_manifest=$SPLIT_MANIFEST" \
  "seed=$MODEL_SEED" \
  wandb.enabled=false

print_run_metadata
echo "dataset=$DATASET_PATH"
echo "split_manifest=$SPLIT_MANIFEST"
echo "nominal_checkpoint=$NOMINAL_CHECKPOINT"
echo "residual_stats=$RESIDUAL_STATS_OUTPUT"

srun "$PIXI_BIN" run --locked python -u scripts/eval/compute_residual_scale.py \
  --checkpoint "$NOMINAL_CHECKPOINT" \
  --config "$RESOLVED_CONFIG" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output "$RESIDUAL_STATS_OUTPUT" \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
