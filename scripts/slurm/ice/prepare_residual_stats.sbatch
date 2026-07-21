#!/usr/bin/env bash
#SBATCH --job-name=lewm-residual-stats
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime

: "${PILOT_GATE_FILE:?Set PILOT_GATE_FILE to the strict scientific-gate JSON}"
: "${EXECUTION_GATE_FILE:?Set EXECUTION_GATE_FILE to the signed execution-gate output}"
: "${DATASET_PATH:?Set DATASET_PATH to the validated HDF5 dataset}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the immutable episode split}"
: "${NOMINAL_CHECKPOINT:?Set NOMINAL_CHECKPOINT to the frozen nominal LeWM checkpoint}"
: "${RESIDUAL_STATS_OUTPUT:?Set RESIDUAL_STATS_OUTPUT to the centered-statistics output}"

DATA_CONFIG="${DATA_CONFIG:-ogb_quadruple_stack}"
MODEL_SEED="${MODEL_SEED:-3072}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-64}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-4}"
STATS_DEVICE="${STATS_DEVICE:-cuda}"
RESOLVED_CONFIG="${RESOLVED_CONFIG:-$(dirname "$RESIDUAL_STATS_OUTPUT")/config.yaml}"

mkdir -p \
  "$(dirname "$EXECUTION_GATE_FILE")" \
  "$(dirname "$RESIDUAL_STATS_OUTPUT")" \
  "$(dirname "$RESOLVED_CONFIG")"

# Parse JSON structurally.  This prevents formatting changes or shell quoting
# from accidentally opening the training gate.
"$PIXI_BIN" run --locked python - "$PILOT_GATE_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    gate = json.load(handle)
if gate.get("status") != "pass":
    raise SystemExit(f"Scientific gate is not pass: {path}")
print(f"scientific_gate=pass path={path}")
PY

if [[ -e "$EXECUTION_GATE_FILE" ]]; then
  "$PIXI_BIN" run --locked python - \
    "$PILOT_GATE_FILE" "$EXECUTION_GATE_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    evidence = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    record = json.load(handle)
if record.get("evidence") != evidence:
    raise SystemExit("Existing execution gate does not match scientific evidence")
print(f"execution_gate=reused path={sys.argv[2]}")
PY
else
  "$PIXI_BIN" run --locked python scripts/record_gate.py \
    --gate quadruple_stack_pilot \
    --evidence "$PILOT_GATE_FILE" \
    --output "$EXECUTION_GATE_FILE" \
    --passed
fi
"$PIXI_BIN" run --locked python scripts/verify_gate.py "$EXECUTION_GATE_FILE"

"$PIXI_BIN" run --locked python scripts/eval/resolve_train_config.py \
  --output "$RESOLVED_CONFIG" \
  "data=$DATA_CONFIG" \
  "+data.dataset.path=$DATASET_PATH" \
  "data.split_manifest=$SPLIT_MANIFEST" \
  "seed=$MODEL_SEED" \
  "wandb.enabled=false"

print_run_metadata
echo "pilot_gate=$PILOT_GATE_FILE"
echo "execution_gate=$EXECUTION_GATE_FILE"
echo "dataset=$DATASET_PATH"
echo "split_manifest=$SPLIT_MANIFEST"
echo "nominal_checkpoint=$NOMINAL_CHECKPOINT"
echo "residual_stats=$RESIDUAL_STATS_OUTPUT"

srun "$PIXI_BIN" run --locked python -u scripts/eval/compute_residual_scale.py \
  --checkpoint "$NOMINAL_CHECKPOINT" \
  --config "$RESOLVED_CONFIG" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output "$RESIDUAL_STATS_OUTPUT" \
  --batch-size "$STATS_BATCH_SIZE" \
  --num-workers "$STATS_NUM_WORKERS" \
  --device "$STATS_DEVICE"
