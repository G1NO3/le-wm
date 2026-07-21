#!/usr/bin/env bash
#SBATCH --job-name=lewm-ice-smoke
#SBATCH --output=logs/ice/%x-%j.out
#SBATCH --error=logs/ice/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="$REPO_DIR/scripts/slurm/ice"
source "$SCRIPT_DIR/common.sh"
prepare_lewm_runtime
print_run_metadata

srun "$PIXI_BIN" run --locked smoke-import
srun "$PIXI_BIN" run --locked smoke-cuda
srun "$PIXI_BIN" run --locked smoke-residual
srun "$PIXI_BIN" run --locked smoke-memory
srun "$PIXI_BIN" run --locked test
