# ICE Setup and Ablation Jobs

This repository uses Pixi for the Python environment and Slurm for GPU jobs on
Georgia Tech PACE ICE. The committed `pixi.lock` is the environment source of
truth for workstations and compute nodes.

ICE account, partition, and QOS access are assigned outside this repository.
Pass those values to each submit helper rather than committing a user-specific
allocation to the batch scripts.

## One-Time Login-Node Setup

Install Pixi without administrator access if it is not already available:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
```

Clone the fork and install exactly what is recorded in `pixi.lock`:

```bash
git clone git@github.com:fei-yang-wu/le-wm.git
cd le-wm
git switch latent-residual-flow
pixi install --locked
pixi run smoke-import
pixi run test
```

Choose a persistent, high-capacity ICE path for datasets and checkpoints. Do
not use node-local temporary storage for the only copy of a checkpoint.

```bash
export STABLEWM_HOME=/path/to/persistent/le-wm-data
export XDG_CACHE_HOME=$STABLEWM_HOME/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
export HF_DATASET_REPO=fei-yang-wu/lewm-stochastic-manipulation
export MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib
scripts/data/download_pusht.sh
```

These environment variables are inherited by Slurm under its normal
`--export=ALL` behavior. Add them to the ICE shell startup file if desired.
Authenticate once on ICE with a write-scoped Hugging Face token before a
publication job. Pass the token through the site's supported secret mechanism;
never write it into this repository or an sbatch file. A credential saved by
`hf auth login` is also supported, provided the publication job receives the
same `HF_HOME` used at login. Validated artifacts are published with
`scripts/slurm/ice/submit_hf_dataset_publish.sh`.

Download by immutable Hub commit or tag and verify the HDF5 hash before use:

```bash
pixi run python scripts/data/download_hf_dataset.py \
  --repo "$HF_DATASET_REPO" \
  --revision HUB_COMMIT \
  --include 'ogbench/double_stack/**' \
  --output "$STABLEWM_HOME/hf"
```

## Discover the ICE Allocation

PACE allocations differ by course or project. Inspect the values available to
the current account before submitting:

```bash
sinfo -o '%P %G %l %a'
sacctmgr show assoc user="$USER" format=Account,Partition,QOS
```

Use the resulting values as `--account`, `--partition`, and, when required,
`--qos` arguments below. The batch files request one generic GPU with
`--gres=gpu:1`; a command-line `--gres` can select a GPU type when ICE exposes
typed GRES names.

## GPU Smoke Job

Run this before training after every new checkout or lockfile change:

```bash
scripts/slurm/ice/submit_smoke.sh \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION
```

The job imports the complete training stack and executes a ResidualFlow tensor
operation on CUDA. Its log is written under `logs/ice/`.

## Ablation Array

The default matrix is
`config/ablations/pusht_residual_flow.tsv`. It compares vanilla LeWM, detached
flow training, target-gradient and condition-gradient variants, fully joint
gradients, and a lower flow-loss weight.

Submit one seed for a quick comparison:

```bash
MAX_EPOCHS=10 \
scripts/slurm/ice/submit_ablation_array.sh \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION
```

Submit the Cartesian product of the matrix and three seeds:

```bash
ABLATION_SEEDS=3072,6144,12288 \
MAX_EPOCHS=10 \
BATCH_SIZE=64 \
WANDB_ENABLED=false \
scripts/slurm/ice/submit_ablation_array.sh \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION
```

Useful environment overrides are:

- `ABLATION_FILE`: alternate pipe-delimited matrix.
- `ABLATION_SEEDS`: comma-separated integer seeds.
- `MAX_EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`: training budget.
- `WANDB_ENABLED`: enable or disable Weights & Biases.
- `RUN_SUBDIR_PREFIX`: checkpoint namespace under `STABLEWM_HOME`.
- `DATA_CONFIG`: Hydra data configuration, default `pusht`.

Each task logs its commit hash, host, GPU, Pixi version, exact Hydra overrides,
seed, and output directory. Outputs default to:

```text
$STABLEWM_HOME/ice/ablations/<ablation>/seed_<seed>/
```

For the complex stochastic study, use the gated wrapper so a large array cannot
run before its prerequisite evidence is marked passed:

```bash
GATE_FILE="$STABLEWM_HOME/gates/gate2.json" \
ABLATION_FILE=config/ablations/double_stack_prediction.tsv \
NOMINAL_CHECKPOINT="$STABLEWM_HOME/nominal/double_stack_object.ckpt" \
RESIDUAL_SCALE_PATH="$STABLEWM_HOME/nominal/double_stack_residual_scale.pt" \
scripts/slurm/ice/submit_gated_array.sh \
  --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION
```

Create a gate record from reviewed evidence with `scripts/record_gate.py`.
Passing `--passed` is an explicit research decision; the script hashes the
evidence but does not decide whether scientific criteria were met.

RoboCasa uses `pixi install --locked -e robocasa`. Run `setup-robocasa` and
download its assets only on persistent storage. Neither operation is performed
by installation or by a Slurm submission helper.

## Residual Evaluation

After a residual-flow array task completes:

```bash
scripts/slurm/ice/submit_residual_evaluation.sh \
  "$STABLEWM_HOME/ice/ablations/rflow_detached/seed_3072/lewm_rflow_detached_seed_3072_epoch_10_object.ckpt" \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION
```

Set `OUTPUT`, `MAX_BATCHES`, `NUM_SAMPLES`, or `FLOW_STEPS` in the environment
to override evaluation defaults. JSON results default to
`$STABLEWM_HOME/ice/evaluations/`.

## Updating ICE from This Checkout

Commit and push locally, then update the cluster clone:

```bash
git push origin latent-residual-flow
```

On ICE:

```bash
git pull --ff-only
pixi install --locked
```

Run the GPU smoke job whenever `pixi.toml`, `pixi.lock`, or the model imports
change. Avoid modifying the cluster environment with `pip install`; update the
Pixi manifest and lockfile in the repository instead.
