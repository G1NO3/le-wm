# sky1 Setup

The project is cloned on the skynet login node at:

```text
fwu91@sky1:~/flash/Research/WM
```

This is the repository root, not a parent folder containing another `le-wm`
directory.

## Current Remote Setup

- Branch: `latent-residual-flow`
- Python environment: Pixi, locked by `pixi.lock`
- Python version: Pixi-managed CPython 3.10
- Slurm binaries are available under:

```text
/opt/slurm/Ubuntu-20.04/current/bin
```

Important commands:

```bash
/opt/slurm/Ubuntu-20.04/current/bin/srun
/opt/slurm/Ubuntu-20.04/current/bin/squeue
/opt/slurm/Ubuntu-20.04/current/bin/sbatch
```

Default job settings for this project:

```text
partition: wu-lab
qos: short
gpu: a40:1
cpus-per-task: 6
```

## Environment Setup

From `~/flash/Research/WM`:

```bash
pixi install --locked
pixi run smoke-import
```

The manifest pins `datasets==2.21.0`, which avoids the old
`stable-pretraining` import failure:

```text
ImportError: cannot import name 'config' from 'datasets'
```

If Pixi is not yet installed on Sky1, use the same user-local bootstrap shown
in `docs/ice-setup.md`.

Use project-local flash storage for datasets, checkpoints, and caches:

```bash
export STABLEWM_HOME=$HOME/flash/Research/WM/data
export XDG_CACHE_HOME=$STABLEWM_HOME/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
export MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib
```

`data/` is ignored by git.

Download and decompress the PushT dataset:

```bash
scripts/data/download_pusht.sh
```

Expected output file:

```text
data/pusht_expert_train.h5
```

## Submitting a First PushT Job

Submit a small import/GPU smoke job first:

```bash
scripts/slurm/submit_smoke_import.sh
```

If `wu-lab` is busy, submit to overcap instead:

```bash
scripts/slurm/submit_smoke_import.sh --partition=overcap --account=overcap
```

Submit the default residual-flow run:

```bash
scripts/slurm/submit_pusht_residual_flow.sh
```

Use a smaller batch size for lower-memory GPUs:

```bash
BATCH_SIZE=32 scripts/slurm/submit_pusht_residual_flow.sh
```

Pass Slurm options before the script path via the submit helper if you need to
override defaults:

```bash
scripts/slurm/submit_pusht_residual_flow.sh --time=06:00:00
```

Overcap training fallback:

```bash
scripts/slurm/submit_pusht_residual_flow.sh --partition=overcap --account=overcap
```

Pass extra Hydra overrides through `EXTRA_OVERRIDES`:

```bash
EXTRA_OVERRIDES='loss.residual_flow.weight=0.3 output_model_name=lewm_rflow_w03' \
  scripts/slurm/submit_pusht_residual_flow.sh
```

Use `+` when adding a new Hydra key that is not already present in the config:

```bash
EXTRA_OVERRIDES='+trainer.limit_train_batches=2 +trainer.limit_val_batches=1' \
  scripts/slurm/submit_pusht_residual_flow.sh --partition=overcap --account=overcap
```

## Residual Distribution Evaluation

Evaluate the 1-epoch residual-flow checkpoint:

```bash
MAX_BATCHES=32 NUM_SAMPLES=16 FLOW_STEPS=8 \
  scripts/slurm/submit_evaluate_pusht_residuals.sh \
  --partition=overcap --account=overcap --time=01:00:00
```

Default checkpoint:

```text
data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_epoch_1_object.ckpt
```

Default JSON output:

```text
data/eval/pusht_rflow_1epoch_residual_eval.json
```

Override either path with `CHECKPOINT=...` or `OUTPUT=...`.

Check jobs:

```bash
/opt/slurm/Ubuntu-20.04/current/bin/squeue -u "$USER"
```

Logs are written under:

```text
logs/
```

Checkpoints are written under `data/<subdir>/`.

## Development Workflow

Use the login node for:

- cloning and pulling code,
- installing dependencies,
- preparing datasets/checkpoints,
- submitting and monitoring jobs,
- light smoke tests.

Use compute jobs for:

- training,
- evaluation rollouts,
- any GPU-heavy run.

Avoid long training commands directly on the login node.
