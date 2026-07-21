# Agent Instructions

This repository is a fork of LeWorldModel used to prototype Flow-Matched
Residual Kernels for robotic world models. The current research direction is:
jointly train LeWM's deterministic latent world model with a conditional
flow-matched residual model, turning deterministic latent transitions into a
stochastic transition kernel.

## Project Context

- Primary branch for this work: `latent-residual-flow`
- Fork remote: `git@github.com:fei-yang-wu/le-wm.git`
- Start future context refreshes from `wiki/README.md`, then read
  `wiki/status.md` and `wiki/evaluation.md` before touching code.
- Main implementation files:
  - `jepa.py`: LeWM model, rollout, residual sampling helpers
  - `train.py`: training loop and residual flow-matching loss
  - `residual_flow.py`: conditional vector field for latent residuals
  - `config/train/lewm.yaml`: residual-flow Hydra config
- Planning and progress tracker: `docs/project-plan.md`
- Wiki front door and compact project memory: `wiki/`

## Development Principles

- Keep changes small and compatible with upstream LeWM.
- Preserve vanilla LeWM behavior unless `loss.residual_flow.enabled=true`.
- Prefer latent-space experiments before pixel-space or full state-space models.
- Treat the deterministic predictor as the nominal dynamics model and the flow as
  a residual uncertainty model.
- Keep the stochastic component optional during training and inference.
- Record major experiment decisions, runs, and results in `docs/project-plan.md`.

## Environment

Pixi is the environment source of truth. Install the locked environment with:

```bash
pixi install --locked
pixi run smoke-import
```

`pixi.toml` explicitly pins the validated `datasets==2.21.0` release and the
lockfile freezes all transitive packages. Do not mutate `.pixi` with ad hoc
`pip install` commands; update and commit the manifest and lockfile instead.

The full `stable-worldmodel[train,env]` extra may fail on newer Python packaging
toolchains because of the legacy `gym==0.21` dependency. Use the train-only
extra for model development unless environment rollout is needed.

Use a large writable data directory for datasets and checkpoints:

```bash
export STABLEWM_HOME=/path/to/stable-wm-storage
```

On sandboxed machines, redirect caches into the repo or another writable
directory:

```bash
export XDG_CACHE_HOME=.cache
export HF_HOME=.cache/huggingface
export MPLCONFIGDIR=.cache/matplotlib
```

## Useful Commands

Vanilla LeWM training:

```bash
pixi run train-pusht wandb.enabled=false
```

Latent residual-flow training:

```bash
pixi run train-pusht-residual \
  wandb.enabled=false
```

Smaller-memory run:

```bash
pixi run train-pusht-residual \
  loader.batch_size=32 \
  wandb.enabled=false
```

Syntax check:

```bash
pixi run check
```

Slurm smoke job on `sky1`:

```bash
scripts/slurm/submit_smoke_import.sh
```

Residual distribution evaluation on `sky1`:

```bash
scripts/slurm/submit_evaluate_pusht_residuals.sh
```

The default `sky1` Slurm target is `partition=wu-lab`, `qos=short`,
`gpus-per-node=a40:1`, and `cpus-per-task=6`.

If `wu-lab` is busy, use:

```bash
scripts/slurm/submit_evaluate_pusht_residuals.sh --partition=overcap --account=overcap
```

## Experiment Hygiene

- Log dataset, commit hash, GPU type, batch size, epochs, residual-flow config,
  and checkpoint path for every run.
- Record major status changes in `wiki/status.md` and keep experiment-specific
  details in `docs/project-plan.md`.
- Compare against vanilla LeWM before interpreting stochastic improvements.
- Track both prediction metrics and control metrics.
- Do not overwrite or delete checkpoints unless explicitly requested.
- Avoid adding large data, checkpoints, caches, or virtual environments to git.

## Open Technical Questions

- Should residual targets be detached during joint training, or should gradients
  from the flow objective update the deterministic predictor?
- Should the residual scale remain an EMA diagonal scale, be precomputed, or be
  learned conditionally?
- Should stochastic rollout be used directly in MPC, or first evaluated only as a
  calibration/prediction model?
- How should one-step and multi-horizon kernels be made consistent?
