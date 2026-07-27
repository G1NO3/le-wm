# LeWM Residual-Flow Research Fork

This repository extends
[LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm) with optional
flow-matched residual kernels for stochastic robotic world models. The
deterministic JEPA predictor remains the nominal dynamics model; a conditional
residual model learns uncertainty around its latent transitions.

The implementation preserves vanilla LeWM behavior unless
`loss.residual_flow.enabled=true`.

<p align="center">
  <img src="assets/lewm.gif" width="80%" alt="LeWorldModel rollout">
</p>

## What is included

- End-to-end LeWM latent dynamics and rollout in `jepa.py`.
- Conditional flow-matched latent residuals in `residual_flow.py`.
- Joint nominal and residual training in `train.py`.
- Optional recurrent residual memory and stochastic particle rollouts.
- Conditional Gaussian, memoryless flow, and GRU-flow baselines.
- Exact simulator-fork evaluation, calibration metrics, and particle MPC tools.
- PushT, OGBench, FetchPush, and FetchSlide experiment workflows.

The compact project history and current research status live in
[`wiki/README.md`](wiki/README.md). See
[`wiki/evaluation.md`](wiki/evaluation.md) for metric definitions and
[`docs/project-plan.md`](docs/project-plan.md) for the experiment log.

## Installation

Linux and Python 3.10 are the validated development target. Pixi is the
environment source of truth and should be used for reproducible training,
evaluation, and cluster jobs:

```bash
pixi install --locked
pixi run smoke-import
pixi run test
```

A pip-compatible file is also provided for lightweight local setup:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` records direct dependencies but is not a replacement for
the committed `pixi.lock`. Do not use ad hoc `pip install` commands inside the
Pixi environment.

The optional RoboCasa stack uses its own Python 3.11 Pixi environment:

```bash
pixi install --locked -e robocasa
pixi run -e robocasa robocasa-smoke
pixi run -e robocasa setup-robocasa
```

RoboCasa assets are downloaded separately and are not stored in this
repository.

## Data and checkpoints

Use a large writable directory for datasets, checkpoints, and evaluation
artifacts:

```bash
export STABLEWM_HOME=/path/to/stable-wm-storage
```

Dataset configs refer to HDF5 files by basename. For example,
`pusht_expert_train` resolves to
`$STABLEWM_HOME/pusht_expert_train.h5`.

Upstream pretrained LeWM checkpoints and datasets are available from the
[LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm).
Artifacts for this fork's stochastic-manipulation experiments are intentionally
kept out of Git.

## Training

Train vanilla LeWM on PushT:

```bash
pixi run train-pusht wandb.enabled=false
```

Enable the latent residual-flow objective:

```bash
pixi run train-pusht-residual wandb.enabled=false
```

For a smaller-memory residual run:

```bash
pixi run train-pusht-residual \
  loader.batch_size=32 \
  wandb.enabled=false
```

Hydra configuration lives under `config/train/`; the residual model is
configured by the `loss.residual_flow` block in
`config/train/lewm.yaml`. Checkpoints are written below `$STABLEWM_HOME`.

## Evaluation

Evaluate a residual-flow checkpoint:

```bash
pixi run python scripts/eval/evaluate_latent_residuals.py \
  --checkpoint \
  "$STABLEWM_HOME/pusht_rflow/lewm_rflow_epoch_1_object.ckpt" \
  --output "$STABLEWM_HOME/eval/pusht_rflow.json"
```

The evaluator compares learned residual samples with simple baselines using
proper scores, calibration, interval coverage, and covariance diagnostics.
Control experiments additionally compare deterministic and stochastic planning
on exact simulator forks.

For standard LeWM planning, `policy` is a checkpoint path relative to
`$STABLEWM_HOME` without the `_object.ckpt` suffix:

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

## Development

Run the fast syntax check and test suite before submitting changes:

```bash
pixi run check
pixi run test
```

Useful source locations:

| Path | Purpose |
| --- | --- |
| `jepa.py` | LeWM model, latent transitions, rollout, and residual sampling |
| `train.py` | Training loop and residual flow-matching loss |
| `residual_flow.py` | Conditional residual vector field |
| `residual_memory.py` | Recurrent residual state |
| `residual_kernels.py` | Stochastic residual baselines |
| `stochastic_metrics.py` | Distribution and calibration metrics |
| `config/` | Hydra training, evaluation, study, and ablation configs |
| `scripts/data/` | Dataset collection and validation |
| `scripts/eval/` | Prediction and control evaluation |
| `scripts/slurm/` | Cluster submission helpers |

Cluster setup and runbooks are documented in
[`docs/ice-setup.md`](docs/ice-setup.md) and
[`docs/sky1-setup.md`](docs/sky1-setup.md).

## Upstream LeWorldModel

This fork builds on
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for
environment management, planning, and evaluation, and
[stable-pretraining](https://github.com/galilai-group/stable-pretraining) for
training.

Upstream resources:

- [Paper](https://arxiv.org/abs/2603.19312)
- [Project website](https://le-wm.github.io/)
- [Checkpoints and datasets](https://huggingface.co/collections/quentinll/lewm)

If you use the upstream model, cite:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and
          LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## License

See [`LICENSE`](LICENSE). Contributions should keep residual-flow behavior
optional and preserve compatibility with upstream LeWM.
