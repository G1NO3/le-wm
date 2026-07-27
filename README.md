# Stochastic Residual Dynamics for Latent World Models

## Flow-Matched Residual Kernels on a Frozen LeWorldModel


This repository accompanies [the paper](stochastic_residual_dynamics_JEPA.pdf), which extends
[LeWorldModel (LeWM)](https://le-wm.github.io/) with a stochastic residual
transition model. The original LeWM is a fast, deterministic joint-embedding
predictive architecture trained directly from pixels. This work freezes that
nominal model and learns the distribution of what it misses.

> **Implementation branch:** the complete residual-flow implementation,
> experiments, tests, and result artifacts live on
> [`latent-residual-flow`](https://github.com/fei-yang-wu/le-wm/tree/latent-residual-flow).
> The `main` branch retains the compact deterministic LeWM base.

For current experiment status, metric definitions, and the full run log, see
[`wiki/README.md`](wiki/README.md),
[`wiki/evaluation.md`](wiki/evaluation.md), and
[`docs/project-plan.md`](docs/project-plan.md).

## Why stochastic residual dynamics?

LeWM predicts one next latent embedding for each state and action. With hidden,
episode-constant physics such as friction, mass, or contact compliance, several
different futures can be valid from the same observation. A predictor trained
with squared error returns their conditional mean, which may lie between the
physical outcomes and describe a state the system never visits.

The method in the paper preserves LeWM's prediction as a frozen nominal
transition and adds a conditional distribution over its residual:

```text
observation ──► frozen LeWM ──► nominal next state ──┐
                                                    ├──► stochastic next state
Gaussian noise ──► conditional residual flow ───────┘
```

For a dynamics variable \(x_t\), action \(a_t\), and frozen predictor
\(P_\theta\), the nominal and realized deltas are

\[
\Delta_{\mathrm{nom}} = P_\theta(x_t,a_t)-x_t,\qquad
\Delta_t=x_{t+1}-x_t.
\]

The model learns the whitened residual

\[
r_t=\sigma^{-1}\left(\Delta_t-\Delta_{\mathrm{nom}}-\mu_r\right)
\]

with a conditional flow-matching objective. At inference, a Gaussian sample is
transported through the learned velocity field and added back to the nominal
transition:

\[
x_{t+1}=x_t+\Delta_{\mathrm{nom}}+\sigma\hat r.
\]

The residual head is zero-initialized, the LeWM parameters remain frozen, and
setting the residual scale to zero exactly recovers deterministic LeWM.

## Method

The implementation includes:

- A four-layer, width-512 residual velocity MLP with a 128-dimensional
  sinusoidal flow-time embedding.
- Flow-matched sampling with 8 function evaluations for prediction experiments
  and 4 inside the planning loop.
- Fixed-scale, conditional diagonal Gaussian, Gaussian-mixture, memoryless
  flow, and recurrent-flow baselines over the same normalized targets.
- An optional functional 128-dimensional GRU that conditions predictions on
  recently observed residuals without mixing planner candidates or particles.
- Two label-free persistent residual heads for tasks whose hidden physical mode
  stays fixed throughout an episode.
- Particle MPC with eight residual particles and common random numbers across
  candidate action sequences. Mean, expected-cost, and CVaR-style selectors are
  supported.
- Exact MuJoCo simulator forks for evaluating predicted conditional
  distributions with energy score, calibration, coverage, and covariance
  diagnostics.

The deployable FetchPush and FetchSlide experiments operate on the robot's
ordinary 28-dimensional observation. They do not expose friction labels or
privileged simulator state to the learned model.

## Main results

| Evaluation | Comparison | Effect | 95% CI |
|---|---|---:|---:|
| FetchPush, 10-step exact forks, 3 seeds | Residual model vs. deterministic LeWM | **34.8%** lower energy score | [32.0, 37.3]% |
| FetchPush, sampling attribution | Residual model vs. its own predictive mean | **20.2%** lower energy score | [19.2, 21.3]% |
| Hidden-friction commitment task | Planning over predicted outcomes vs. their mean | **+3.81 pp** success | [1.76, 5.86] pp |
| Ordinary receding-horizon control | Residual model vs. deterministic LeWM | +2 pp success | [-3, 7] pp |
| Exact-physics commitment oracle | Range-aware vs. average-based planning | **+22.0 pp** context success | [14.0, 30.0] pp |

The result is deliberately narrower than “stochastic models always improve
control.” Distributional predictions improve reliably, but ordinary
step-by-step MPC shows no significant success-rate gain because frequent
re-planning already absorbs surprises. The control benefit appears when the
agent must commit before the hidden physics can be inferred.

The ablations also show that uncertainty at the correct scale provides most of
the one-step gain. Conditional flow modeling better captures correlated errors
and helps over longer rollouts. The GRU remains useful over long replayed
histories, but probing indicates that it adapts to recent context rather than
explicitly identifying the hidden friction mode.

## Quick start

Clone the fork and switch to the paper's implementation branch:

```bash
git clone https://github.com/fei-yang-wu/le-wm.git
cd le-wm
git switch --track origin/latent-residual-flow
```

The research environment is managed with
[Pixi](https://pixi.sh/) and locked in `pixi.lock`:

```bash
pixi install --locked
pixi run smoke-import
pixi run test
```

For a lightweight local setup, `requirements.txt` records the direct
dependencies. The committed Pixi lockfile remains the reproducible environment
source of truth.

Use a persistent directory for datasets, checkpoints, and evaluation artifacts:

```bash
export STABLEWM_HOME=/path/to/stable-wm-storage
```

On a restricted or shared system, it is also useful to redirect caches:

```bash
export XDG_CACHE_HOME="$STABLEWM_HOME/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
```

Upstream LeWM datasets and checkpoints are available from the
[Hugging Face collection](https://huggingface.co/collections/quentinll/lewm).
Large stochastic-manipulation artifacts from this project are intentionally
kept out of Git.

## Training

Train the deterministic PushT base model:

```bash
pixi run train-pusht wandb.enabled=false
```

Train the latent residual-flow variant:

```bash
pixi run train-pusht-residual wandb.enabled=false
```

For a smaller GPU:

```bash
pixi run train-pusht-residual \
  loader.batch_size=32 \
  wandb.enabled=false
```

The residual component is configured under `loss.residual_flow` in
`config/train/lewm.yaml`. Important options include the kernel type, flow loss
weight, frozen nominal checkpoint, immutable residual-scale artifact, gradient
detachment, recurrent memory, and sampling steps.

The paper's state-space Fetch experiments use the dedicated
`scripts/train_fetch_state_dynamics.py` and
`scripts/train_fetch_episode_modes.py` entry points. Experiment matrices live
under `config/ablations/`; cluster launchers and their gated variants are under
`scripts/slurm/ice/`.

## Evaluation

Run the lightweight correctness checks before an experiment:

```bash
pixi run check
pixi run smoke-residual
pixi run smoke-memory
pixi run test
```

Evaluate a trained latent residual checkpoint with:

```bash
pixi run evaluate-residual \
  --checkpoint /path/to/checkpoint_object.ckpt
```

The research branch includes dedicated tools for:

- creating exact-fork FetchPush and FetchSlide samples;
- scoring residual kernels and calibration;
- comparing mean and distribution-aware commitment policies;
- probing recurrent memory for hidden-mode information;
- fitting physical probes to frozen LeWM embeddings; and
- rendering and summarizing the reported experiments.

See [`docs/ice-setup.md`](docs/ice-setup.md) for the end-to-end PACE ICE and
Slurm workflow. The scripts record dataset/checkpoint hashes, fixed
episode-disjoint splits, seeds, and scene-cluster bootstrap results for
reproducibility.

## Repository map

Key files on `latent-residual-flow`:

| Path | Purpose |
|---|---|
| `jepa.py` | Frozen LeWM rollout plus stochastic residual sampling |
| `residual_flow.py` | Conditional flow-matching velocity field |
| `residual_kernels.py` | Gaussian and mixture residual baselines |
| `residual_memory.py` | Functional GRU residual context |
| `residual_policy.py` | Particle-based stochastic planning |
| `state_residual_dynamics.py` | Deployable state-space Fetch model |
| `stochastic_metrics.py` | Energy score and calibration metrics |
| `stochastic_physics.py` | Hidden-physics environments and exact forks |
| `train.py` | LeWM and latent residual training |
| `scripts/data/` | Collection, validation, splits, and provenance |
| `scripts/eval/` | Fork generation, evaluation, probes, and summaries |
| `tests/` | Residual, stochastic-physics, MPC, and split tests |

The `main` branch retains the compact upstream-style LeWM implementation in
`jepa.py`, `module.py`, `train.py`, and `eval.py`.

## Limitations

- The learned residual model does not yet collect most of the control headroom
  demonstrated by the exact-physics oracle.
- Predicted intervals are mildly overconfident.
- The memory does not recover an explicit belief over hidden friction, even
  though the information is observable from motion.
- Long-horizon FetchSlide rollouts fail: at 50 steps the residual model performs
  worse than the deterministic base, and control success does not improve.
- The reported +3.81 percentage-point commitment gain compares two planning
  strategies using the same residual model. It is not a direct success-rate
  comparison against deterministic LeWM.

## Citation

If you use this work, cite the accompanying paper:

```bibtex
@article{wu2026stochasticresidualdynamics,
  title   = {Stochastic Residual Dynamics for Latent World Models:
             Flow-Matched Residual Kernels on a Frozen LeWorldModel},
  author  = {Wu, Feiyang and Wang, Jiyi},
  year    = {2026}
}
```

This project builds on
[LeWorldModel](https://github.com/lucas-maes/le-wm). Please also cite the
original LeWM paper when using the base architecture:

```bibtex
@article{maes2026leworldmodel,
  title   = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive
             Architecture from Pixels},
  author  = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and
             LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint arXiv:2603.19312},
  year    = {2026}
}
```

## License

See [LICENSE](LICENSE).
