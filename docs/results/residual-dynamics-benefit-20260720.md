# How Much Does Modeling Residual Dynamics Buy? (E2 + E9, 2026-07-20)

Status: first single-seed result on the quadruple-stack pilot. Prediction-level
only; control-level benefit (E3) is unmeasured. Companion JSONs live beside
this file; experiment-log entries are in `docs/project-plan.md`.

## Question

Quantify the benefit of the stochastic residual kernel over the **deterministic
JEPA baseline** (vanilla LeWM, zero residual), separating the rungs of the
ladder so no rung's gain is silently attributed to a higher one:

1. deterministic → naive whitened Gaussian noise (no learning),
2. → learned conditional kernel,
3. → flow instead of Gaussian,
4. → GRU memory,
5. → control/task-level gain.

## Setup and provenance

- Task: OGBench quadruple task 5 (four-high stacking), strong hidden physics
  (8 episode-constant modes: mass / surface friction / pad friction, plus
  transient slips).
- Dataset: `quadruple_stack_strong_1000_seed13041.h5`, 1,000 episodes /
  383,766 transitions, SHA `6adb1851...c0f6`; immutable episode-disjoint
  8/1/1 split (split SHA `3f0f97ce...eca08`).
- **Task-adapted nominal** (ICE job 5522476): vanilla LeWM, 30 epochs, batch
  64, seed 13041. Validation `pred_loss` plateaued at **0.00227** — 25x below
  the official `lewm-cube` checkpoint's 0.0559 on the same data. Checkpoint
  SHA `f52d4417...b514e8`.
- Centered statistics (job 5522716, 291,925 train clips): residual mean norm
  **0.0484** vs **1.1098** for the official nominal (23x less systematic
  bias); mean diagonal scale **0.0280** vs **0.4941**. Residuals against the
  task-adapted nominal are close to pure aleatoric noise.
- Kernel heads (gated array 5522721): conditional diagonal Gaussian,
  memoryless flow, GRU-memory flow; one frozen nominal, shared statistics,
  10 epochs, batch 64, seed 13041, L40S.
- Evaluations: held-out val episodes; one-step latent residuals, 2,048
  targets, 16 samples/context, flow NFE 8 (jobs 5522758-60); E9 memory-filter
  replay diagnostic, 100 episodes, depths 1-200 (job 5522757).

## Results: the benefit ladder vs deterministic JEPA

Energy-score skill is `1 − ES(model)/ES(deterministic)`; positive = better
than the zero-residual baseline.

| Rung | Model | Skill vs deterministic | Increment over previous rung |
|---|---|---|---|
| 0 | Deterministic LeWM | 0 (by definition) | — |
| 1 | Unit Gaussian in whitened space (no learning) | **+23.1%** | +23.1 pts |
| 2 | Conditional diagonal Gaussian | **+32.3%** | +9.2 pts |
| 2 | Memoryless flow | **+28.8%** | (+5.7 pts) |
| 4 | GRU-memory flow | **+30.3%** | +1.5 pts over memoryless flow |
| 5 | Control (particle MPC) | **unmeasured** | E3 not yet run |

Supporting distributional metrics (same evaluation):

| Metric | Unit Gaussian | Cond. Gaussian | Flow | GRU flow |
|---|---|---|---|---|
| Covariance rel. Frobenius (lower better) | 0.906 | 0.874 | 0.774 | **0.769** |
| Energy score (lower better) | 10.84 | **9.55** | 10.04 | 9.83 |
| Quantile ECE | 0.0242 | 0.0252 | **0.0239** | 0.0241 |
| 90% interval coverage (target 0.90) | 0.809 | 0.793 | 0.814 | 0.811 |
| Stochastic skill vs residual-mean-only | 0.255 | 0.258 | 0.258 | 0.259 |

Reading:

- **Largest single share of the benefit is rung 1**: simply acknowledging
  noise at the right (whitened) scale recovers 23 of the ~30 points. Learned
  conditioning adds a solid but smaller increment (+6..9 pts).
- **The gain is genuinely distributional**, not mean-correction: sampling
  beats the residual predictive mean alone by ~26% for every kernel.
- **The flow has not separated from the conditional Gaussian one-step** on a
  well-fit nominal: the Gaussian wins raw energy score; the flow's edge is
  correlation structure (0.77 vs 0.87 covariance error). The full-covariance
  Gaussian oracle (E6) is required before claiming non-Gaussianity matters;
  the flow's real chance is the exact-fork (conditional, multi-step) gate.
- **Watch item**: 90% coverage is ~0.81 for all heads, below the
  preregistered 0.87-0.93 gate band.
- Deterministic prediction quality is untouched by construction (frozen
  nominal), so these gains come at no cost to the base model.

## E9: what the memory actually does (replicated across nominals)

Memory-vs-reset energy skill and hidden-mode probe accuracy versus replay
depth (task-adapted nominal, job 5522757):

| Depth (updates) | 1 | 2 | 5 | 12 | 30 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| Memory-vs-reset skill | +.24 | +.27 | +.26 | +.24 | +.17 | +.09 | +.16 | +.05 | +.08 |
| Mode probe (exact; majority = 0.18) | .16 | .22 | .14 | .12 | .12 | .12 | .18 | .06 | .08 |

- Skill is strongest for the most recent ~12 transitions and decays but stays
  positive (all paired CIs > 0) out to 200 updates, despite training with at
  most 2 teacher-forced updates. States remain stable (norms 4-7, <=4%
  saturated units). No burn-in/TBPTT fix is required for stability.
- The hidden mode is **not decodable** from the memory state at any depth —
  under either the official nominal (job 5522477) or the task-adapted one.
  Linear (regularization-swept) and MLP probes sit at the majority baseline.
  The bias-masking hypothesis is rejected.
- Conclusion: the GRU is **adaptive recent-context conditioning**, not a
  Bayes filter over modes. Paper framing updated accordingly
  (`docs/paper-plan.md`, figure 2).

## Scientific takeaways

1. **Nominal bias dominated the original residuals** (mean norm 1.11 → 0.05
   after task adaptation), yet learnable aleatoric structure survives the
   cleanup — the E2 dual-nominal design separates bias correction from
   stochasticity modeling and is worth a paper section on its own.
2. **Residual dynamics deliver a real but moderate one-step distributional
   gain (~30% proper-score skill)** over deterministic JEPA, most of it from
   correctly-scaled noise; conditioning adds the rest; memory helps most for
   recent context.
3. **The decisive comparisons are still ahead**: the exact-fork conditional
   gate (where correlation structure and multimodality actually matter) and
   E3 control — the task-level number that ultimately justifies the method.

## Artifacts

- `task_adapted_conditional_gaussian_residual_eval_20260720.json`
- `task_adapted_flow_residual_eval_20260720.json`
- `task_adapted_flow_gru_memory_residual_eval_20260720.json`
- `task_adapted_flow_gru_memory_memory_filter_20260720.json`
- `quadruple_stack_memory_filter_20260719.json` (official-nominal E9)
- Raw memory-state payloads remain on ICE under `data/ice/evaluations/`.
