# Residual-Kernel Evaluation

Last updated: 2026-07-19

## Immediate Question

Does the conditional residual flow produce held-out latent residual samples that
match the empirical residual distribution better than a simple Gaussian baseline?

This is still a prediction-distribution test, not a control result.

For the new primary study, the question is conditional: does the flow match
128 exact simulator forks from the same contact/grasp context better than a
conditional Gaussian or two-component GMM, and does the calibrated tail help
particle MPC? `scripts/eval/evaluate_fork_samples.py` reports energy score,
sliced Wasserstein, MMD, randomized PIT, 50/80/90% coverage and width,
mean/covariance error, horizon growth, physical-probe distributions, and paired
bootstrap intervals. `scripts/eval/evaluate_surprise.py` measures AUROC/AUPRC
and false positives on valid slip/drop outcomes.

Every prediction comparison also reports three residual-explanation metrics:

- `residual_energy_skill = 1 - ES(residual model) / ES(nominal LeWM)`; positive
  values mean residual dynamics improve the paired proper score.
- `residual_mean_error_explained`; the fraction of nominal squared error removed
  by the residual model's predictive mean.
- `stochastic_energy_skill_vs_residual_mean`; whether sampled stochastic spread
  improves energy score beyond using the residual mean as a deterministic
  correction. This guards against attributing a bias-correction gain to
  stochastic modeling.

Persistent-memory comparisons additionally report:

- `memory_energy_skill = 1 - ES(GRU flow) / ES(memoryless flow)`.
- trajectory energy score over the complete future rather than isolated steps.
- lag-one residual autocorrelation and adjacent-step cross-covariance error.
- correct-memory comparisons against zero/reset memory and memory shuffled
  across both episode and hidden-physics mode.

The double-stack memory gate is automated by
`scripts/eval/evaluate_memory_gate.py`; the control gate is automated by
`scripts/eval/evaluate_memory_control_gate.py`.

## Evaluation Script

Entry point:

```bash
python scripts/eval/evaluate_latent_residuals.py \
  --checkpoint data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_epoch_1_object.ckpt \
  --output data/eval/pusht_rflow_1epoch_residual_eval.json
```

The script:

- loads the object checkpoint and its saved `config.yaml`,
- verifies and uses its immutable episode split manifest,
- computes held-out latent residuals `target = S^{-1}(z_target - z_pred)`,
- compares standard Gaussian samples against residual-flow samples,
- writes a JSON summary.

## Slurm Command

Default Sky1 submission:

```bash
scripts/slurm/submit_evaluate_pusht_residuals.sh
```

Overcap fallback:

```bash
MAX_BATCHES=32 NUM_SAMPLES=16 FLOW_STEPS=8 \
  scripts/slurm/submit_evaluate_pusht_residuals.sh \
  --partition=overcap --account=overcap --time=01:00:00
```

Expected output:

```text
data/eval/pusht_rflow_1epoch_residual_eval.json
```

Logs are under `logs/lewm-eval-resid-<jobid>.out` and
`logs/lewm-eval-resid-<jobid>.err`.

## Metrics

The JSON contains:

- `deterministic.latent_mse`: latent residual MSE of the nominal predictor.
- `deterministic.normalized_residual_mse`: average normalized residual energy.
- `gaussian.cov_relative_frobenius`: covariance mismatch for the diagonal
  Gaussian baseline.
- `flow.cov_relative_frobenius`: covariance mismatch for flow samples.
- `gaussian.quantile_ece` and `flow.quantile_ece`: per-dimension calibration.
- `interval_90_coverage`: empirical coverage of the central 90% sample interval.
- `flow.eval_fm_loss`: held-out flow-matching regression loss.
- `residual_energy_skill`, `residual_mean_error_explained`, and
  `stochastic_energy_skill_vs_residual_mean`: paired with/without-residual
  attribution metrics defined above.
- `nfe`: number of vector-field evaluations per sample.

## How to Interpret

Healthy plumbing:

- script completes without NaNs or shape errors,
- JSON is written,
- `num_targets` is nonzero,
- flow metrics are present,
- `flow.eval_fm_loss` is finite.

Useful residual model signal:

- `flow.cov_relative_frobenius < gaussian.cov_relative_frobenius`,
- `flow.quantile_ece <= gaussian.quantile_ece`,
- `flow.interval_90_coverage` is closer to `0.90` than the Gaussian baseline.

Research-grade evidence still requires:

- a vanilla LeWM baseline at the same budget,
- longer residual-flow training,
- multiple random seeds,
- planning or weak-metric evaluation tied to task cost.

The old pooled covariance result below is retained only as historical smoke
evidence. It is not the primary metric for the complex stochastic study.

## First Result

Run:

```text
Slurm job: 3080285
Checkpoint: data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_epoch_1_object.ckpt
Output: data/eval/pusht_rflow_1epoch_residual_eval.json
```

Summary:

- The script completed successfully on `wu-lab` in `00:02:05`.
- `flow.cov_relative_frobenius = 0.4756399989128113`.
- `gaussian.cov_relative_frobenius = 0.8401789665222168`.
- `flow.interval_90_coverage = 0.8628132939338684`.
- `gaussian.interval_90_coverage = 0.8166148066520691`.
- `flow.quantile_ece = 0.02631089650094509`.
- `gaussian.quantile_ece = 0.024060126394033432`.
- `flow.eval_fm_loss = 1.2551902532577515`.

Interpretation:

The first residual-flow checkpoint has a real distributional signal: covariance
matching improves substantially and coverage moves closer to 0.90. Calibration
by quantile ECE is not yet better than Gaussian, so the next run should not be
framed as solved. It is a good reason to continue with a clean same-budget
baseline and a post-`time_scale=1000` residual-flow rerun.
