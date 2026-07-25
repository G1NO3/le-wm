# Residual-Kernel Evaluation

Last updated: 2026-07-23

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

## Confirmed FetchPush Decision Result (2026-07-22)

The obstacle-free, receding-horizon full-task controller is a documented null:
even exact low/high mode planning did not yield a robust success benefit because
the same action is usually adequate in both modes. The positive control endpoint
is therefore a push-onset commitment task in which one open-loop pulse must be
chosen before hidden friction unfolds. It uses actual MuJoCo fork success, not a
model-space proxy.

On 1,024 fresh paired contexts (2,048 exact low/high episodes), learned
persistent residual-mode selection reached 57.47% success versus 51.03% for the
same model's residual mean. The paired scene-cluster bootstrap gain is +6.45
percentage points with 95% CI [+4.59, +8.30]. The exact distribution oracle
reached 77.73% versus 60.79% for its mean (+16.94 points, CI
[+15.48, +18.41]). Both preregistered gates pass.

This is currently a mechanism/upper-bound result: the two diagnostic mode heads
used privileged friction labels during training, although planning received
only a balanced prior and never the realized mode. The next primary gate uses
the label-free conditional flow with common antithetic samples and compares
distribution-aware success scoring against the predictive mean from those same
samples. See `docs/results/fetch_push_commitment_mode_confirmation_20260722.json`.

The powered label-free flow test is now complete and null. Its 64-context
validation screen was +4.69 points, but on 1,024 fresh contexts the flow mean
reached 62.45% and distribution-aware flow reached 62.21%: -0.24 points with
paired 95% CI [-1.71, +1.22]. The exact oracle remained +17.58 points (CI
[+16.11, +19.04]). This rejects the current practice of holding a one-step
flow base sample constant over a rollout; it does not reject the task or a
sequence-level stochastic residual. See
`docs/results/fetch_push_commitment_flow_confirmation_20260722.json`.

## Label-Free Stochastic-Control Confirmation

The sequence-consistency follow-up passes. The trainer uses exact scene-pair
membership but never the realized friction label: within each pair it discovers
opposite residual outcomes from moving-object ordinary observations and keeps
one outcome assignment for the complete episode. All three training seeds pass
the H=10 exact-fork proper-score attribution gates. For fixed primary seed
26041, energy is 0.3484 for the frozen nominal and 0.2847 for stochastic
persistent outcomes; the paired gain CI is [+0.0502, +0.0775], and the sampled
distribution also beats its own mean with CI [+0.0206, +0.0253].

On the preregistered fresh control population (1,024 contexts / 2,048 exact
episodes), residual-mean action selection reached 52.34% success and stochastic
selection reached 56.15%. The +3.81-point gain has scene-cluster bootstrap 95%
CI [+1.76, +5.86], with 239 stochastic-only versus 161 mean-only successes.
Every shard was positive and both friction modes improved. This is the primary
positive control result. See
`docs/results/fetch_push_paired_modes_confirmation_20260722.json`.

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

## FetchSlide Transfer Verdict (2026-07-23)

FetchSlide cleanly separates task value from model value. On 64 exact
pre-strike contexts, the exact mean-state selector reached 0% balanced success
and the exact distribution selector reached 47.66%, a +47.66-point gain with
scene-cluster 95% CI [+44.53, +50.00]. This is a stronger oracle decision gate
than FetchPush.

The learned one-step persistent residual does not transfer. With 100 training
pairs, H=50 stochastic energy was 0.5475 versus 0.4067 nominal; its paired
gain was negative with CI [-0.2185, -0.0586]. In exact control it reached 0%
versus 3.91% for its residual mean, while the exact oracle retained a
+42.97-point gain. The control gate fails.

The failure survives two diagnostic interventions. Increasing collection to
500 pairs reduced nominal validation loss from 0.1381 to 0.0347, but H=50
stochastic energy worsened to 1.1273 versus 0.3090 nominal. Constraining the
two residual heads to zero mean preserved a small H1 benefit (+1.45% energy
skill) but failed from H5 onward and reached -519.94% skill at H50. This
identifies autoregressive horizon inconsistency: the heads contain useful
one-step uncertainty, but repeatedly applying a persistent mode produces
unphysical ballistic trajectories. A direct terminal/multi-horizon outcome
kernel is required before another FetchSlide control evaluation.

The architecture-faithful deterministic baseline is now also complete. A
10-epoch vanilla LeWM trained on the same 500-pair dataset, with no residual
modules, reached 7.81% native goal-image commitment success on the same 64
held-out scenes / 128 exact episodes: 1.56% under low friction and 14.06% under
high friction. Its scene-cluster 95% success interval is [3.91%, 12.50%], and
it solved neither mode jointly in any scene. The exact deterministic
mean-state oracle reached 0.78%; the exact distribution oracle reached 43.75%.
Thus original LeWM is better than that deterministic heuristic but remains
35.94 points below the uncertainty-aware upper bound. Its exact-fork latent
energy rises from 0.8560 at H1 to 13.8325 at H50 with zero interval coverage,
as expected for a single point forecast. Those latent scores must not be
numerically compared with the ordinary-state MLP scores above.

Compact evidence:

- `docs/results/fetch_slide_friction_oracle_probe_20260723.json`
- `docs/results/fetch_slide_state_fork_metrics_100pairs_20260723.json`
- `docs/results/fetch_slide_commitment_100pairs_20260723.json`
- `docs/results/fetch_slide_state_fork_metrics_500pairs_validation_20260723.json`
- `docs/results/fetch_slide_state_fork_metrics_500pairs_centered_validation_20260723.json`
- `docs/results/fetch_slide_vanilla_lewm_fork_metrics_20260723.json`
- `docs/results/fetch_slide_vanilla_lewm_commitment_20260723.json`

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
