# Next Steps

Last updated: 2026-07-23

Paper framing and experiment matrix: `docs/paper-plan.md` (E1–E9 tiers).

## Now

1. Retain the powered distinction between kernels: ordinary one-step flow is a
   control null (-0.24 points, CI [-1.71, +1.22]), while label-free persistent
   paired-outcome residuals improve success by +3.81 points with CI
   [+1.76, +5.86] on 1,024 fresh contexts.
2. Replicate the control endpoint across independently trained paired-outcome
   head seeds 26042 and 26043; do not replace or pool away primary seed 26041.
3. Replace collection-time scene pairing with a learned episode-level latent
   variable so the same persistent outcome structure can be trained on
   unpaired trajectories. A sequence-level flow or mixture-of-flows is the
   natural continuation; one-step base-noise reuse is ruled out.
4. Treat FetchSlide as the horizon-consistency stress test: its oracle task
   value is +47.66 points, but both 100- and 500-pair one-step persistent
   residuals fail by H5-H50. Do not run another control screen until a direct
   multi-horizon prediction kernel passes against the frozen nominal.

## Next Experiment Batch

1. Treat pair-contrastive persistent residual modes as the primary positive
   control kernel and memoryless flow as the documented marginal-prediction
   baseline.
2. Replicate the frozen commitment endpoint across training seeds and report a
   crossed training-seed/context interval.
3. Add an unpaired sequence-latent model and compare it with the paired upper
   bound at equal parameter and planning budgets.
4. Train a direct FetchSlide terminal-position distribution conditioned on the
   pre-strike state and full action pulse. Evaluate it first at H=50 against
   the exact two-mode forks and its own mean; submit control only after both
   paired lower confidence bounds are positive.

## Implementation Backlog

- Add the asset-backed RoboCasa replay runner after macros/assets are installed.
- Fit and validate cube/robot physical probes before enabling collateral cost.
- Connect the implemented privileged simulator-particle cost object to the
  selected validation-tuned solver configuration.
- Freeze wall-clock-specific CEM candidate/iteration settings on validation.
- Add an exact simulator action-ranking diagnostic before another control run.
- Add a stable multi-horizon outcome head with horizons 1/5/10/50 and explicit
  loss weighting; keep the one-step transition kernel available for ablation.

## Transfer Task Ladder

Use the same entry screen for every task: exact hidden-mode forks must produce
action-dependent outcome separation, both modes must remain controllable,
preferred actions must disagree, and distribution-aware exact planning must
beat mean-state planning with a positive scene-cluster lower bound.

1. FetchSlide: lowest integration cost and strongest existing oracle value.
   Continue only with a direct multi-horizon/terminal kernel.
2. Contact assembly: peg insertion, gear meshing, and nut threading with hidden
   clearance, friction, compliance, or controller dead-zone. These naturally
   create jam/insert modes and commitment-sensitive force choices.
3. Tool and articulated-object tasks: hammering, door/drawer opening, and
   tool-mediated pulling with hidden nail/hinge/contact friction. Prefer tasks
   with exact simulator state restore and a short commitment point.
4. Locomotion traction: fixed but hidden ground friction or actuator strength
   across an episode. This is scientifically strong but requires a different
   observation/action model and longer-horizon stability.
5. Grasp/pick-place with hidden object friction or mass only after the simpler
   tasks pass. Reject variants where one conservative grip solves every mode;
   hidden variability without action-ranking disagreement has no control value.

## Decision Points

- Keep `detach_condition=true` for the first fair comparison, then ablate joint
  gradients later.
- Treat `time_scale=1000` as the default for new runs; old object checkpoints
  should retain their serialized behavior through the compatibility fallback.
- Do not claim control improvement until prediction-distribution metrics and
  vanilla baseline comparisons are in place.
- Use exact oracle action-ranking disagreement as the generic task-entry gate:
  hidden physics alone is insufficient unless distributions prefer different
  actions and improve exact success over mean-state planning.
