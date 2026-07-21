# Next Steps

Last updated: 2026-07-19

Paper framing and experiment matrix: `docs/paper-plan.md` (E1–E9 tiers).

## Now

1. Let prediction array `5521894` finish, then run the E9 memory-filter
   diagnostic (`scripts/eval/evaluate_memory_filter.py`) on the GRU-flow
   checkpoint before any control experiments.
2. Train the task-adapted nominal (E2): vanilla LeWM on the quadruple-stack
   pilot, then centered statistics and a kernel array against it.
3. Run Gate 1: reproduce deterministic PushT and single-cube OGBench within
   five success points using the corrected evaluator.
2. Run the 1,000-episode double-stack pilot at the ordered strong/medium/mild
   levels; select the first level passing the success and mode-separation gate.
3. Compute one centered residual-statistics artifact from the frozen nominal
   checkpoint, then train conditional Gaussian, memoryless flow, and GRU-memory
   flow heads on identical episode manifests and targets.

## Next Experiment Batch

1. Build 512-context, 128-future exact forks for double-cube stacking.
2. Gate on horizon-10/20 memory energy skill and correct-memory improvements
   over memoryless, reset-memory, and shuffled-memory flow.
3. Only after that gate, run three-seed double-stack control and transfer to
   triple-cycle/RoboCasa annotated subtask windows.

## Implementation Backlog

- Add the asset-backed RoboCasa replay runner after macros/assets are installed.
- Fit and validate cube/robot physical probes before enabling collateral cost.
- Connect the implemented privileged simulator-particle cost object to the
  selected validation-tuned solver configuration.
- Freeze wall-clock-specific CEM candidate/iteration settings on validation.

## Decision Points

- Keep `detach_condition=true` for the first fair comparison, then ablate joint
  gradients later.
- Treat `time_scale=1000` as the default for new runs; old object checkpoints
  should retain their serialized behavior through the compatibility fallback.
- Do not claim control improvement until prediction-distribution metrics and
  vanilla baseline comparisons are in place.
