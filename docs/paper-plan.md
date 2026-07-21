# Paper Plan: Flow-Matched Residual Kernels

Last updated: 2026-07-19

Companion to `docs/project-plan.md` (engineering tracker) and
`config/studies/complex_stochastic.yaml` (preregistered gates). This file
records the submission framing and the experiment matrix.

## Framing

Two co-contributions:

1. **Method.** A conditional flow-matched residual kernel plus a small explicit
   GRU filter turns a frozen deterministic latent world model (LeWM) into a
   calibrated, memory-aware stochastic transition kernel, cheap enough for
   risk-sensitive particle MPC (8 particles, NFE 4, common random numbers,
   mean/CVaR objectives).
2. **Evaluation protocol.** Exact simulator forks (contexts x realizations
   under resampled hidden physics modes, commanded-action audit, hashed
   provenance) provide ground-truth conditional distributions, so stochastic
   world models are scored with proper scoring rules against the true kernel —
   not only marginal likelihood or downstream return.

Central causal chain the paper must demonstrate:
**calibrated residual distribution → hidden-mode inference via memory →
better risk-sensitive control.** Every experiment attaches to one link.

## Experiment matrix

Tier 1 — required for submission:

| ID | Experiment | Infra | Status |
|----|-----------|-------|--------|
| E1 | Prediction: {cond. Gaussian, GMM, flow, GRU-flow} x 3 seeds on 512x128 forks, horizons 1/5/10/20, memory ablations correct/reset/shuffled, paired bootstrap CIs | `evaluate_fork_samples.py`, `evaluate_memory_gate.py`, array `5521894` (1-seed) | 1-seed array running |
| E2 | Dual nominal: repeat E1 key heads with a **task-adapted nominal** (vanilla LeWM trained on the pilot data) to separate nominal-bias correction from stochasticity modeling | `config/ablations/quadruple_stack_nominal.tsv`, then `compute_residual_scale.py` + kernel array with new `NOMINAL_CHECKPOINT` | job prepared |
| E3 | Control: particle MPC, all kernels, mean vs CVaR, vs deterministic CEM, 100 starts x 5 seeds, privileged simulator-particle oracle ceiling, wall-clock budgets 1/4/10 s | `eval.py` stochastic_planning, `privileged_mpc.py` | gated on E1 + E9 |
| E4 | Non-regression: joint-trained vs vanilla LeWM on PushT (pred_loss within ±10%, control unchanged); extend to cube/reacher/tworoom if compute allows | existing PushT scripts | vanilla baseline still owed |
| E5 | Spread-growth diagnostic: model particle spread vs exact-fork spread at H=1/5/10/20 | fork payloads + rollout | todo |
| E9 | Memory-filter probe: mode-ID accuracy from GRU state vs number of observed transitions; memory-state norm/saturation vs update count; memory-vs-reset residual energy skill vs update depth. Decides whether the TBPTT memory-training fix is needed. **Runs before E3.** | `scripts/eval/evaluate_memory_filter.py` | **done (job 5522477)**: skill +0.10..+0.23 stable out to 200 updates, norms stable → no TBPTT needed; mode NOT decodable from state at any depth → memory is adaptive recent-context conditioning, not a mode filter. See `docs/results/quadruple_stack_memory_filter_20260719.json` |

Tier 2 — competitiveness:

| ID | Experiment | Notes |
|----|-----------|-------|
| E6 | External baselines: probabilistic ensemble of nominal predictors (PETS-style, 5 members), diffusion residual head at matched NFE, full-covariance Gaussian oracle (analysis only) | first cut if time-limited: diffusion |
| E7 | Ablations: detach variants, loss weight, EMA vs frozen stats, conditioning=none, scheduled-sampling memory, NFE in {1,4,8,16} | TSVs mostly exist under `config/ablations/` |
| E8 | Surprise/VoE study: energy-score surprise vs deterministic MSE on 7 event types | `config/studies/surprise_events.yaml`, `evaluate_surprise.py` |
| — | Calibration→control correlation: prediction energy skill vs control gain across kernels x seeds | cheap, novel |
| — | One-step distilled sampling (rectified/shortcut flow) + NFE-quality-latency Pareto | protects LeWM's fast-planning identity |

Tier 3 — stretch: RoboCasa `PickPlaceCounterToCabinet` transfer (gate 5;
replay runner still on backlog), octuple-stack transfer, dataset-scale curve
(1k vs 10k episodes).

## E2 runbook (task-adapted nominal)

All steps reuse existing ICE helpers; run from the ICE checkout with the
pilot dataset and its immutable split manifest in place.

```bash
# 1. Train the vanilla nominal on the pilot data (model seed matches the
#    kernel array). Verify pred_loss has plateaued before freezing; raise
#    MAX_EPOCHS if the validation curve is still falling.
export DATASET_PATH=$STABLEWM_HOME/ogbench/cube_quadruple_stochastic_stack.h5
export SPLIT_MANIFEST=$STABLEWM_HOME/ogbench/cube_quadruple_stochastic_stack.split.json
ABLATION_FILE=config/ablations/quadruple_stack_nominal.tsv \
ABLATION_SEEDS=13041 DATA_CONFIG=ogb_quadruple_stack \
MAX_EPOCHS=30 BATCH_SIZE=64 \
  scripts/slurm/ice/submit_ablation_array.sh

# 2. Centered statistics from the new frozen nominal (immutable artifact).
#    Reuses the pilot scientific gate; records/verifies the execution gate.
export PILOT_GATE_FILE=<strict pilot gate JSON (job 5521744 output)>
export EXECUTION_GATE_FILE=$STABLEWM_HOME/ice/gates/quadruple_stack_pilot.json
export NOMINAL_CHECKPOINT=<nominal_object.ckpt from step 1>
export RESIDUAL_STATS_OUTPUT=$STABLEWM_HOME/ice/stats/quadruple_task_adapted_stats.pt
MODEL_SEED=13041 scripts/slurm/ice/submit_residual_stats.sh

# 3. Kernel array against the task-adapted nominal (gate-verified submit).
export RESIDUAL_SCALE_PATH=$RESIDUAL_STATS_OUTPUT
GATE_FILE=$EXECUTION_GATE_FILE \
ABLATION_FILE=config/ablations/quadruple_stack_prediction.tsv \
ABLATION_SEEDS=13041 DATA_CONFIG=ogb_quadruple_stack \
MAX_EPOCHS=10 BATCH_SIZE=64 \
  scripts/slurm/ice/submit_gated_array.sh
```

## E9 runbook (memory-filter diagnostic)

After the GRU-flow head from array `5521894` (or any successor) completes:

```bash
scripts/slurm/ice/submit_memory_filter.sh <flow_gru_memory_object.ckpt>
```

Reads the held-out val episodes, replays them into the GRU exactly as
`ResidualWorldModelPolicy` does, and writes probe-accuracy / norm /
energy-skill curves versus update count plus a raw-state payload for the
posterior-sharpening figure. Interpretation: if `memory_energy_skill_vs_reset`
or probe accuracy degrades beyond the trained update count (window allows at
most `history_size - 1 = 2` teacher-forced updates), implement the burn-in
(TBPTT) memory-training option before E3 control experiments.

## Key figures

1. **Slip bimodality**: at a grasp-onset context, fork ground-truth futures
   (slip vs no-slip clusters in probe space) vs flow samples vs Gaussian blur
   vs memory-flow sharpening after 1–2 observed post-contact transitions.
2. **Memory mechanism** (from E9): memory-vs-reset energy skill vs update
   count (stable, positive to 200 updates) next to the null mode-probe curve
   at the majority baseline — the memory adapts to recent residual context
   rather than identifying the hidden mode. Rerun against the task-adapted
   nominal to test whether mode identifiability emerges once nominal bias
   stops dominating residuals.
3. **Calibration over horizon** (from E5): model vs fork spread growth.
4. **NFE/latency Pareto** and control success vs wall-clock budget.

## Sequencing

1. Let ICE array `5521894` finish → 1-seed prediction gate.
2. E9 memory-filter diagnostic on the GRU-flow checkpoint. If long-history
   memory degrades → implement config-gated TBPTT burn-in training before E3.
3. Train task-adapted nominal (E2, independent of everything else), then
   centered stats + kernel array against it.
4. Three-seed E1/E2 arrays → memory gate → E3 control → Tier 2.

Target venues from 2026-07-19: ICLR 2027 (~late Sept abstract) or ICRA 2027
(mid-Sept). Tier 1 + E6(diffusion)/E7/E9 fits ~8 weeks on ICE; cut RoboCasa
first if compressed.

## Reviewer-proofing already in place

Paired bootstrap CIs, episode-disjoint immutable split manifests, dataset /
checkpoint / statistics SHA-256 provenance on every artifact, preregistered
gates in `config/studies/complex_stochastic.yaml`, commanded-action audit
columns, privileged oracle separated from deployable kernels. Lean on these in
the reproducibility statement.
