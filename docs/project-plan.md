# Flow-Matched Residual Kernels Plan

Last updated: 2026-07-19

Submission framing and experiment matrix: `docs/paper-plan.md`.

## Goal

Convert LeWorldModel from a deterministic latent transition model into a
stochastic, control-compatible transition kernel by jointly training:

```text
deterministic latent predictor + conditional flow-matched residual model
```

In latent form:

```math
z_{t+1} = \Phi_\psi(z_{\le t}, u_{\le t}) + S T_\theta^{z_{\le t},u_{\le t}}(\epsilon)
```

where LeWM provides the nominal predictor `Phi_psi`, and the new residual flow
learns a distribution over normalized latent residuals.

## Current Status

- **Harder primary task screened on ICE:** four-high OGBench quadruple stacking
  replaces double-stack as the primary scientific task. A 100-episode strong-
  profile screen reached 95% scripted-expert success, while the official
  deterministic LeWM reached 38% on ordinary reachable windows and 26% on
  stage-balanced post-contact windows. Exact-fork mode separation is 5.180
  pooled SD. The task gate passes, but no 1,000-episode collection or residual
  training has been submitted.

- **Environment/ICE ready locally (2026-07-18)**: added a locked Pixi Linux
  environment, verified the full training imports and CUDA tensor smoke test,
  and added portable ICE Slurm smoke, ablation-array, and residual-evaluation
  jobs. An on-cluster smoke submission is still required before training.
- Fork remote is set to `git@github.com:fei-yang-wu/le-wm.git`.
- Working branch is `latent-residual-flow`, pushed to the fork.
- Commit `ca701a0` added the first latent residual-flow training scaffold.
- Commit `7b23d64` added project instructions and this planning tracker.
- `sky1` clone path: `fwu91@sky1:~/flash/Research/WM`. Setup notes:
  `docs/sky1-setup.md`. Project data root: `~/flash/Research/WM/data`.
- Default Slurm target: `partition=wu-lab`, `qos=short`, `gpus-per-node=a40:1`,
  `cpus-per-task=6`. Overcap fallback: `--partition=overcap --account=overcap`.
- Vanilla LeWM behavior remains the default because residual flow is disabled in
  config unless `loss.residual_flow.enabled=true`.
- **M0 complete**: sky1 install verified, PushT downloaded, smoke import +
  tiny residual-flow training run passed on overcap A40
  (see Experiment Log entry `2026-04-29 / fe29db85`).
- **M1 smoke complete**: the first full-dataset, 1-epoch residual-flow PushT
  run completed on overcap A40 and saved checkpoints. The full fair comparison
  still needs a same-budget vanilla run and a post-`time_scale` residual-flow
  rerun.
- **M2 smoke complete**: residual distribution evaluation ran on the 1-epoch
  checkpoint. The flow improved covariance matching and 90% interval coverage
  versus the diagonal Gaussian baseline, but quantile ECE was slightly worse.

## Implemented

- Added `residual_flow.py` with:
  - sinusoidal flow-time embedding,
  - conditional MLP vector field,
  - zero-initialized output head for stable startup.
- Extended `JEPA` in `jepa.py` with:
  - optional `residual_flow`,
  - centered residual mean/scale statistics,
  - functional 128-dimensional GRU memory,
  - Euler residual sampling,
  - particle-specific stochastic memory rollout.
- Extended `train.py` with:
  - flow-matching residual loss,
  - residual scale updates,
  - residual flow construction from Hydra config,
  - joint objective `LeWM loss + lambda_fm * residual_fm_loss`.
- Added `config/train/lewm.yaml` options under `loss.residual_flow`.
- Added Slurm helpers under `scripts/slurm/` for PushT residual-flow training.
- Added Slurm helpers under `scripts/slurm/` for vanilla PushT training.
- Added `scripts/eval/evaluate_latent_residuals.py` for held-out latent
  residual distribution metrics.
- Added `scripts/data/download_pusht.sh` for project-local PushT data setup.
- Added `wiki/` as the compact project context front door.
- Added `pixi.toml`/`pixi.lock` and ICE Slurm helpers under
  `scripts/slurm/ice/` for reproducible multi-seed ablation studies.

## Verification So Far

- `python -m compileall jepa.py train.py residual_flow.py` passes.
- A minimal tensor smoke test for `ResidualFlow` passed after installing
  train-only dependencies.
- Full `train.py` import was blocked by dependency/version friction in
  `stable-pretraining` and `datasets` on the local Mac environment.
- Full `train.py` import passes on `sky1` after upgrading to
  `datasets==2.21.0`.
- PushT dataset is downloaded under `data/pusht_expert_train.h5`.
- Slurm smoke import job passed on overcap A40.
- Tiny residual-flow training smoke passed on overcap A40.
- One-epoch residual-flow PushT job `3030237` completed on overcap A40 in
  `01:42:04` with finite validation metrics and saved checkpoints.
- Residual evaluation script passes syntax checks locally.
- Residual evaluation job `3080285` completed on `wu-lab` in `00:02:05` and
  wrote `data/eval/pusht_rflow_1epoch_residual_eval.json`.

## Recommended Compute

Active development runs on sky1 (`partition=wu-lab`, A40) with overcap
fallback. See `docs/sky1-setup.md` and `scripts/slurm/`.

First real run target: PushT, 10 epochs, residual flow enabled.

```bash
python train.py data=pusht \
  trainer.max_epochs=10 \
  loss.residual_flow.enabled=true \
  wandb.enabled=false
```

Future portability (other 24GB+ GPUs, e.g., L40S/A100/A6000): cap
`loader.batch_size=32` and rely on the same Hydra overrides.

The primary stochastic study now starts with OGBench quadruple task 5. Use
double-stack only for compatibility/non-regression and use octuple stacking as
an optional harder transfer after the quadruple prediction and control gates.

## Milestones

### M0: Reproducible Setup

- Clone fork on a GPU/Linux machine.
- Install train dependencies.
- Download or mount LeWM datasets under `$STABLEWM_HOME`.
- Confirm vanilla LeWM training starts.
- Confirm residual-flow training starts.

### M1: First Valid Training Run

- Train vanilla LeWM on PushT for the same budget.
- Train LeWM + latent residual flow on PushT.
- Log training losses:
  - `pred_loss`
  - `sigreg_loss`
  - `residual_fm_loss`
  - total loss
- Save checkpoint paths and commit hash.

**Success criteria:**

- `residual_fm_loss` decreases monotonically over the run and ends below the
  initial `‖x − ε‖²` plateau (≈ 2 in normalized space) by a clear margin.
- Vanilla `pred_loss` of the joint run stays within ±10% of the LeWM-only
  baseline (no regression of the deterministic objective).
- Both runs reach the same epoch count without crashes; checkpoints loadable.

### M2: Prediction Distribution Evaluation

- Build an evaluation script for held-out latent residuals.
- Compare:
  - deterministic residual magnitude,
  - Gaussian residual baseline,
  - latent residual flow samples.
- Metrics:
  - latent MSE,
  - residual covariance match,
  - sample coverage,
  - per-dim quantile calibration (ECE),
  - residual energy skill versus nominal LeWM,
  - nominal squared error explained by the residual predictive mean,
  - stochastic energy skill versus the residual-mean-only correction,
  - NFE used to draw each sample,
  - weak metrics such as expected goal distance or contact indicators if
    available.

**Success criteria:**

- Flow residual covariance Frobenius error to held-out empirical covariance
  beats the per-dim Gaussian baseline by ≥ 20%.
- Per-dim quantile ECE for the flow ≤ Gaussian baseline ECE.
- Flow samples cover the empirical residual support (no obvious mode collapse
  on a 2D PCA projection).

### M3: Stochastic Planning

- Add a config switch for stochastic rollout during planning.
- Compare deterministic CEM against stochastic expected-cost CEM.
- Try risk-sensitive variants:
  - mean cost,
  - mean plus variance,
  - CVaR-like elite cost.

### M4: Joint-Training Ablations

- `detach_residual_target=true` versus `false`.
- `detach_condition=true` versus `false`.
- residual-flow loss weight sweep.
- EMA residual scale versus fixed precomputed scale.
- one-step residual kernel versus multi-horizon residual kernels.

### M5: Research-Grade Results

- Run across PushT, Cube, Reacher, and TwoRoom if compute allows.
- Compare against:
  - deterministic LeWM,
  - Gaussian residual model,
  - mixture residual model if implemented,
  - diffusion residual model if implemented.
- Report:
  - prediction quality,
  - stochastic calibration,
  - MPC return/success,
  - planning latency,
  - number of function evaluations.

## Near-Term TODO

Open items only — completed setup work has moved to **Current Status / M0**.

1. Re-run the residual-flow smoke on sky1 after the
   `SinusoidalTimeEmbedding` `time_scale` fix (commit lands in
   `residual_flow.py`); confirm `residual_fm_loss` dynamics change.
2. Add a shape-sanity unit test for `JEPA.residual_condition` covering
   `num_preds ∈ {1, 2}` and `history_size ∈ {3, 4}` so the
   `[ctx_emb, ctx_act, pred_emb]` concat is verified beyond the current
   `num_preds=1, history_size=3` happy path.
3. Decide and document the default for `loss.residual_flow.detach_condition`
   for the first real M1 run. Recommended starting point:
   `detach_condition=true` for an apples-to-apples comparison vs. vanilla
   LeWM, then ablate in M4.
4. Let the same-budget vanilla PushT job finish and record its metrics.
5. Add an optional full-covariance Gaussian oracle baseline for analysis only;
   keep the diagonal Gaussian baseline as the fair deployable baseline.
6. Expose a stochastic-rollout switch in `get_cost` / evaluation config so
   M3 planning experiments can be triggered without code changes.

## Experiment Log Template

```text
Date:
Commit:
Machine/GPU:
Dataset:
Command:
Config overrides:
Checkpoint:
Training result:
Evaluation result:
NFE (sampling):
Notes:
Next action:
```

## Experiment Log

```text
Date: 2026-07-19
Commit: 14da4a3353897682fc2859484d016dd08089a02a + uncommitted persistent-memory implementation
Machine/GPU:
  Local: MEL07876D, NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  ICE: smoke job 5521300, NVIDIA A100-PCIE-40GB
Dataset:
  Existing ten-episode double-stack plumbing dataset; not a scientific pilot.
Training result:
  Exact centered statistics computed from 1,057 frozen-nominal training clips.
  Residual mean norm 3.2903; average diagonal scale 0.00487. A frozen-nominal
  GRU-flow completed two one-batch epochs and preserved validation prediction
  MSE at 0.0559137798845768. One epoch is unsupported by the upstream cosine
  scheduler, so smoke training uses a two-epoch minimum.
Evaluation result:
  Checkpoint reload, stochastic latent rollout, and three-particle CVaR MPC
  passed. Synthetic AR(1) trajectory energy improved 0.7283 -> 0.5950 and
  lag-one error improved 0.8006 -> 0.3320 versus memoryless dynamics.
Verification:
  Locked Pixi install, compile checks, 35 tests, and smoke-memory pass locally.
  ICE job 5521300 completed in 00:04:26 with exit 0:0; CUDA imports,
  residual-flow execution, smoke-memory, and all 34 tests passed.
Notes:
  No pilot/training/ablation array submitted. The ten-episode strong profile
  still has 100% expert success and cannot satisfy Gate 2.
Next action:
  Finish ICE smoke 5521300, then collect the fixed 1,000-episode calibration
  pilot before any residual-head array.
```

```text
Date: 2026-07-19
Commit: 14da4a3353897682fc2859484d016dd08089a02a + uncommitted study implementation
Machine/GPU:
  Local: MEL07876D, NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  ICE: job 5520712, atl1-1-01-005-11-0, NVIDIA A40
Dataset:
  Temporary 10-episode OGBench double-cube stacking plumbing dataset; strong
  hidden-physics profile, episode-disjoint 8/1/1 split.
Training result:
  Two one-batch epochs completed for nominal LeWM, frozen conditional Gaussian,
  and frozen residual flow. Shared scale used 1,057 nominal training clips.
  Frozen nominal validation MSE was invariant at 0.0559137798845768.
Evaluation result:
  Plumbing only. Strong-profile expert success was 10/10, so this temporary
  sample fails the 40-80% calibration criterion and is not a pilot result.
ICE result:
  Job 5520712 completed in 00:00:31 (exit 0:0). Locked Pixi import, CUDA, and
  ResidualFlow tensor checks passed. Job 5520709 was a zero-second bootstrap
  failure that exposed and led to fixing Slurm spool-relative helper lookup.
Notes:
  No collection, training, evaluation, or array job was submitted to ICE.
  Final local suite passed 22 tests. ICE passed the synced 20-test suite before
  two evaluator-selection regression tests were added locally.
Next action:
  Run Gate 1 deterministic reproduction, then the 1,000-episode calibration
  pilot only after reviewing its fixed start/seed protocol.
```

```text
Date: 2026-04-29
Commit: fe29db85
Machine/GPU: sky1 Slurm overcap, node shakey, 1x NVIDIA A40
Dataset: PushT, data/pusht_expert_train.h5
Command:
  MAX_EPOCHS=1 BATCH_SIZE=16 NUM_WORKERS=2 \
  EXTRA_OVERRIDES="+trainer.limit_train_batches=2 +trainer.limit_val_batches=1 output_model_name=lewm_rflow_smoke subdir=smoke_pusht_rflow" \
  scripts/slurm/submit_pusht_residual_flow.sh --partition=overcap --account=overcap --time=00:30:00
Checkpoint:
  data/smoke_pusht_rflow/lewm_rflow_smoke_epoch_1_object.ckpt
  data/smoke_pusht_rflow/lewm_rflow_smoke_weights.ckpt
Training result:
  Completed 2 train batches and 1 validation batch.
  fit/loss: 35.2742
  fit/pred_loss: 0.2305
  fit/residual_fm_loss: 34.5769
  validate/loss: 9.3510
  validate/pred_loss: 0.0654
  validate/residual_fm_loss: 8.6646
Notes:
  First job failed because new Hydra trainer keys require +trainer.* syntax.
  Resubmitted with +trainer.limit_train_batches and +trainer.limit_val_batches.
Next action:
  Submit a short but real PushT residual-flow run, then add evaluation tooling.
```

```text
Date: 2026-04-29
Commit: df52ab4
Machine/GPU: sky1 Slurm overcap, node deebot, 1x NVIDIA A40
Dataset: PushT, data/pusht_expert_train.h5
Command:
  MAX_EPOCHS=1 BATCH_SIZE=128 NUM_WORKERS=6 \
  scripts/slurm/submit_pusht_residual_flow.sh --partition=overcap --account=overcap --time=02:00:00
Checkpoint:
  data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_epoch_1_object.ckpt
  data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_weights.ckpt
Training result:
  Completed job 3030237 in 01:42:04 with exit code 0:0.
  fit/loss: 1.474353551864624
  fit/pred_loss: 0.07256205379962921
  fit/residual_fm_loss: 1.2396820783615112
  fit/sigreg_loss: 1.796875
  validate/loss: 1.5134869813919067
  validate/pred_loss: 0.06748738884925842
  validate/residual_fm_loss: 1.2505505084991455
  validate/sigreg_loss: 2.1714375019073486
Evaluation result:
  Slurm job 3080285 completed in 00:02:05 with exit code 0:0.
  JSON: data/eval/pusht_rflow_1epoch_residual_eval.json
  num_targets: 6144
  latent_dim: 192
  deterministic.latent_mse: 0.06709294766187668
  deterministic.normalized_residual_mse: 1.0047131776809692
  flow.cov_relative_frobenius: 0.4756399989128113
  gaussian.cov_relative_frobenius: 0.8401789665222168
  flow.interval_90_coverage: 0.8628132939338684
  gaussian.interval_90_coverage: 0.8166148066520691
  flow.quantile_ece: 0.02631089650094509
  gaussian.quantile_ece: 0.024060126394033432
  flow.eval_fm_loss: 1.2551902532577515
NFE (sampling):
  8
Notes:
  This checkpoint predates the time_scale=1000 time-embedding default. Object
  checkpoint loading preserves old behavior with a compatibility fallback.
  The flow has a real smoke-level distributional signal: covariance matching
  and interval coverage beat the diagonal Gaussian baseline. Quantile ECE does
  not beat Gaussian yet, so this is encouraging but not conclusive.
Next action:
  Let the vanilla PushT baseline finish, then re-run residual-flow training with
  the time_scale=1000 embedding fix and detach_condition=true default.
```

```text
Date: 2026-07-19
Commit: 14da4a3 plus uncommitted persistent-memory implementation
Machine/GPU:
  Local: MEL07876D, NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  ICE: jobs 5521321/5521327, NVIDIA A100-PCIE-40GB
Dataset:
  Two-episode strong-profile double-stack collection smoke, immutable HDF5.
Validation result:
  302 transitions; commanded actions match; two hidden modes; one slip event;
  dataset SHA 26057c7d16c9bae9c11021608eb65a37bdb1bef8a72e363dfbd9d75fb6f5b9ff.
Fork result:
  Two contexts x eight independent realizations x five steps; physical state
  dimension 54; evidence SHA 23e73cbaa7e69f1d11cd282a482414b8da97be2b5481ec3e56738387aae29a5b.
Data management:
  Created private HF dataset fei-yang-wu/lewm-stochastic-manipulation. The
  validated smoke bundle is stored at Hub commit
  41b2a420d4594ad46989019b27133671c5fb39c0.
Pilot result:
  ICE job 5521332 completed 1,000 episodes / 151,286 transitions in 00:26:41.
  Expert success was 1.0 with 210 slip events and all eight hidden modes.
  Dataset SHA: 836de31d00b9ad9016a449651015d1fd2f01ea3a662f6780f2b7ca80ac7161e7.
  Job 5521372 generated 64 x 32 x H=5 physical forks; grouped mode separation
  was 2.50598 pooled SD. The gate status is fail_too_easy, so no model training
  or milder-profile collection was submitted.
Hub result:
  Compact pilot provenance is stored at commit
  39902a20a9b2ffa159de5bebee2f27cbc35c3f78. The validated 22 GB HDF5 remains
  on ICE until that host is authenticated to the private dataset repository.
Next action:
  Make failures consequential (longer slip across a full macro action or a
  recoverable release/impulse perturbation), run a small difficulty sweep, and
  repeat the 1,000-episode pilot only after a 100-episode bracket reaches the
  40-80% target. Do not train on this too-easy pilot.
```

```text
Date: 2026-07-19
Commit: 14da4a3 plus uncommitted harder-task implementation
Machine/GPU: ICE jobs 5521459/5521460/5521483/5521492/5521515, NVIDIA A100
Task: OGBench quadruple task 5, four-high stacking, strong hidden physics
Screen dataset:
  100 episodes / 36,022 transitions / 5,535,199,575 bytes.
  Expert success 0.95; all eight hidden modes; 305 contact onsets; 66 slips.
  Dataset SHA 788559b4a78fac428ea018a2b3244a9642b6af627f566206285962f036a278eb.
Fork result:
  32 contact-balanced contexts x 16 realizations x H=5.
  Within-context mode separation 5.180331707 pooled SD.
Official deterministic LeWM:
  quentinll/lewm-cube revision b0747c5002e86d2ce8f3cd8178004b97524c587d.
  100 ordinary reachable windows: 0.38 success (job 5521492).
  100 post-contact windows: 0.26 success (job 5521515).
  Contact-stage success: 0.228571 / 0.228571 / 0.333333 for stages 1 / 2 / 3.
Gate result:
  PASS. Expert/model gap 0.69 and mode separation 5.180 pooled SD.
Data management:
  Compact artifacts uploaded to private HF dataset
  fei-yang-wu/lewm-stochastic-manipulation at commit
  d8ce3c25d5e7b5b1b5f94eff1f6b99d3b49fc429.
  The 5.2 GB HDF5 remains validated on ICE pending Hub authentication there.
Notes:
  One V100 allocation (job 5521469) failed before inference because its driver
  was incompatible with the locked PyTorch build. Evaluation jobs now request
  an A100. No 1,000-episode collection or training was submitted.
Next action:
  Authenticate ICE to the private HF dataset repo, then collect and publish the
  1,000-episode quadruple pilot before computing statistics or training heads.
```

```text
Date: 2026-07-19
Commit: 14da4a3 plus uncommitted persistent-memory/stochastic-study implementation
Machine/GPU:
  ICE collection/forks: NVIDIA A100
  ICE statistics/smoke/prediction array: NVIDIA L40S
Task: OGBench quadruple task 5, four-high stacking, strong hidden physics
Pilot result:
  Job 5521684 completed 1,000 episodes / 383,766 transitions with 0.914 expert
  success, 3,086 contact onsets, 781 slips, and all eight hidden modes.
  Dataset SHA: 6adb185168216c544bbaad65d1aac04b74f7fe781fc67d624e4710890582b0f6.
Fork and gate result:
  Job 5521685 produced 64 contexts x 32 realizations x H=5; mode separation
  was 10.0942936 pooled SD and fork SHA was
  1b74bfdeac7396be55befc8fabb353df045ad9c45cbdd3fb6a9f1dc67fbaada1.
  Job 5521744 passed the strict pilot gate against deterministic LeWM success
  0.26 (expert/model gap 0.654).
Data management:
  Hub retry 5521810 published the 58.96 GB HDF5 plus validation, metadata,
  split, forks, and gate to private dataset
  fei-yang-wu/lewm-stochastic-manipulation at commit
  9e73de75ca3e5ebce9cf90b4bd2ebf300d0aa1cd. Hub LFS SHA/size match locally.
Preflight:
  Job 5521880 computed centered statistics over 291,925 train clips using the
  official frozen LeWM checkpoint. Mean SHA cb6571a3...7714d1e; scale SHA
  324be2d0...ab47607. Job 5521881 passed the two-train-batch/one-validation-
  batch conditional-Gaussian smoke.
Prediction launch:
  Gated launcher 5521882 submitted array 5521894 after both dependencies
  passed. Tasks 0/1/2 are conditional Gaussian, memoryless flow, and GRU flow;
  all use model seed 13041, 10 epochs, batch size 64, the same frozen nominal
  checkpoint, centered statistics, immutable split, and L40S hardware.
Notes:
  The initial Hub job failed due to HF_HOME redirection and the first stats job
  exposed stdout-contaminated Hydra YAML. Both infrastructure bugs were fixed
  before smoke or full training. No failed gate opened a downstream array.
Next action:
  Let array 5521894 finish, then run the exact-fork prediction gate including
  nominal/residual skill, correct/reset/shuffled memory, and horizon-10/20
  energy scores before any control or transfer jobs.
```

```text
Date: 2026-07-19
Commit: 14da4a3 plus uncommitted paper-plan/E9 implementation
Machine/GPU: ICE job 5522477, NVIDIA L40S (3m30s, exit 0:0)
Dataset: quadruple_stack_strong_1000_seed13041.h5 (SHA 6adb1851...c0f6)
Command:
  scripts/slurm/ice/submit_memory_filter.sh on
  lewm_flow_gru_memory_seed_13041_epoch_10_object.ckpt (array 5521894 task 2)
Evaluation result (E9 memory-filter diagnostic, 100 val episodes):
  Memory-vs-reset energy skill is positive at every replay depth out to 200
  updates (+0.10 to +0.23; paired CI excludes zero everywhere except a
  non-significant dip at depth 50), despite training with at most 2
  teacher-forced updates. Memory-state norms plateau at 6-7.5 with <=12%
  saturated units: no drift or blow-up, so no TBPTT fix is required for
  stability and E3 control is unblocked from the E9 side.
  However, the hidden physics mode is NOT decodable from the GRU state at any
  depth: pooled and per-depth ridge probes and an MLP probe all sit at the
  0.18 majority baseline (exact), per-axis at chance, and skill does not grow
  with observation count. JSON: docs/results/quadruple_stack_memory_filter_20260719.json.
Interpretation:
  The memory behaves as adaptive recent-context conditioning, not a
  sharpening Bayes filter over modes. The paper's memory narrative must be
  framed accordingly (or mode identifiability improved); the preregistered
  gate-3 comparison (GRU-flow vs memoryless flow head) is unaffected.
Next action:
  Let nominal-training job 5522476 finish, then centered stats and the
  task-adapted kernel array (E2 runbook in docs/paper-plan.md). Rerun this
  diagnostic against the task-adapted nominal to test whether mode
  identifiability emerges once nominal bias stops dominating residuals.
```

```text
Date: 2026-07-20
Commit: 14da4a3 plus uncommitted paper-plan/E2 implementation
Machine/GPU: ICE jobs 5522476 / 5522716 / 5522721, NVIDIA L40S
Dataset: quadruple_stack_strong_1000_seed13041.h5 (SHA 6adb1851...c0f6)
Training result (E2 task-adapted nominal, job 5522476, 04:49:19, exit 0:0):
  Vanilla LeWM, 30 epochs, batch 64, model seed 13041, immutable pilot split.
  Validation pred_loss plateaued at 0.00227 over the final epochs — roughly
  25x below the official lewm-cube nominal's 0.0559 on the same data.
  Checkpoint: data/ice/nominal_task_adapted/nominal_task_adapted/seed_13041/
  lewm_nominal_task_adapted_seed_13041_epoch_30_object.ckpt
  (SHA f52d4417...b514e8).
Statistics result (job 5522716, 00:09:00, exit 0:0):
  Centered stats over the same 291,925 train clips:
  residual mean norm 0.0484 (official nominal: 1.1098, 23x lower bias) and
  average diagonal scale 0.02802 (official: 0.49413, ~18x smaller residuals).
  Mean SHA 67aa7ed9...613ff4, scale SHA 155fb757...e6ddf7; dataset/split/
  checkpoint hashes verified. Residuals against the task-adapted nominal are
  therefore far closer to pure aleatoric noise.
Prediction launch:
  Gated array 5522721 (gate quadruple_stack_pilot, evidence SHA d771c48e...)
  trains conditional Gaussian, memoryless flow, and GRU flow against the
  frozen task-adapted nominal under
  ice/quadruple_stack_prediction_task_adapted/, 10 epochs, batch 64,
  seed 13041, L40S.
Prediction result (array 5522721, three heads, ~1h each, exit 0:0):
  Held-out one-step residual evaluation (jobs 5522758-60, 2048 targets):
  deterministic latent MSE 0.00097 (69x below the official nominal). All
  learned kernels beat the unit Gaussian (cov Frobenius 0.77-0.87 vs 0.91,
  residual energy skill 0.29-0.32 vs 0.23), so learnable aleatoric structure
  survives bias removal. GRU-flow >= memoryless flow (skill 0.303 vs 0.288,
  cov 0.769 vs 0.774); the conditional Gaussian is competitive on energy
  (9.55) — with a near-perfect nominal, one-step residuals are close to
  conditionally Gaussian, making the full-covariance oracle baseline (E6)
  important before claiming non-Gaussianity. Coverage_90 ~0.81 for all heads
  (below the 0.87-0.93 gate band): watch at the fork-based gate.
E9 rerun on the task-adapted GRU-flow (job 5522757, 03:17, exit 0:0):
  Memory-vs-reset skill is strongest at short depths (+0.24..+0.27 for 1-12
  updates) and decays but stays positive with all-positive CIs out to 200
  (+0.05..+0.16 beyond depth 30). States remain stable. The hidden mode is
  STILL not decodable from the memory state at any depth (probe at/below the
  0.18 majority baseline) — replicating the official-nominal null. The
  bias-masking hypothesis is rejected: across both nominals the GRU is
  adaptive recent-context conditioning, not a mode filter.
  JSONs: docs/results/task_adapted_*_20260720.json.
Next action:
  Run the exact-fork prediction gate (evaluate_fork_samples + memory gate)
  on the task-adapted heads, add the full-covariance Gaussian oracle to the
  analysis, and consider sampled_history_probability>0 / TBPTT to improve
  long-depth memory retention before three-seed arrays and E3 control.
```

```text
Date: 2026-07-20
Commit: 14da4a3 plus uncommitted E3 control implementation
Machine/GPU: ICE jobs 5522961/5522969/5522970, NVIDIA A100-40GB
Task: quadruple-stack post-contact reachable windows, strong hidden physics.
Setup: 10 contexts x 1 sim seed (SMOKE, n=10), eval seed 12041, budget 50,
  goal offset 25, identical hash-pinned context selection across policies.
  Task-adapted nominal frozen; kernels from array 5522721.
E3 smoke result (success / worst-decile distance / latency per episode):
  deterministic CEM (baseline)         40%  0.215  3.6 s
  memoryless flow, 8 particles, mean   50%  0.186  8.3 s
  GRU flow, 8 particles, mean          40%  0.157  8.9 s  (reset-memory)
Interpretation:
  First downstream-planning signal. Stochastic particle MPC improves the
  worst-decile (risk) distance for both kernels vs the deterministic baseline
  (0.186 and 0.157 vs 0.215); memoryless flow also nudged success (+1 episode
  at n=10, not significant). Encouraging but underpowered; the pilot (100
  evals/head) decides.
Known issue:
  Online residual memory is disabled. The env reports action history
  [batch, 1, 5] (last raw action) and pixels [batch, 1, 3, 224, 224), but the
  encoder expects frameskip-chunked [batch, time, 25]. residual_policy.py now
  logs this and falls back to reset memory, so the GRU head currently runs as
  its reset-memory ablation. A real fix must buffer the policy's own executed
  action chunks and pixel history at model rate before online memory updates.
Next action:
  Fix online action/pixel chunking in ResidualWorldModelPolicy, then run the
  100-eval pilot: deterministic vs memoryless-flow-mean vs GRU-mean vs
  GRU-CVaR, all on identical contexts, reported vs the deterministic baseline.
```

## Key Design Decisions

- Start in latent space, not pixels.
- Keep residual flow optional and off by default.
- Use exact train-only centered residual statistics from the frozen nominal
  checkpoint; keep the EMA path only for small plumbing runs.
- Keep temporal persistence simple: one deterministic GRU state, no hierarchy,
  mode labels, discrete latent modes, or transformer memory.
- Use four-high quadruple stacking as the primary task because the scripted
  expert remains reliable but the original deterministic model is weak after
  informative contacts. Keep double-stack as a non-regression benchmark.
- Train the stochastic model with flow matching, not likelihood.
- Treat stochastic planning as a later milestone after training and residual
  sampling are verified.
- Residual kernel is **one-step** (predicts `z_{t+1} − Φ(z_{≤t}, u_{≤t})`).
  Multi-horizon consistency is an explicit open problem — see M4
  ("one-step residual kernel versus multi-horizon residual kernels").
- For the first M1 run, default to **detached residual targets** and detached
  condition so the flow objective does not perturb the deterministic
  predictor; the joint-gradient variant is an M4 ablation.
- Time-conditioning uses a sinusoidal embedding scaled for τ ∈ [0, 1]
  (`time_scale=1000` so the standard `max_period=10000` regime applies);
  raw τ without scaling collapses the high-frequency channels.
