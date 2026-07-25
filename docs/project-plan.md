# Flow-Matched Residual Kernels Plan

Last updated: 2026-07-23

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

- **Primary positive result:** the label-free, sequence-consistent FetchPush
  residual kernel improved exact commitment success from 52.34% to 56.15% on
  1,024 fresh contexts: +3.81 points with paired 95% CI [+1.76, +5.86].
- **FetchSlide transfer complete:** the task oracle is very strong
  (+47.66 points, CI [+44.53, +50.00]), but the one-step persistent residual
  fails from H5 onward even with 500 exact training pairs and a centered-head
  ablation. The next justified model is a direct multi-horizon/terminal
  outcome kernel; no further FetchSlide control run is authorized until its
  exact-fork prediction gate passes.
- **Matched vanilla LeWM baseline complete:** a 10-epoch original LeWM with
  residuals disabled reached 7.81% exact FetchSlide commitment success on the
  same 64 held-out scenes (1.56% low friction, 14.06% high friction). The exact
  mean-state oracle reached 0.78% and the exact distribution oracle 43.75%.
  This replaces the earlier state-transition MLP as the architecture-faithful
  deterministic reference; their prediction energies live in different
  representations and must not be compared numerically.

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

```text
Date: 2026-07-20
Commit: 14da4a3 plus uncommitted online-memory fix
Machine/GPU: ICE job 5524399, NVIDIA A100-40GB
Online-memory fix:
  Probed the live eval objects: the env reports one raw frame plus one
  normalized 5-d action per env step, while the encoder consumes 25-d
  frameskip chunks (action_block=5). Because process['action'] is a per-raw-
  dimension StandardScaler, concatenating five normalized raw actions
  reproduces the 25-d chunk the model trained on. ResidualWorldModelPolicy now
  buffers per-env model-rate frame embeddings and raw-action chunks across
  get_action calls and updates memory once per completed model step; the
  dataset-window replay path is retained for pre-chunked callers.
  Two new tests assert the online path reproduces observe_transition exactly
  and that no update lands before a model-step boundary. Suite: 54 pass.
E3 smoke with live memory (n=10, identical selection SHA a97e9382...):
  policy                        success  worst-decile  drops  retries  lat/ep
  deterministic CEM (baseline)     40%        0.215     0.5     3.2     3.6 s
  memoryless flow, 8p mean         50%        0.186     0.1       -     8.3 s
  GRU flow, memory disabled        40%        0.157     0.2       -     8.9 s
  GRU flow, memory live            40%        0.138     0.3     2.4    12.0 s
  Metadata confirms residual_memory.online_update=true with no fallback
  warning. Enabling memory moved worst-decile 0.157 -> 0.138, the best arm.
Pilot launch:
  Jobs 5524913-5524917 run a 2x2 (kernel x objective) plus baseline at 250
  evals each (50 post-contact contexts x 5 physics seeds, eval seed 12041):
  deterministic, flow-mean, flow-CVaR, GRU-mean, GRU-CVaR.
Next action:
  Read the pilot with paired-bootstrap CIs over contexts; report success and
  worst-decile against the deterministic baseline per the standing rule.
```

```text
Date: 2026-07-21
Commit: 14da4a3 plus uncommitted E3 pilot
Machine/GPU: ICE jobs 5524913-5524917, NVIDIA A100-40GB
E3 pilot (250 evals = 50 contexts x 5 physics seeds; shared selection SHA
f358835f; paired over contexts). First two arms complete:
  arm             overall  stage1  stage2  stage3   worst-decile
  deterministic     36.8%    26%     29%     56%        0.130
  flow-mean         34.0%    19%     27%     58%        0.114
  (flow-CVaR, GRU-mean, GRU-CVaR still queued behind A100 availability)
Findings:
  - The n=10 smoke success bump (50%) did NOT survive at n=250: mean-objective
    stochastic planning does not raise success and slightly lowers it, while
    improving worst-decile distance.
  - Stage-conditioned: flow-mean HURT the hardest stage (stage1 26%->19%) and
    barely moved the easy one. Signature of inflated/miscalibrated particle
    spread under a mean objective in exactly the stages where the nominal is
    weak. (n~85/stage, ~5pp SE: directional.)
Diagnosis (why mean-objective is ~a no-op):
  Averaging cost over particles ranks candidates like the deterministic cost
  unless residual variance is ACTION-DEPENDENT (heteroscedastic). The design
  lever is risk-sensitivity (CVaR) + a cost aligned with the physical failure
  (drop/slip), not last-step latent-MSE-to-goal. Low one-step latent MSE
  (0.00097) does NOT imply the task is near-solved (40%): one-step error is
  decoupled from horizon-compounded task success and from the latent-goal cost
  proxy - so the residual has real room to help, in the right objective.
Approved improvement track (user, 2026-07-21):
  1. Physical-probe cost: fit RidgeProbe (latent->cube pos), enable
     collateral_penalty_weight, rerun CVaR arms against a physical failure cost.
  3. E5 spread-calibration: model particle spread vs exact-fork spread at
     H=1/5/10/20. Gate for whether CVaR/multi-horizon are meaningful (if model
     spread >> fork spread, CVaR selects against model hallucination).
  4. Stage-conditioned reporting (started above; finalize with all 5 arms).
  5. Multi-horizon residual-kernel consistency: train on multi-step residuals
     so the particle distribution is valid over the planning horizon.
  Diagnostic to add: verify the flow's residual variance is heteroscedastic
  across candidate actions (the action-dependent-residual crux); if not, no
  risk objective can discriminate candidates.
Next action:
  Let the pilot finish (paired analysis staged: scripts/eval/analyze_e3_pilot.py).
  Build/run E5 spread-calibration before trusting CVaR; then physical-probe cost.
```

```text
Date: 2026-07-21
E5 ground-truth half (true fork spread over horizon; existing evidence
quadruple_stack_strong_1000_seed13041.h5_forks.pt, 64 ctx x 32 real x H=5,
phys_dim=80). Mean within-context per-dim SD of physical state:
  H=1 0.0048 | H=2 0.0084 | H=3 0.0073 | H=4 0.0063 | H=5 0.0051
  Final-horizon spread BY CONTACT STAGE:
    stage 1: 0.00002   stage 2: 0.00005   stage 3: 0.00003   stage 4: 0.02024
KEY FINDING: the true aleatoric spread is concentrated almost entirely at the
FINAL placement stage (stage 4) - 400-1000x larger than stages 1-3, which are
nearly DETERMINISTIC over H=5. Implications:
  1. The E3 control pilot used contact_stages [1,2,3] (stage 4 excluded as
     "mostly retries"). So control was evaluated where there is essentially no
     aleatoric uncertainty to exploit over the horizon - stochastic planning
     cannot help success there by construction.
  2. This explains flow-mean HURTING stage 1 (26%->19%): the kernel injects
     latent residual spread where the real future is deterministic, corrupting
     the plan. That is over-dispersion, the exact miscalibration E5 targets.
  3. The aleatoric action is at stage 4 (final slip-dominated placement), which
     the pilot excluded. To demonstrate control benefit, plan/evaluate AT
     stage 4 and/or over longer horizons.
Caveats: fork horizon is only H=5; longer horizons may show more stage-1-3
divergence. trace(cov) is velocity-dominated and noisy; per-dim SD is the
stable summary.
Reprioritization:
  - Add a stage-4 (and full-episode) control comparison; the current [1,2,3]
    pilot understates where stochasticity matters.
  - Still build the model-spread half of E5 to confirm the kernel is
    over-dispersed at stages 1-3 (probe or latent space).
  - Regenerate forks at H=10/20 for the full spread-growth curve.
```

```text
Date: 2026-07-21
H=20 fork regeneration (job 5525752, 128 ctx x 32 real x H=20, stage-balanced).
Raw mean-per-dim SD was misleading: physical state SETTLES to rest, so variance
over all 80 dims rises during manipulation (~H3-9) then re-converges by H=20 -
diluted, not task-relevant. Corrected to peak/top-5 divergent-dim SD per stage:
  stage  peak-dim SD  @H   behavior over horizon
    1       0.019     1    collapses to 0 by H=3
    2       0.181     1    settles to ~0.001 by H=5
    3       0.050     2    settles to ~0.001 by H=7
    4       0.445     6    SUSTAINED 0.3-0.4 through H=9; 0.04 residual to H=19
CONCLUSION (horizon-robust, not an H=5 artifact): task-relevant aleatoric
uncertainty is overwhelmingly at STAGE 4 (final placement). Stages 1-3 are
effectively deterministic beyond a few steps - futures briefly diverge at the
grasp then re-converge. The persistent stage-4 residual (0.04 at H=19) is the
cube-on-top vs cube-on-table outcome gap. This confirms the E3 [1,2,3] pilot
tested a near-deterministic regime, so its success null is expected, not a
method failure. The strong prediction-metric wins (energy skill, mode-sep 10 SD)
come from modeling the residual DISTRIBUTION; control benefit requires planning
where outcomes are genuinely uncertain (stage 4), which is data-sparse
(11/100 val episodes reach it).
Decision point (options, for user):
  A. Honest positioning: prediction calibration + evaluation protocol are the
     contributions; control = tail-risk reduction + no CVaR regression + a
     stage-4 case study. Do not claim success-rate gains.
  B. Pivot to a task with pervasive aleatoric uncertainty (slip at any stage)
     where stochastic planning can move success.
  C. Targeted stage-4 data collection (start states near final placement) to
     run a properly-powered stage-4 control test.
```

```text
Date: 2026-07-21
E3 FINAL VERDICT (quadruple stacking, task-adapted nominal).
Stage-1-3 pilot complete (5 arms, 250 evals, paired cluster bootstrap over 50
shared contexts, baseline deterministic 36.8% / worst-decile 0.130):
  arm         success  Δ  95% CI (pp)     Δ worst-decile
  flow-mean    34.0%  -2.8 [-6.8,+0.0]      -0.016
  flow-CVaR    35.6%  -1.2 [-3.6,+0.8]      -0.014
  GRU-mean     36.4%  -0.4 [-2.4,+1.6]      -0.008
  GRU-CVaR     33.6%  -3.2 [-7.2,+0.0]      +0.002
  Reading: NO arm improves success; flow-mean and GRU-CVaR are borderline
  HARMFUL (CI upper bound touches 0); flow-CVaR and GRU-mean tie baseline.
  Flow arms give a small tail-distance benefit; GRU tightens the distribution
  (safer mean, less hedging). All consistent with a near-deterministic regime.
Stage-4 attempt (job 5525761): within_stage selection gives success 100.0%,
  worst-decile 0.0001 - a SELECTION ARTIFACT. within-stage-4 contexts are
  near-terminal (4th cube already placed; goal_offset=25 ~ current state), so
  "finish from here" is trivial. The informative selection is stage-4 ONSET,
  but that is ~1 context/episode over ~39 success-biased episodes - too
  sparse/biased for a clean test. Cancelled the 3 redundant arms.
CONCLUSION: quadruple stacking cannot cleanly showcase stochastic CONTROL
benefit - its task-relevant uncertainty is concentrated at a final stage that
is either near-terminal (within-stage) or data-starved (onset). Prediction/
calibration remains the strong result on this task. To demonstrate control,
pivot to a task with PERVASIVE, INFERABLE aleatoric uncertainty (Option A:
hidden-friction non-prehensile push; keep stacking for calibration).
Paper positioning: stacking = calibration + evaluation protocol; push = control.
```

```text
Date: 2026-07-21
Hidden-friction push task - substrate viability (scoping + local test).
Scoping (agent): PushT is pymunk (2D), shares NONE of the MuJoCo hidden-physics
/ fork machinery - wrong substrate. OGBench swm/OGBCube-v0 env_type=single is
MuJoCo, reuses the wrapper+forks unchanged, success = cube within 4cm (push-
satisfiable) - but its oracle grasps; a push policy had to be written.
Local viability test of a scripted 3-phase planar pusher (push_expert.py):
  The UR5e cube robot CANNOT push reliably. In a 4-direction test the cube
  moved only +x (0.039 m, partial) and 0.000 m in -x/+y/-y. The arm is a top-
  down pick-place manipulator; pushing forward at table height near reach
  extension makes the effector rise instead of advancing. Fork spread from
  these non-pushing trajectories was ~0 (uninformative, as expected).
CONCLUSION: no CHEAP substrate gives pervasive push uncertainty. Pervasive,
continuous-contact uncertainty requires a purpose-built pusher:
  - swm/FetchPush-v3 (gymnasium-robotics): genuinely non-prehensile, MuJoCo,
    object mass already randomized - BUT the hidden-physics wrapper, forks,
    fork-evidence, and collection all assume cube-env internals (_model/_data,
    _cube_geom_ids_list, gripper pads) and would need re-plumbing for Fetch
    (.model/.data, object0 body, no pads) + a scripted push expert. Multi-day.
  - swm/OGBMaze-v0 ball-push variant: continuous contact but a locomotion
    agent; JEPA + wrapper both need adaptation. Multi-day.
DECISION POINT (for user):
  A. Commit to FetchPush: re-plumb hidden-physics+forks for Fetch, write a push
     expert. Real control story with pervasive uncertainty. Multi-day.
  B. Reposition the paper on the STRONG results: prediction calibration +
     exact-fork evaluation protocol; control = risk-aware tail benefit + a
     mechanistic explanation (fork-spread analysis) of when/where stochastic
     planning helps. No new task. Publishable as-is.
WIP files (exploratory, not wired into any pipeline): push_expert.py,
scripts/data/probe_push_uncertainty.py, scripts/data/debug_push.py.
```

```text
Date: 2026-07-22
FetchPush hidden-friction pipeline implementation and launch.
Task:
  swm/FetchPush-v3 with one episode-constant latent factor only: object/table
  friction multiplier in {0.2, 3.0}. Object mass, gripper contacts, and slip
  events are fixed. The scripted controller reads only the ordinary flattened
  observation.
Local calibration (100 scenes/mode; 64 attempted fork contexts):
  low success 90%, high success 36%, balanced success 63%.
  Identical-action median mode separation: H1 0.0053 m, H5 0.1228 m,
  H10 0.3135 m, H20 0.6282 m. Passive H20 separation 0.0022 m; action-
  dependence ratio 284.9x. 60 of 64 attempted contexts reached push onset.
Implemented:
  locked gymnasium-robotics 1.4.2; Fetch-only friction wrapper and exact state
  replay; observation-only expert; paired low/high collection; action audit;
  pair integrity validator; version-3 group-aware immutable split; Fetch data
  config and matched nominal/Gaussian/flow/GRU-flow matrices; held-out residual
  scoring; producer plus consumer for exact pre-contact prior-mixture forks.
Verification:
  Four-pair serial HDF5 smoke passed exact paired start, mode, action, success,
  and split invariants. Full local suite: 60 passed.
ICE:
  First vectorized collection 5525798 completed the episode budget but the
  validator rejected two incomplete final pairs (501 IDs for 1,000 episodes).
  The 4.2 GB file was preserved as *.incomplete_pairs.h5 and never reached a
  split or model. Serial collection now enforces num_envs=1.
  Serial collection 5525807 passed with 500 complete pairs, 1,000 episodes,
  29,559 transitions, 62.9% balanced success (91.0% low / 34.8% high), and
  dataset SHA c1e61c1...7be816. The first nominal 5525808 was assigned a
  stale-driver V100 and exited before training. Compatible A100 chain:
  nominal 5525815 -> centered stats 5525816 -> residual array 5525817 ->
  held-out evals 5525818-20 and exact fork eval 5525821.
Decision rule:
  Call the stochastic component useful only if it improves a proper score over
  both nominal LeWM and the residual predictive mean. Bias correction alone is
  not evidence that sampled spread helps.
```

```text
Date: 2026-07-22
FETCHPUSH ONE-SEED PREDICTION VERDICT: STOCHASTIC RESIDUAL HELPS.
Data and nominal:
  Serial paired collection 5525807: 500 complete low/high scene pairs, 1,000
  episodes, 29,559 transitions, balanced success 62.9% (91.0% low / 34.8%
  high), exact paired initial state, dataset SHA c1e61c1...7be816.
  A100 nominal 5525815: final validation pred_loss 0.005381288.
  Centered stats 5525816: 18,225 train transitions, mean L2 0.04523, average
  diagonal scale 0.05809; dataset/split/checkpoint hashes all pinned.
Residual heads:
  Array 5525817 completed conditional Gaussian, flow, and GRU-flow for 10
  epochs at batch size 64 on A100s, with detached targets/conditions and the
  nominal frozen. Checkpoints are under
  data/ice/fetch_push_prediction_a100/{conditional_gaussian,flow,
  flow_gru_memory}/seed_13041/*epoch_10_object.ckpt. The identical nominal
  validation pred_loss stayed 0.005381288 for every head. Memoryless flow
  final validation FM loss was 2.1118; GRU-flow was 2.10 at held-out
  evaluation. Conditional Gaussian
  validation NLL deteriorated late, but the prespecified proper-score gate was
  used rather than selecting an epoch after seeing validation results.
Held-out one-step proper scores (2,048 targets):
  model                 energy skill vs det   stochastic skill vs own mean
  conditional Gaussian       +35.4%                    +19.5%
  flow                       +37.2%                    +25.2%
  GRU-flow                   +38.5%                    +25.1%
Exact H=10 prior-mixture forks (32 contexts x 32 samples; balanced low/high):
  model                 energy score   skill vs det   skill vs own mean
  deterministic             2.676          --               --
  conditional Gaussian      1.825        +31.8%           +13.4%
  flow                      1.706        +36.3%           +20.9%
  GRU-flow                  1.703        +36.3%           +21.0%
  Flow paired absolute gain vs deterministic: 0.971, 95% CI [0.851, 1.098].
  Flow paired gain vs its predictive mean: 0.450, 95% CI [0.412, 0.487].
  GRU vs memoryless flow: +0.0023, 95% CI [-0.045, 0.047] -- a tie.
Attribution:
  All three heads pass all six held-out/exact-fork gates. Sampled spread, not
  merely residual-mean bias correction, improves the proper score. Memory adds
  no detectable value in this pre-contact prior-mixture regime, so memoryless
  flow is the primary model for replication and control work.
Pipeline repairs:
  Evaluators 5525818/19 failed before metrics because saved configs retained
  oc.env paths without exported inputs. Hardened entry point plus pinned retry
  jobs 5525830-32 pass. Verdict 5525833 exposed nvidia-smi's no-device exit on
  CPU nodes; common metadata now reports gpu=none, and retry 5525834 passes.
Scope:
  This is a one-seed prediction result. It supports a multi-seed replication,
  post-contact fixed-mode adaptation test, and only then shared-scene particle
  MPC. It is not yet evidence of improved task success under control.
Artifacts:
  docs/results/fetch_push_friction_validation_20260722.json
  docs/results/fetch_push_residual_stats_20260722.json
  docs/results/fetch_push_stochastic_verdict_20260722.json
```

```text
Date: 2026-07-22
FETCHPUSH CONFIRMATORY THREE-SEED SUBMISSION.
Design frozen before the new jobs start:
  Reuse immutable paired dataset c1e61c1...7be816 and grouped split
  dd7b4d40...83541b; do not recollect. Train independent nominal seeds 13042
  and 13043 for 10 epochs, batch size 64, then compute separate centered
  train-only statistics and train conditional Gaussian plus memoryless flow.
  Re-evaluate existing seed 13041 rather than retraining it.
  Exact test: 128 shared previously unused contexts starting at seed 203072,
  32 samples/context, H=10. This replaces the exploratory 32-context fork set
  for confirmation. Held-out and exact gates still require skill versus both
  deterministic LeWM and the model's own predictive mean.
  Final inference uses a crossed bootstrap over model seeds and shared exact
  simulator contexts; the strict result requires every per-seed gate plus
  positive 95% lower bounds versus deterministic and residual mean.
ICE graph:
  nominal array 5526484
  seed 13042: stats 5526485, heads 5526486, held-out 5526487-88,
              forks 5526489, verdict 5526490
  seed 13043: stats 5526491, heads 5526492, held-out 5526493-94,
              forks 5526495, verdict 5526496
  seed 13041: fresh confirmatory forks 5526497, verdict 5526498
  crossed multi-seed verdict 5526499
  GPU jobs run on the validated L40S nodes; downstream dependencies are
  afterok. The initial A100 requests were changed in place before startup when
  ICE showed idle L40S capacity, preserving all job IDs and dependencies.
```

```text
Date: 2026-07-22
FETCHPUSH FIXED-MODE POSTERIOR DIAGNOSIS AND STATE-MEMORY FOLLOW-UP.
Reliable equal-horizon, common-random-number posterior results (seed 13041):
  Memoryless flow after 5 observed model steps, H=5: energy 1.2149 versus
  deterministic 1.9660 and predictive mean 1.5815, but spread contraction was
  only 0.0036% and paired mode-contrast improvement had 95% CI
  [-0.000158, 0.000353].
  GRU-flow: energy 1.1922 versus mean 1.5564, but contraction was only 0.0472%
  and mode-contrast improvement was significantly negative, 95% CI
  [-0.000400, -0.000006].
Conclusion:
  Sampled residual uncertainty continues to improve the proper score, but
  neither pixel-conditioned flow adapts its distribution to the fixed friction
  mode. This is a posterior-adaptation failure, not a stochastic-score failure.
Mechanistic probe (128 paired scenes, grouped five-fold CV):
  Ordinary observed object displacement decoded friction at 99.6% after five
  model steps. DINO/projected observation embeddings, residual conditions, and
  GRU memory remained near chance (about 50%). The physical signal exists, but
  the current representation discards it before the residual kernel sees it.
Follow-up:
  Added an optional deployable-state-delta input to residual-memory updates;
  vanilla LeWM and residual-only GRU behavior remain unchanged. No hidden mode
  label or future state enters training or rollout. Local suite: 66 passed.
  ICE chain: state-GRU flow training 5526742 -> matched H=5 prior 5526752 ->
  five-step fixed-mode posterior and grouped mode probe 5526753. The first
  launcher attempt 5526735 failed before GPU work because the new matrix had
  not yet been synchronized to ICE; held dependents 5526736-37 were cancelled.
Result:
  Training completed in 4:01 on an L40S; final validation flow-matching loss
  was 2.0706 and frozen nominal validation MSE remained 0.0053840. At H=5 the
  state-GRU flow achieved energy 1.1893 versus deterministic 1.9660 and its own
  mean 1.5527 (23.41% stochastic skill versus mean). It contracted spread by
  only 0.0387%. Its paired mode-contrast gain versus deterministic was
  -0.000211, 95% CI [-0.000702, 0.000125], so the fixed-mode adaptation verdict
  remains false. The state-aware memory probe reached only 55.1% after the
  first observed transition and 53.5% after five, despite the ordinary state
  delta itself reaching 99.6% and 68.0%, respectively; full physical
  displacement history remained 99.6% decodable.
Interpretation:
  Simply exposing state deltas to the unsupervised residual GRU is insufficient:
  the flow objective does not force memory to preserve the friction statistic,
  and the latent prediction target itself is nearly mode-blind. Do not fan this
  variant out to more seeds. The next architecture experiment, if pursued,
  should add a deployable self-supervised dynamics-identification objective or
  replace the image representation with one that preserves object kinematics.
Artifacts:
  docs/results/fetch_push_state_memory_posterior_20260722.json
  docs/results/fetch_push_state_memory_mode_probe_20260722.json
```

```text
Date: 2026-07-22
FETCHPUSH CONTROL SCREEN AND ORDINARY-STATE RESIDUAL FOLLOW-UP.
Latent controller screen:
  Guided deterministic LeWM succeeded on 1/4 shared episodes. Flow mean and
  mean+0.25std tied at 1/4 on a different mode; CVaR, mean+0.5std, conditional
  Gaussian mean, and Gaussian mean+0.25std were 0/4. No stochastic arm beat the
  deterministic arm, so the latent controller result is a null.
Representation-preserving follow-up:
  Added a two-environment-step ordinary-state dynamics model. It consumes only
  the ordinary 28-D Fetch observation and two commanded actions. A nominal MLP
  and 25-D conditional residual flow are stored in one checkpoint; no hidden
  friction label, qpos/qvel, or future state is used. MPC reads object and goal
  coordinates directly from the ordinary observation.
ICE training and exact-fork gate:
  Job 5526852 trained on immutable dataset c1e61c1...7be816 and grouped split
  dd7b4d40...83541b. It used 21,975 train, 2,796 validation, and 2,850 test
  two-step transitions. Best normalized nominal validation MSE was 0.105516;
  best validation flow-matching loss was 2.042249. Checkpoint SHA:
  b59d18d611eb3ff7c3fba51eb517159e79085be39d48ae4e3762f5ccedb244c2.
  Exact-fork job 5526854 used 64 contexts x 32 samples, H=10. Physical
  object-position energy was 0.352952 deterministic and 0.334603 sampled flow
  (5.20% skill). Sampling improved 11.02% over the flow predictive mean even
  though that mean was worse than deterministic. Paired 95% absolute-gain CIs:
  [0.000723, 0.035824] versus deterministic and [0.039022, 0.043794] versus
  the flow mean. The stochastic attribution gate therefore passes.
Control gate:
  Jobs 5526855-57 use ten paired low/high scenes (20 episodes), identical seed
  range 503072-503091, horizon 10, and the same checkpoint. Arms are nominal
  deterministic, eight-particle expected cost (primary), and eight-particle
  CVaR (secondary). Deterministic reached 10%; expected-cost flow and CVaR each
  reached 15%, adding one low-friction episode. This screen was directional,
  not powered.
Fresh-scene confirmation:
  Jobs 5526871-72 evaluated seeds 603072-603171. Deterministic reached 11%
  (12% low / 10% high); expected-cost flow reached 12% (12% / 12%). After
  joining asynchronous completions by seed, flow had nine unique successes and
  deterministic eight, with three shared successes. Absolute gain was +1 point;
  50-scene cluster-bootstrap 95% CI [-7, +9]. Strict improvement gate: fail.
Validation-only controller calibration:
  Deterministic variance screen 5526875-78 selected candidates 0.01 and 0.10.
  Exact fixed-block 100-episode reruns 5526883-84 selected 0.10 by the frozen
  overall-success metric: 18% versus 17%. Flow job 5526885 at variance 0.10
  reached 20%. It gained seven episodes and lost five; +2 points with 95% CI
  [-3, +7]. Strict improvement gate: fail.
Protocol repairs:
  The trace exposed that EnvPool's persistent `_needs_flush` marker caused
  guided MPC to discard its action buffer every raw step. The policy now
  consumes and clears the marker once. Paired full-task evaluation uses fixed
  even seed blocks in wait mode, preventing the episode budget from censoring
  one member of a final low/high pair. Failed pre-result jobs 5526868-69
  identified an EnvPool shape error in the first wrapper-level fix; 5526880
  identified asynchronous final-pair censoring. Neither produced a result used
  in analysis.
Verdict:
  Ordinary-state flow passes the proper-score attribution gate, but neither the
  fresh-scene nor validation-calibrated control comparison has a positive 95%
  lower bound. Current evidence says stochastic residual sampling helps
  prediction, not task success. Further test-objective sweeps would be
  post-hoc; the next experiment should diagnose exact simulator action ranking.
Artifact:
  docs/results/fetch_push_state_dynamics_forks_20260722.json
  docs/results/fetch_push_state_control_confirm_20260722.json
  docs/results/fetch_push_state_control_tuned_20260722.json
```

```text
Date: 2026-07-22
FETCHPUSH MODE-MIXTURE CONTROL CONFIRMATION (PREDECLARED BEFORE FRESH TEST).
Rationale:
  Full-task Fetch terminates when the object first enters the 5 cm goal
  tolerance, but the state MPC ranked only terminal distance. This penalized
  predicted trajectories that pass through the goal and later overshoot.
  The controller now optionally ranks minimum distance over the horizon; a
  unit test fixes the first-entry behavior.
Model and scope:
  Checkpoint 0995360b...1a1f69 is the two-mode residual diagnostic trained on
  the immutable paired data. Its two residual heads used low/high friction
  labels during training, but runtime planning receives no realized mode and
  always uses a balanced prior. This is an explicit-mode stochastic residual
  upper bound, not a label-free flow result.
Frozen validation decision:
  On seeds 503072-503171, threshold-aligned success-probability MPC with
  minimum-horizon distance reached 26% using the residual-mode mean (one
  particle) and 29% using eight persistent balanced mode particles. CEM
  variance remains the previously selected 0.10; horizon 10, action block 2,
  receding horizon 5, 15 CEM iterations, and 300 candidates are unchanged.
Fresh confirmation:
  Evaluate exactly 2,000 episodes / 1,000 paired scenes at consecutive fresh
  seeds 803072-805071. Compare one mean particle to eight persistent balanced
  mode particles. The sole primary endpoint is episode success; the gate is a
  positive scene-clustered bootstrap 95% lower bound for stochastic minus
  deterministic success. Do not tune on this seed range or add objectives
  after seeing it.
```

```text
Date: 2026-07-22
FETCHPUSH PUSH-ONSET COMMITMENT CONFIRMATION (PREDECLARED BEFORE FRESH TEST).
Task qualification:
  The obstacle-free receding-horizon task showed no robust decision value even
  for an exact two-mode stochastic upper bound. The replacement is a controlled
  push-onset commitment task: the observation-only expert approaches the object,
  then the controller selects one open-loop push pulse before hidden friction
  unfolds. Exact simulator forks evaluate both friction modes for every choice.
Validation screen (not confirmatory data):
  On 64 contexts beginning at seed 903072, exact distribution-aware selection
  reached 72.66% versus 61.72% for exact-mean selection, a +10.94-point gain
  with scene-cluster bootstrap 95% CI [+6.25, +16.41]. The learned persistent
  residual modes reached 54.69% versus 50.78% for the learned residual mean,
  a directional +3.91 points with CI [-2.34, +10.16]. No settings are changed
  after this single validation screen.
Frozen protocol:
  Checkpoint 0995360b...1a1f69; friction modes 0.2 and 3.0; 5 cm first-entry
  success; raw horizon 20; constant goal-direction actions; ten speeds from
  0.1 to 1.0; even durations 2 through 20; 100 candidates. Deterministic
  selection minimizes minimum distance under the learned residual-mode mean.
  Stochastic selection minimizes balanced expected smooth failure probability
  under the two persistent learned modes, with temperature 0.01. Exact actions,
  contexts, and success evaluation are otherwise shared.
Fresh confirmation population:
  Evaluate eight immutable shards of 128 contexts (1,024 contexts / 2,048
  paired mode episodes). Shard start seeds are 1003072, 1013072, 1023072,
  1033072, 1043072, 1053072, 1063072, and 1073072; ranges are deliberately
  separated so accepted context seeds cannot overlap. Use 50,000 bootstrap
  draws with seed 1003072 after rejecting duplicate context seeds and checking
  all shard metadata. ICE jobs 5526994 through 5527001 were submitted in that
  seed order with one L40S GPU, six CPUs, and 32 GB requested per shard.
Primary endpoint and gate:
  The sole primary endpoint is exact simulator episode success for the action
  selected by learned persistent modes versus the action selected by their
  learned mean. The stochastic residual helps task success only if the paired
  scene-cluster bootstrap 95% lower bound is strictly positive. The exact-mode
  oracle comparison must also retain a positive lower bound. Do not tune or add
  objectives using these seeds.
Limitation fixed in advance:
  The two residual heads used privileged low/high labels during training; the
  runtime selector receives only a balanced prior. A pass establishes a
  stochastic-residual mechanism/upper bound, after which the same frozen task
  should be repeated with label-free residual training.
Confirmation result:
  All eight jobs completed with exit code 0 and all eight learned shard deltas
  were positive. Across 1,024 unique contexts, learned-mean selection reached
  51.03% and learned stochastic persistent-mode selection reached 57.47%:
  +6.45 percentage points with scene-cluster bootstrap 95% CI
  [+4.59, +8.30]. There were 237 stochastic-only and 105 deterministic-only
  successes. The exact task oracle reached 77.73% versus 60.79% for its mean:
  +16.94 points with CI [+15.48, +18.41]. Both frozen gates pass.
Artifacts:
  Compact Git record:
    docs/results/fetch_push_commitment_mode_confirmation_20260722.json
  ICE raw summary (SHA 1bd31250...1deda):
    data/ice/fetch_state_mode_mixture/seed_24041/
    commitment_confirm_1024ctx/paired_summary_1024ctx.json
```

```text
Date: 2026-07-22
FETCHPUSH LABEL-FREE FLOW COMMITMENT SCREEN (PREDECLARED BEFORE EVALUATION).
Question:
  Does the already-trained conditional flow's sampled residual distribution
  improve exact simulator success over its own predictive mean on the frozen
  push-onset commitment task, without friction labels at training or runtime?
Frozen model and task:
  Use checkpoint 0995360b...1a1f69, but access only its nominal network and
  label-free residual_flow; never access the supervised mode heads. Keep the
  confirmed candidate grid unchanged: raw horizon 20, ten speeds 0.1..1.0,
  even durations 2..20, 100 candidates, 5 cm first-entry success, and smooth
  failure temperature 0.01.
Sampling and attribution:
  Use the existing 32-sample prediction budget and four Euler flow steps.
  Draw 16 context-seeded Gaussian bases and their 16 antithetic negatives with
  base seed 24041. Use common random numbers across candidates and hold each
  base sample fixed through the ten two-step model transitions, matching the
  episode-persistent hidden friction assumption. The deterministic selector
  minimizes distance under the 32-sample predictive-mean future. The
  stochastic selector minimizes expected smooth failure over those identical
  32 futures. Thus the only comparison is distribution-aware scoring versus
  mean-only scoring; nominal bias correction and Monte Carlo samples are shared.
One validation screen:
  Evaluate 64 contexts / 128 exact low-high episodes beginning at fresh seed
  1083072. A positive learned success-rate delta is the sole directional gate
  for a powered fresh confirmation. Do not tune particles, flow steps,
  temporal coupling, candidate grid, or objective after seeing this screen.
  ICE validation job: 5527008 (L40S, six CPUs, 32 GB requested).
Validation result:
  Job 5527008 completed with exit code 0 in 00:05:49. Flow-mean selection
  reached 61.72% and distribution-aware flow selection reached 66.41%:
  +4.69 percentage points, with ten stochastic-only and four mean-only exact
  successes. The 64-context scene-bootstrap CI [-0.78, +10.16] is
  underpowered, while the predeclared directional gate passes. The exact
  oracle retained +18.75 points with CI [+12.50, +25.00]. Artifact SHA:
  39179df7...c7d092.
Fresh confirmation (predeclared before evaluation):
  Repeat the identical frozen model, candidates, sampling, temporal coupling,
  attribution comparison, and simulator-success endpoint on eight immutable
  128-context shards: 1,024 contexts / 2,048 paired exact episodes. Shard
  start seeds are 1203072, 1213072, 1223072, 1233072, 1243072, 1253072,
  1263072, and 1273072. Reject duplicate accepted context seeds and require
  identical checkpoint/task/flow metadata. Use 50,000 scene-cluster bootstrap
  draws with seed 1203072. The sole primary gate is a strictly positive 95%
  lower bound for distribution-aware label-free flow success minus its shared
  predictive-mean success; the exact oracle lower bound must also stay
  positive. Do not tune on these seeds. ICE jobs 5527018 through 5527025 were
  submitted in shard-seed order with one L40S, six CPUs, and 32 GB requested
  per job.
Fresh confirmation result:
  All eight jobs completed with exit code 0 and produced 1,024 unique contexts.
  The validation direction did not replicate. Flow-predictive-mean selection
  reached 62.45%; distribution-aware flow selection reached 62.21%, a
  -0.24-point difference with scene-cluster bootstrap 95% CI
  [-1.71, +1.22]. There were 108 stochastic-only and 113 mean-only successes;
  the primary gate fails. The exact oracle remained strongly positive at
  +17.58 points with CI [+16.11, +19.04], so the failure is the learned
  temporal kernel rather than the task. This rejects the existing one-step
  flow with hand-imposed persistent Gaussian noise as a control-success model.
  Do not retune this confirmation. The next model change is label-free
  episode-level mode discovery, trained with complete-episode assignments so
  residual outcomes remain coherent over the push horizon.
Artifacts:
  Compact Git record:
    docs/results/fetch_push_commitment_flow_confirmation_20260722.json
  ICE raw summary (SHA 561c6dbe...f74397c):
    data/ice/fetch_state_mode_mixture/seed_24041/
    flow_commitment_confirm_1024ctx/paired_summary_1024ctx.json
```

```text
Date: 2026-07-22
FETCHPUSH LABEL-FREE EPISODE-MODE TRAINING (PREDECLARED BEFORE TRAINING).
Motivation:
  The one-step flow passes proper-score prediction gates but its hand-imposed
  persistent Gaussian coupling is a powered control null. The supervised
  persistent residual modes pass control, indicating that temporal outcome
  identity—not marginal spread—is the missing ingredient.
Frozen base and label boundary:
  Start from checkpoint 0995360b...1a1f69 and freeze its nominal network,
  normalizers, residual scale, and conditional flow. Discard and reinitialize
  both diagnostic residual heads. The new trainer may read only HDF5 ep_len,
  ordinary state, and commanded action arrays plus non-privileged split episode
  IDs; it must never read the friction-mode dataset. Train head seed 25041 as
  the fixed primary and seeds 25042/25043 as non-selective stability replicas.
Objective and budget:
  For each epoch, average each head's normalized residual MSE over every
  complete training episode, then assign exactly half of episodes to each head
  by globally balanced winner-take-all advantage. One assignment applies to
  every transition in its episode. Train only the assigned heads for 120
  full-episode epochs with AdamW, learning rate 3e-4 and weight decay 1e-5.
  Select each seed's epoch solely by the same balanced validation WTA loss.
  Do not inspect privileged labels or control outcomes for checkpoint choice.
Prediction gate before control:
  Evaluate all three immutable checkpoints on the same 64 fresh exact contexts
  beginning at seed 1403072, with 32 balanced persistent head samples and
  horizon 10. The fixed primary seed 25041 must improve physical trajectory
  energy score versus both the frozen predictive mean and nominal deterministic
  rollout; replicas are reported, not used to replace the primary. Only after
  this gate may seed 25041 receive one commitment validation screen on fresh
  contexts. No head/diversity regularizer or post-hoc seed selection is allowed.
  ICE training jobs: 5527040, 5527041, and 5527042 in seed order (L40S, six
  CPUs, 32 GB requested each). Shared-fork prediction jobs: 5527045, 5527046,
  and 5527047 in the same order.
Prediction protocol repair before control:
  Jobs 5527045-47 correctly establish stochastic samples versus their residual
  mean, but exposed that the payload's deterministic arm also used the head
  mean rather than the preregistered pure frozen nominal. Preserve those
  outputs, change only that baseline arm to model.transition(noise=None), and
  rerun the identical 64 contexts. The sampled heads and their mean attribution
  remain unchanged. No commitment result has been generated yet. Repair jobs:
  5527052, 5527053, and 5527054 in training-seed order.
Prediction result:
  All repair jobs completed with exit code 0 and all three seeds pass both
  attribution comparisons at H=10. The fixed primary seed 25041 has physical
  object-trajectory energy 0.362379 for the pure frozen nominal and 0.297225
  for label-free persistent episode modes (17.98% skill), with paired absolute
  gain CI [0.052983, 0.077692]. Stochastic sampling also beats the same heads'
  predictive mean with gain CI [0.025479, 0.029998]. Seeds 25042 and 25043
  pass versus nominal with CIs [0.018603, 0.054215] and
  [0.031996, 0.059859], and versus their means with strictly positive CIs.
One control validation screen (predeclared before evaluation):
  Evaluate only primary seed 25041 on 64 fresh commitment contexts / 128 exact
  episodes beginning at seed 1503072. Reuse the confirmed raw horizon 20,
  100-candidate speed/duration grid, 5 cm threshold, temperature 0.01, and
  balanced persistent heads without modification. A positive stochastic-minus-
  mean success point difference is the sole directional gate for a powered
  fresh confirmation. Do not substitute replica seeds or tune on this screen.
  ICE validation job: 5527060 (L40S, six CPUs, 32 GB requested).
Control validation result:
  Job 5527060 completed with exit code 0. Residual-mean selection reached
  63.28%; stochastic persistent episode modes reached 57.81%, a -5.47-point
  difference with 95% CI [-14.84, +3.91]. There were 11 stochastic-only and
  18 mean-only successes. The exact oracle remained +21.09 points with CI
  [+14.84, +27.34]. The directional gate fails; do not run confirmation or
  substitute a replica. This separates general trajectory proper-score skill
  from task-aligned outcome discovery.
Post-freeze diagnostic:
  Job 5527072 may read friction labels only to measure what the already-frozen
  episode assignments represent. It compares full-residual assignments with
  ordinary-observation object-motion views and cannot select or modify this
  failed checkpoint.
```

```text
Date: 2026-07-22
FETCHPUSH LABEL-FREE PAIRED-OUTCOME MODES (PREDECLARED BEFORE TRAINING).
Diagnostic basis:
  Frozen whole-episode assignments have permutation-invariant friction accuracy
  51.2% train and 50.0% validation; the control screen is negative. Object
  displacement has a numerical floor near 2.4e-7 and 75th percentile near
  1.1e-3, so 1e-4 is fixed as a label-free moving-object threshold before the
  new training run. The failed checkpoint is not reused or tuned.
Label-free paired objective:
  The collection protocol supplies exact scene pairs but the trainer never
  reads which member is low or high friction. Freeze base checkpoint
  0995360b...1a1f69, reinitialize both heads, and group consecutive episode IDs
  by floor(id/2). Within each pair, compute head assignment costs only on
  moving transitions (>1e-4 m object displacement) and ordinary object-related
  residual dimensions [3:9, 14:20]. Choose the lower-cost of the two possible
  one-head-per-episode permutations. Use that single episode assignment to
  train the complete 25-D residual over all transitions. This preserves a
  global persistent outcome identity without friction labels.
Budget and fixed seeds:
  Train primary head seed 26041 and non-selective replicas 26042/26043 for 120
  full-pair epochs, AdamW 3e-4, weight decay 1e-5. Select epochs only by the
  paired validation loss. Evaluate all three on 64 shared fresh exact contexts
  beginning at seed 1603072, 32 balanced persistent samples, H=10, against the
  pure nominal and each model's mean. Primary 26041 is fixed irrespective of
  replicas. If and only if it passes both prediction attributions, give primary
  26041 one unchanged commitment screen on 64 contexts beginning at 1703072.
  A positive success point difference is required to proceed; no replica
  substitution, threshold sweep, or additional screen is allowed. ICE training
  jobs: 5527076, 5527077, and 5527078 in seed order. Shared-fork prediction
  jobs: 5527079, 5527080, and 5527081. Post-freeze primary assignment
  diagnostic: 5527082 (not a selection gate).
Prediction result:
  All three seeds pass both H=10 attributions. Primary 26041 has exact physical
  trajectory energy 0.348355 nominal versus 0.284673 stochastic (18.28% skill),
  paired gain CI [0.050169, 0.077525], and stochastic-versus-own-mean gain CI
  [0.020617, 0.025319]. Replicas 26042/26043 also have strictly positive CIs
  versus nominal and their means. Post-freeze global label alignment is 63.5%
  train and 62.5% validation; this is descriptive only, since balanced planning
  is invariant to global head permutation. Primary 26041 proceeds to the one
  frozen commitment screen at seed 1703072. ICE job: 5527083 (L40S, six CPUs,
  32 GB requested).
Control validation result:
  Job 5527083 completed with exit code 0. Residual-mean selection reached
  44.53%; stochastic paired-outcome selection reached 54.69%, a +10.16-point
  gain with scene-cluster bootstrap 95% CI [+3.13, +17.19]. There were 19
  stochastic-only and six mean-only exact successes. The exact oracle retained
  +12.50 points with CI [+7.03, +17.97]. The directional gate passes.
Fresh confirmation (predeclared before evaluation):
  Evaluate the identical immutable seed-26041 checkpoint, candidates,
  balanced persistent heads, threshold, temperature, and exact success
  endpoint on eight 128-context shards (1,024 contexts / 2,048 paired mode
  episodes). Shard start seeds are 1803072, 1813072, 1823072, 1833072,
  1843072, 1853072, 1863072, and 1873072. Reject duplicate context seeds and
  require identical metadata. Use 50,000 scene-cluster bootstrap draws with
  seed 1803072. The sole primary gate is a strictly positive 95% lower bound
  for stochastic minus residual-mean success; the exact oracle lower bound
  must also stay positive. Do not tune or substitute checkpoints on these data.
  ICE confirmation jobs: 5527134 through 5527141 in shard-seed order (L40S,
  six CPUs, 32 GB requested each).
Fresh confirmation result:
  All eight jobs completed with exit code 0; all eight shard point deltas were
  positive and all 1,024 accepted context seeds were unique. Residual-mean
  selection reached 52.34%; stochastic paired-outcome selection reached
  56.15%, a +3.81-point gain with scene-cluster bootstrap 95% CI
  [+1.76, +5.86]. There were 239 stochastic-only and 161 mean-only successes.
  Low-friction success improved 80.57% to 85.74%; high-friction success
  improved 24.12% to 26.56%. The exact oracle retained +17.09 points with CI
  [+15.67, +18.55]. Both preregistered gates pass. This establishes the target
  label-free stochastic-residual control benefit while retaining the powered
  null for ordinary one-step flow sampling.
Artifacts:
  Compact Git record:
    docs/results/fetch_push_paired_modes_confirmation_20260722.json
  ICE raw summary (SHA 363ee868...bbfd00):
    data/ice/fetch_state_paired_modes/seed_26041/
    commitment_confirm_1024ctx/paired_summary_1024ctx.json
```

```text
Date: 2026-07-23
FetchSlide hidden-friction transfer: complete oracle -> data -> model ->
prediction -> control pipeline.

Task and oracle gate:
  swm/FetchSlide-v3 with episode-constant puck/table friction in {0.2, 1.0}.
  The selector commits to one strike before friction is revealed and success
  is terminal distance <= 5 cm after 100 raw steps. On 64 exact low/high
  state forks, mean-state selection reached 0/128 successes while both smooth
  and discrete distribution selectors reached 61/128 (47.656%). The paired
  scene-cluster CI for the +47.656-point gain is [+44.531, +50.000].
  Preferred actions disagreed on every scene, both modes were controllable,
  no candidate solved both modes, and median low/high terminal separation was
  0.5528 m. All task-value gates pass. Compact artifact SHA:
  bf81e7c4...45b1f1d7.

Implementation:
  Added FetchSlide hidden/paired-friction wrappers, an ordinary-state approach
  plus pair-matched open-loop strike policy, exact paired collector and
  validator, a group-aware data config, oracle probe, exact state-fork
  prediction evaluator, commitment evaluator, tests, and ICE launchers. The
  counterfactual member exactly replays the first member's complete commanded
  action sequence; the learned model reads only ep_len/state/action. A
  no-motion pair fix retains complete scene pairs while giving immobile rows
  zero assignment evidence. An optional backward-compatible centered-head
  ablation now forces the two residual outcomes to average to zero per state.

100-pair pilot:
  ICE jobs 5528059, 5528060, failed pre-output mode job 5528061, corrected
  mode job 5528074, prediction 5528075, and control 5528076. Dataset:
  100 pairs / 200 episodes / 26,000 transitions, SHA
  71ea0ee7...d495e84. The mode-job failure exposed an odd episode count after
  dropping a no-motion pair member; the pair-preserving fix is tested.
  At H=50, stochastic state-space energy was 0.54746 versus the nominal
  state-transition MLP at 0.40669:
  paired improvement -0.14077, CI [-0.21852, -0.05865]. It did beat its own
  residual mean by +0.10571, CI [+0.09395, +0.11856], so spread was useful but
  shared/rollout bias erased the benefit versus the frozen nominal.
  The learned commitment result was 0.00% distribution-aware versus 3.91%
  residual mean (-3.91 points, CI [-7.81, -0.78]). The exact oracle on those
  same scenes improved 0.78% to 43.75% (+42.97 points, CI
  [+38.28, +46.88]). The learned control gate fails.

500-pair data-size check:
  ICE jobs 5528084-5528087 completed. The 9.4 GB immutable dataset has 500
  exact pairs / 1,000 episodes / 130,000 transitions, 4.7% random-strike
  success, and SHA 64edaf12...b74b64cf. Nominal validation loss fell from
  0.13811 to 0.03473, flow from 4.12530 to 0.80890, and label-free mode loss
  from 4.51550 to 1.48446. Despite better fitting, H=50 stochastic energy
  worsened to 1.12726 versus nominal 0.30896: paired improvement -0.81830,
  CI [-0.93142, -0.70534]. It still beat its own mean by +0.19311, CI
  [+0.16634, +0.22221]. More data therefore rules out simple scarcity.

Stochastic-only centered ablation:
  ICE jobs 5528097-5528098 used the same frozen nominal, data, assignment
  seed, validation contexts, and budget, but constrained the two residual
  heads to zero mean. H1 energy improved slightly (0.004672 versus 0.004740;
  +1.45% skill), then autoregressive rollout diverged: H5/H10/H50 skills were
  -202.45%, -219.52%, and -519.94%. At H50, energy was 1.91529 versus
  nominal 0.30895; paired improvement -1.60635, CI
  [-1.74700, -1.46185]. No control job was submitted because the predeclared
  prediction attribution gate failed.

Original deterministic LeWM baseline:
  ICE job 5528311 trained vanilla LeWM for 10 epochs on the same immutable
  500-pair dataset (base commit 14da4a3353897682fc2859484d016dd08089a02a,
  seed 28041, batch 64). Residual flow and residual kernel were disabled and
  are both absent from checkpoint SHA 80d232db...690e05. Final validation
  prediction loss was 0.00382292. The run predates a metadata-only fix and its
  raw `run_metadata.json` incorrectly records the configured default kernel
  type as `flow`; the resolved config says `loss.residual_flow.enabled=false`,
  direct checkpoint inspection finds neither residual module, and both
  evaluators enforce this invariant. The metadata writer now emits `none` for
  disabled runs.
  Exact latent-fork job 5528312 measured deterministic energy 0.85604,
  2.04449, 3.42349, and 13.83248 at H1/H5/H10/H50, respectively, with zero
  interval coverage as expected from a point predictor. These latent-space
  values are not numerically comparable to the ordinary-state MLP values
  above.

  Native goal-image commitment job 5528313 selected among the same 52 strike
  candidates using terminal distance in LeWM's own latent space, then replayed
  the selected strike in both exact friction forks. On the same 64 held-out
  scenes / 128 episodes used by the 100-pair control test, vanilla LeWM reached
  10/128 = 7.8125% success (1/64 = 1.5625% low friction; 9/64 = 14.0625% high
  friction; no scene succeeded in both modes), with scene-cluster bootstrap
  95% CI [3.90625%, 12.5%]. The exact deterministic mean-state oracle reached
  1/128 = 0.78125%, while the exact distribution oracle reached 56/128 =
  43.75%, a +42.96875-point oracle gain with CI [+38.28125, +46.875].
  LeWM selected the exact mean-state candidate in only 1/64 scenes and the
  distribution-oracle candidate in 0/64 scenes.
  The original LeWM therefore beats the deterministic mean-state heuristic
  but remains 35.94 points below the uncertainty-aware oracle. This is the
  requested no-residual architecture-faithful baseline.

Conclusion:
  FetchSlide is a very strong hidden-mode decision task, but the current
  one-step persistent residual head is not helping at its 50-step ballistic
  horizon, and original deterministic LeWM reaches only 7.81% exact control
  success despite a 43.75% uncertainty-aware oracle. The residual failure is
  horizon consistency: one-step signal is positive, then mode-conditioned
  autoregression becomes unstable. The next justified model is a direct
  multi-horizon/terminal outcome kernel (or a stable sequence-level latent
  transition), not another one-step data-scale sweep.

Compact artifacts:
  docs/results/fetch_slide_friction_oracle_probe_20260723.json
  docs/results/fetch_slide_state_fork_metrics_100pairs_20260723.json
  docs/results/fetch_slide_commitment_100pairs_20260723.json
  docs/results/fetch_slide_state_fork_metrics_500pairs_validation_20260723.json
  docs/results/fetch_slide_state_fork_metrics_500pairs_centered_validation_20260723.json
  docs/results/fetch_slide_vanilla_lewm_fork_metrics_20260723.json
  docs/results/fetch_slide_vanilla_lewm_commitment_20260723.json
```

## Key Design Decisions

- Start in latent space, not pixels.
- Keep residual flow optional and off by default.
- Use exact train-only centered residual statistics from the frozen nominal
  checkpoint; keep the EMA path only for small plumbing runs.
- Keep temporal persistence explicit and minimal: the positive control model
  uses two persistent residual outcomes discovered from paired trajectories,
  with no privileged mode labels or transformer hierarchy.
- Use four-high quadruple stacking as the primary task because the scripted
  expert remains reliable but the original deterministic model is weak after
  informative contacts. Keep double-stack as a non-regression benchmark.
- Retain flow matching for one-step marginal residuals; use pair-contrastive
  complete-episode assignments for the sequence-consistent control kernel.
- Require prediction attribution before stochastic planning, then evaluate
  success on fresh paired simulator contexts with scene-cluster intervals.
- Residual kernel is **one-step** (predicts `z_{t+1} − Φ(z_{≤t}, u_{≤t})`).
  Multi-horizon consistency is an explicit open problem — see M4
  ("one-step residual kernel versus multi-horizon residual kernels").
- For the first M1 run, default to **detached residual targets** and detached
  condition so the flow objective does not perturb the deterministic
  predictor; the joint-gradient variant is an M4 ablation.
- Time-conditioning uses a sinusoidal embedding scaled for τ ∈ [0, 1]
  (`time_scale=1000` so the standard `max_period=10000` regime applies);
  raw τ without scaling collapses the high-frequency channels.
