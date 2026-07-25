# Project Status

Last updated: 2026-07-23

## Repository

- Local repo: `/home/fwu91/Storage/Projects/ResDyn/le-wm`
- Branch: `latent-residual-flow`
- Fork remote: `git@github.com:fei-yang-wu/le-wm.git`
- Sky1 clone: `fwu91@sky1:~/flash/Research/WM`
- Environment: `pixi.toml` plus committed `pixi.lock` on Linux.

## Implemented

- Optional latent residual flow in `residual_flow.py`.
- JEPA residual helpers in `jepa.py`:
  - residual condition construction,
  - synchronized centered residual statistics,
  - normalized residual sampling,
  - optional stochastic rollout with particle-specific memory.
- One-layer, 128-dimensional explicit residual GRU in `residual_memory.py` and
  online observation updates in `residual_policy.py`.
- Joint residual flow-matching loss in `train.py`.
- Hydra config block under `loss.residual_flow` in `config/train/lewm.yaml`.
- Slurm helpers for import smoke tests, PushT residual-flow training, vanilla
  PushT training, and residual distribution evaluation.
- ICE Slurm helpers for GPU smoke checks, multi-seed ablation arrays, and
  residual checkpoint evaluation.
- Train-only distributed residual normalization, strict vanilla checkpoint
  state dictionaries, deployment-token loss, and immutable episode splits.
- Conditional Gaussian, memoryless flow, and GRU-memory flow kernels trained
  against one frozen nominal checkpoint and shared centered residual targets.
- Particle MPC with separate candidate/particle axes, common random numbers,
  chunking, mean/CVaR objectives, and optional physical-probe collateral cost.
- Hidden OGBench/RoboCasa physics, ten-substep slip transients, commanded-action
  logging, simulator forks, proper scores, physical probes, and surprise tools.

## Cluster and Data

- Large stochastic-manipulation artifacts are managed in the private Hugging
  Face dataset repo `fei-yang-wu/lewm-stochastic-manipulation`; Git retains
  schemas, hashes, and compact run records only. Uploads require successful
  HDF5 validation and simulator-fork evidence generation.

- Slurm defaults: `partition=wu-lab`, `qos=short`, `gpus-per-node=a40:1`,
  `cpus-per-task=6`.
- Working fallback when `wu-lab` is busy:
  `--partition=overcap --account=overcap`.
- Project data root on Sky1: `~/flash/Research/WM/data`.
- PushT data: `data/pusht_expert_train.h5`.
- Python environment: Pixi-managed Python 3.10 with `stable-worldmodel==0.1.1`,
  `stable-pretraining==0.1.8`, and `datasets==2.21.0`.
- The default environment includes HDF5, PushT, and OGBench 1.2.1. A separate
  Python 3.11 environment imports RoboCasa 1.0.1 and robosuite 1.5.2; its
  macros/assets are not configured yet.
- Verified ICE allocation for this account: account `cse`, QOS `coe-ice`, with
  GPU access through `ice-gpu` (among other partitions). Submission helpers
  still require explicit allocation arguments.
- ICE Pixi root: `/storage/ice1/3/2/fwu91/.pixi`; the 2026-07-19 smoke checkout
  is `/storage/ice1/3/2/fwu91/le-wm-codex-smoke-20260719`.

## Completed Runs

Tiny residual-flow smoke:

- Dataset: PushT.
- Scope: 1 epoch, 2 train batches, 1 validation batch.
- Result: completed on overcap A40 and saved object/weights checkpoints.

One-epoch residual-flow run:

- Slurm job: `3030237`.
- Result: completed in `01:42:04` with exit code `0:0`.
- Checkpoints:
  - `data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_epoch_1_object.ckpt`
  - `data/pusht_rflow_1epoch/lewm_rflow_pusht_1epoch_weights.ckpt`
- Final validation metrics from the training log:
  - `validate/loss: 1.5134869813919067`
  - `validate/pred_loss: 0.06748738884925842`
  - `validate/residual_fm_loss: 1.2505505084991455`
  - `validate/sigreg_loss: 2.1714375019073486`

## Paper Program (2026-07-19)

- Submission framing and E1–E9 experiment matrix: `docs/paper-plan.md`.
- New E9 diagnostic `scripts/eval/evaluate_memory_filter.py` (+
  `submit_memory_filter.sh` on ICE): replays held-out episodes into the GRU,
  reports mode-probe accuracy, state saturation, and memory-vs-reset energy
  skill versus update count. Runs before any control experiment; decides
  whether burn-in (TBPTT) memory training is needed.
- Task-adapted nominal (E2) prepared: `config/ablations/quadruple_stack_nominal.tsv`
  plus a runbook in `docs/paper-plan.md` (train vanilla nominal on the pilot,
  centered stats, kernel array against it) to separate nominal-bias correction
  from stochasticity modeling.

## E9 Memory-Filter Result (2026-07-19)

- ICE job `5522477` ran the diagnostic on the array-`5521894` GRU-flow
  checkpoint: memory-vs-reset energy skill stays +0.10..+0.23 out to 200
  replay updates with stable state norms — no long-horizon collapse, no TBPTT
  fix needed, E3 unblocked from this side.
- The hidden mode is not linearly (or MLP-) decodable from the memory state at
  any depth (exact accuracy at the 0.18 majority baseline). The memory is
  adaptive recent-context conditioning, not a mode filter; the paper figure
  and framing change accordingly. Result JSON:
  `docs/results/quadruple_stack_memory_filter_20260719.json`.
- **E2 chain complete (2026-07-20)**: kernel array `5522721` trained all
  three heads against the task-adapted nominal; residual evals `5522758-60`
  show all kernels beating the unit Gaussian on near-pure aleatoric residuals
  (GRU-flow best on covariance/skill; conditional Gaussian competitive on
  energy; coverage_90 ~0.81 needs watching). E9 rerun `5522757` replicates
  the mode-probe null under the task-adapted nominal — the memory is
  recent-context adaptation under both nominals, with skill decaying but
  staying positive out to 200 updates. Results:
  `docs/results/task_adapted_*_20260720.json`; benefit-vs-deterministic
  writeup: `docs/results/residual-dynamics-benefit-20260720.md`.
- Task-adapted nominal (E2) is trained and frozen: job `5522476` reached
  validation pred_loss 0.00227 (25x below the official nominal on this data).
  Centered stats (job `5522716`): residual mean norm 0.0484 vs 1.1098 and
  scale 0.028 vs 0.494 against the official nominal — residuals are now
  nearly pure aleatoric noise. Gated kernel array `5522721` (conditional
  Gaussian / flow / GRU-flow) is training against it under
  `ice/quadruple_stack_prediction_task_adapted/`.

## Current Focus

FetchPush with one episode-constant hidden friction mode is now the primary
task for testing whether the stochastic residual itself helps.  The low/high
surface-friction multipliers are 0.2x and 3.0x; mass, gripper friction, and
slips remain fixed.  A local 100-seed screen reached 90% low-mode success and
36% high-mode success (63% balanced), with identical post-contact actions
producing 0.123 m median separation at five raw steps and 0.628 m at twenty;
the passive twenty-step separation was only 0.0022 m.

The complete paired pipeline is implemented: observation-only Fetch expert,
exact MuJoCo state forks, commanded-action audit, low/high scene pairing,
group-aware train/validation/test splits, task-adapted nominal training,
centered residual statistics, Gaussian/flow/GRU-flow training, held-out proper
scores, and exact prior-mixture simulator-fork scoring.  The local suite passes
84 tests under the locked Pixi environment; `pixi run --locked check` also
passes.

ICE job `5525798` collected all 1,000 requested episodes but correctly failed
the pairing validator because asynchronous vector termination stopped on two
half-pairs.  Its 4.2 GB HDF5 was preserved under an
`incomplete_pairs` quarantine name and was never split or used for training.
Serial paired collection `5525807` passed: 500 complete pairs, 1,000 episodes,
29,559 transitions, 62.9% balanced success (91.0% low and 34.8% high), exact
paired initial state, and dataset SHA `c1e61c1...7be816`.  The first nominal
attempt `5525808` landed on a stale-driver V100 and exited before training.
The compatible A100 chain is nominal `5525815`, residual statistics `5525816`,
three-head array `5525817`, and exact-fork evaluation `5525821`; all completed.
The first held-out evaluators `5525818-20` exposed missing `oc.env` path exports
before writing any metrics.  Replacements `5525830-32` completed after the ICE
entry point was hardened, and CPU verdict retry `5525834` completed after GPU
metadata probing was made optional.

The one-seed strict prediction gate passes for conditional Gaussian, flow, and
GRU-flow.  Held-out residual energy skill is respectively 35.4%, 37.2%, and
38.5%; stochastic sampling versus each model's predictive mean adds 19.5%,
25.2%, and 25.1%.  On 32 exact balanced low/high forks at H=10, deterministic
energy score is 2.676 versus 1.825, 1.706, and 1.703.  Flow's paired absolute
improvement is 0.971 versus deterministic (95% CI [0.851, 1.098]) and 0.450
versus its mean (95% CI [0.412, 0.487]).  GRU and memoryless flow are tied:
paired GRU improvement over flow is 0.0023 with CI [-0.045, 0.047].  Therefore
sampled residual spread is helping prediction beyond deterministic bias
correction, while residual memory is not justified on the pre-contact prior
mixture.  This remains one seed and is not a control-performance claim.

Three-seed confirmation was submitted on ICE on 2026-07-22.  Nominal array
`5526484` trains fresh seeds 13042 and 13043; their statistics, Gaussian/flow
heads, held-out evaluations, exact forks, and per-seed verdicts are
`5526485-96`.  Existing seed 13041 is re-evaluated on the confirmatory forks in
`5526497-98`.  Final crossed seed/context bootstrap job `5526499` waits for all
three verdicts.  The exact test is predeclared as 128 shared fresh contexts
from seed 203072, 32 samples per context, and H=10.  No data is recollected.
All pending GPU jobs were moved in place from constrained A100 requests to
validated idle L40S nodes without changing job IDs or dependencies.

The first statistically confirmed control benefit is now established on an
uncertainty-sensitive FetchPush push-onset commitment task. The earlier
obstacle-free receding full-task controller remained null even with an exact
two-mode upper bound, because both friction outcomes usually prefer the same
action. In the commitment task, 1,024 fresh paired contexts force one push
pulse to be selected before hidden friction unfolds. Learned persistent-mode
selection reached 57.47% exact simulator success versus 51.03% for the same
model's residual mean: +6.45 percentage points with paired scene-cluster 95%
CI [+4.59, +8.30]. The exact distribution oracle improved 60.79% to 77.73%
(+16.94 points, CI [+15.48, +18.41]). Jobs `5526994-5527001` all completed;
every shard was positive. This is a supervised explicit-mode residual upper
bound—the two diagnostic heads used friction labels during training—so the
next gate repeats the frozen commitment task with the label-free conditional
flow. Compact result:
`docs/results/fetch_push_commitment_mode_confirmation_20260722.json`.

That label-free flow gate is now a powered null. One validation screen on 64
contexts was directional (+4.69 points), but the preregistered 1,024-context
fresh confirmation reached 62.21% distribution-aware success versus 62.45%
for the shared flow predictive mean: -0.24 points with paired 95% CI
[-1.71, +1.22]. Jobs `5527018-5527025` all completed cleanly; the exact oracle
on the same contexts remained +17.58 points with CI [+16.11, +19.04]. Thus
the task still rewards correct stochastic prediction, but persistent reuse of
one-step flow base noise does not recover coherent episode-level friction.
Next: train two residual outcomes with label-free, balanced complete-episode
winner-take-all assignments, then repeat prediction and commitment gates.
Compact result: `docs/results/fetch_push_commitment_flow_confirmation_20260722.json`.

The target label-free control result now passes after making outcome identity
sequence-consistent. A pair-contrastive trainer sees only ordinary state,
commanded action, episode length, and which two episodes share an initial
scene; it never reads which member has low or high friction. It assigns the two
members to opposite residual heads using moving-object dimensions, then trains
the complete residual under one persistent assignment per episode. Three head
seeds all pass exact-fork prediction versus both the frozen nominal and their
own predictive means. Fixed primary seed 26041 was selected before evaluation.

On 1,024 fresh commitment contexts / 2,048 exact low-high episodes, its
residual mean reached 52.34% success and stochastic paired-outcome planning
reached 56.15%: +3.81 percentage points with paired scene-cluster 95% CI
[+1.76, +5.86]. All eight shards were positive; there were 239 stochastic-only
and 161 mean-only successes. Both low friction (80.57% to 85.74%) and high
friction (24.12% to 26.56%) improved. Jobs `5527134-5527141` all completed and
the exact oracle retained +17.09 points with CI [+15.67, +18.55]. This proves
that a label-free sequence-consistent stochastic residual can improve actual
simulator success over its deterministic residual mean. It does not rescue the
ordinary one-step flow null; persistent outcome structure is the key finding.
Compact result:
`docs/results/fetch_push_paired_modes_confirmation_20260722.json`.

FetchSlide is the first transfer test of the same generic hidden-mode scheme.
Its exact task gate is exceptionally strong: on 64 fresh scenes, mean-state
selection reached 0% while exact distribution-aware selection reached 47.66%
(+47.66 points, 95% CI [+44.53, +50.00]). The complete paired collection,
label-free training, exact-fork prediction, and exact control pipeline is now
implemented and exercised on ICE.

The learned transfer is a clean negative. With 100 training pairs, the H=50
mode-mixture state-space energy score was 0.547 versus 0.407 for the frozen
state-transition MLP, and the learned control selector reached 0% versus 3.91%
for its residual mean.
A 500-pair rerun improved nominal validation loss fourfold but worsened H=50
mode-mixture energy to 1.127 versus 0.309 nominal. A zero-mean residual-head
ablation retained +1.45% H1 skill but diverged by H5 and reached 1.915 energy
at H50. Both long-horizon prediction gates fail, so no 500-pair control job
was submitted. The diagnosis is one-step-to-ballistic-horizon inconsistency,
not lack of task value or merely too little data. Next use a direct
multi-horizon/terminal outcome kernel. Compact artifacts are
`docs/results/fetch_slide_*_20260723.json`.

A matched original-LeWM baseline now removes the architecture ambiguity in
that comparison. ICE job `5528311` trained vanilla LeWM for 10 epochs on the
same 500-pair data, with both residual modules absent from checkpoint
`80d232db...690e05`. On 64 exact latent forks, point-prediction energy grew
from 0.856 at H1 to 13.832 at H50 and interval coverage was zero. Native
goal-image action selection on the same 64 held-out control scenes reached
7.81% success (1.56% low friction, 14.06% high friction), versus 0.78% for
the exact deterministic mean-state oracle and 43.75% for the exact
distribution oracle. The latent LeWM and ordinary-state MLP prediction scores
are in different representations and are not compared numerically. Jobs
`5528312-5528313` and compact artifacts
`docs/results/fetch_slide_vanilla_lewm_*_20260723.json` record the result.

M2 evaluation has a first smoke result for the 1-epoch checkpoint. The learned
flow improves covariance matching and central interval coverage versus the
diagonal Gaussian baseline, but its quantile ECE is slightly worse.

The simple persistent residual infrastructure is implemented and its local and
ICE correctness gates pass. Double-stack remains a non-regression task because
its strong-profile expert succeeded on 100% of the 1,000-episode pilot.
Four-high quadruple stacking is now the primary task: a 100-episode screen
passed with 95% expert success, 26% official deterministic LeWM success on
post-contact windows, and H=5 exact-fork mode separation of 5.180 pooled SD.
Its 1,000-episode pilot, exact-fork gate, Hub publication, centered-statistics
preflight, and conditional-Gaussian smoke now pass. The matched three-head
prediction array is running on ICE L40S GPUs.

First residual evaluation:

- Slurm job: `3080285`.
- Result: completed in `00:02:05` with exit code `0:0`.
- JSON output: `data/eval/pusht_rflow_1epoch_residual_eval.json`.
- Held-out targets: `6144`; latent dim: `192`; flow NFE: `8`.
- Deterministic latent MSE: `0.06709294766187668`.
- Flow covariance relative Frobenius: `0.4756399989128113`.
- Gaussian covariance relative Frobenius: `0.8401789665222168`.
- Flow 90% interval coverage: `0.8628132939338684`.
- Gaussian 90% interval coverage: `0.8166148066520691`.
- Flow quantile ECE: `0.02631089650094509`.
- Gaussian quantile ECE: `0.024060126394033432`.

Complex stochastic local/ICE plumbing smoke (2026-07-19):

- Local RTX PRO 6000 Blackwell: collected ten temporary double-stack episodes,
  trained/validated a two-step nominal LeWM smoke, computed a frozen residual
  scale from 1,057 training clips, and trained conditional-Gaussian and flow
  heads against the same frozen nominal checkpoint.
- Frozen nominal validation MSE remained exactly
  `0.0559137798845768` before and after both head-training epochs.
- Final local correctness suite: 22 tests passed; the synced ICE suite passed
  the then-current 20-test set before the evaluator-selection tests were added.
- ICE job `5520712`: completed in 31 seconds on one NVIDIA A40; Pixi locked
  imports, CUDA tensor execution, and residual-flow tensor execution passed.
- ICE job `5520709`: bootstrap-only failure caused by resolving `common.sh`
  relative to Slurm's spool copy. All ICE batch entry points now resolve it
  from `REPO_DIR`.
- No dataset collection, training, evaluation, pilot, or ablation array was
  submitted to ICE.

Persistent residual-memory local gate (2026-07-19):

- Exact train-split statistics now include residual mean and diagonal scale,
  plus dataset, split, and nominal-checkpoint hashes. The ten-episode nominal
  smoke had residual-mean norm `3.2903` and average residual scale `0.00487`,
  confirming that the old uncentered normalization was materially wrong.
- A two-epoch, one-batch-per-epoch frozen-nominal double-stack GRU-flow smoke
  completed locally. Nominal validation MSE remained exactly
  `0.0559137798845768`; checkpoint reload, stochastic rollout, and particle
  CVaR MPC passed.
- The synthetic AR(1) gate reduced trajectory energy from `0.7283` to `0.5950`
  and lag-one correlation error from `0.8006` to `0.3320` versus a memoryless
  conditional Gaussian. The complete local suite passes 35 tests.
- ICE smoke job `5521300` completed in `00:04:26` on one A100 with exit `0:0`.
  CUDA imports, residual-flow execution, the AR-memory gate, and all 34 tests
  passed.

FetchPush posterior adaptation (2026-07-22):

- Three-seed prediction replication passes: stochastic flow improves H=10
  exact-fork energy by 34.76% versus deterministic LeWM (95% CI
  [32.03%, 37.30%]) and 20.22% versus its own predictive mean (95% CI
  [19.18%, 21.29%]).
- Fixed-friction posterior tests do not show adaptation. After five observed
  model steps, memoryless-flow spread contracts only 0.0036%; GRU-flow only
  0.0472%. Neither improves paired low/high mode contrast over deterministic
  LeWM, despite retaining roughly 23% stochastic skill versus its own mean.
- A grouped probe decodes friction from ordinary physical displacement at
  99.6%, while observation embeddings, residual conditions, and GRU memory are
  near chance. Representation loss, not lack of task signal, is the bottleneck.
- An optional state-delta-conditioned residual GRU is implemented without mode
  labels or future-state leakage. Local tests pass (66). ICE training `5526742`
  and evaluation jobs `5526752`-`5526753` completed successfully.
- The state-GRU still fails fixed-mode adaptation: H=5 energy is 1.1893 versus
  deterministic 1.9660 and its own mean 1.5527, but spread contracts only
  0.0387% and paired mode-contrast improvement has 95% CI
  [-0.000702, 0.000125]. Its memory decodes friction at only 53.5% after five
  observations. This variant should not be replicated across seeds.

FetchPush control and ordinary-state follow-up (2026-07-22):

- A shared-scene latent LeWM controller screen found no success-rate gain. The
  guided deterministic arm succeeded on 1/4 episodes; flow-mean and
  flow-mean-plus-0.25-std also reached 1/4 but on a different friction mode,
  while flow-CVaR, stronger variance penalties, and Gaussian particles were
  0/4. This is a control null, not evidence for stochastic control.
- A deployable ordinary-state follow-up now models two raw Fetch steps at a
  time from the ordinary 28-D observation and two commanded actions. A nominal
  MLP and conditional residual flow live in the same checkpoint; deterministic
  versus stochastic MPC therefore changes only whether residual particles are
  sampled. No friction labels or simulator state enter the model.
- ICE training `5526852` used the immutable paired dataset and grouped split
  (21,975/2,796/2,850 train/validation/test transitions). Best normalized
  nominal validation MSE was 0.10552 and best flow-matching validation loss was
  2.04225. The checkpoint SHA is `b59d18d6...b244c2`.
- Exact state-space forks `5526854` pass the sampling attribution gate at H=10:
  flow energy is 0.33460 versus deterministic 0.35295 (5.20% skill) and its own
  mean is worse than deterministic, yet sampled flow improves 11.02% over that
  mean. Paired 95% absolute-improvement CIs are [0.00072, 0.03582] versus
  deterministic and [0.03902, 0.04379] versus the flow mean.
- The initial 20-episode control screen was 10% deterministic versus 15% for
  both flow expected cost and CVaR. A fresh 100-episode/50-scene comparison was
  11% deterministic versus 12% flow: +1 percentage point with scene-clustered
  95% CI [-7, +9], nine flow-only successes and eight deterministic-only.
- Validation-only CEM calibration selected variance 0.10 by deterministic
  success (18%, versus 17% at variance 0.01). At that frozen setting, flow
  reached 20% on the same 100 validation episodes: +2 points, 95% CI [-3, +7],
  seven flow-only and five deterministic-only successes. The strict control
  improvement gate fails. Stochastic residuals improve proper prediction
  scores but have not shown a success-rate benefit.
- Paired full-task evaluation now runs fixed even seed blocks in wait mode and
  clears reset markers after one policy call. This prevents asynchronous final
  pairs from being censored and restores the configured receding-horizon action
  buffer instead of replanning every raw step.
- ICE collection validation job `5521321` completed with exit `0:0` on the
  two-episode strong-profile smoke: 302 transitions, exact commanded-action
  audit, dataset SHA `26057c7d...6f5b9ff`.
- ICE simulator-fork job `5521327` completed with exit `0:0`: two contexts,
  eight physics realizations, five steps, and evidence SHA
  `23e73cba...ae29a5b` linked to the collection hash.
- The complete smoke bundle is stored under `smoke/ice/` in the private HF
  dataset repo at commit `41b2a420d4594ad46989019b27133671c5fb39c0`.
- ICE job `5521332` completed the first 1,000-episode calibration pilot in
  `00:26:41`: 151,286 transitions, all eight hidden modes, 210 slip events,
  100% expert success, and dataset SHA `836de31d...ac7161e7`.
- ICE job `5521372` produced 64 contexts x 32 realizations x five-step futures
  in `00:02:32`; within-context mode separation was 2.506 pooled SD and the
  evidence SHA was `ea8bc1ed...07915dcf`.
- The pilot gate is `fail_too_easy`. Milder profiles were not submitted and no
  model training was launched. Compact manifests/evidence are in the private
  HF repo at commit `39902a20a9b2ffa159de5bebee2f27cbc35c3f78`; the 22 GB HDF5
  remains validated on ICE pending Hub authentication on that host.

Harder-task screen (2026-07-19):

- Quadruple task 5 performs four-high stacking under the same hidden strong
  mass/friction/slip profile. ICE job `5521460` collected 100 episodes / 36,022
  transitions: expert success 95%, all eight hidden modes, 305 grasp onsets,
  and 66 transient slip events. The immutable 5.2 GB HDF5 has SHA
  `788559b4...a278eb`.
- ICE job `5521483` generated 32 contact-stage-balanced contexts x 16 physics
  realizations x five steps. Within-context mode separation is 5.180 pooled SD;
  the fork evidence SHA is `d42f6919...ea5575`.
- The official `quentinll/lewm-cube` checkpoint is pinned at revision
  `b0747c5002e86d2ce8f3cd8178004b97524c587d` and strictly converted into the
  current model. Job `5521492` reached 38% success over 100 ordinary reachable
  windows. Job `5521515` reached only 26% over 100 stage-balanced post-contact
  windows (22.9%, 22.9%, and 33.3% after contacts 1-3).
- The task screen passes: the expert/model post-contact gap is 69 percentage
  points and the stochastic futures are strongly separated. Contact stage 4 is
  mostly retries/failures, so primary analysis uses stages 1-3.
- Compact fork, split, validation, and deterministic-evaluation artifacts are
  in private HF dataset `fei-yang-wu/lewm-stochastic-manipulation` at commit
  `d8ce3c25d5e7b5b1b5f94eff1f6b99d3b49fc429`. The 5.2 GB HDF5 remains on ICE
  until that host is authenticated. No 1,000-episode collection or training
  job was submitted.

Quadruple-stack Gate 2 pilot and prediction launch (2026-07-19):

- ICE job `5521684` collected 1,000 strong-profile episodes / 383,766
  transitions. Expert success was 91.4%, with all eight hidden modes, 3,086
  contact onsets, and 781 slips. Dataset SHA:
  `6adb185168216c544bbaad65d1aac04b74f7fe781fc67d624e4710890582b0f6`.
- Job `5521685` generated 64 contact-stage-balanced contexts x 32 independent
  realizations x H=5. Mode separation was 10.094 pooled SD and fork SHA was
  `1b74bfdeac7396be55befc8fabb353df045ad9c45cbdd3fb6a9f1dc67fbaada1`.
- Strict gate job `5521744` passed: expert 0.914, deterministic post-contact
  LeWM 0.26, expert-model gap 0.654, and mode separation 10.094 SD.
- The first Hub job failed because the compute cache hid the login credential.
  Retry `5521810` published all seven immutable artifacts at private Hub commit
  `9e73de75ca3e5ebce9cf90b4bd2ebf300d0aa1cd`; Hub LFS hashes and sizes match.
- L40S job `5521880` computed exact centered train-only statistics from 291,925
  clips. Mean SHA is `cb6571a3...7714d1e`, scale SHA is
  `324be2d0...ab47607`, and the dataset/split/checkpoint hashes match the run.
- Conditional-Gaussian smoke `5521881` completed two train batches and one
  validation batch. Gated launcher `5521882` then submitted array `5521894`.
  Its three running tasks are conditional Gaussian, memoryless flow, and
  GRU-memory flow, all with frozen nominal LeWM, model seed 13041, 10 epochs,
  batch size 64, and identical dataset/split/statistics hashes.

## Open Risks

- The 1-epoch checkpoint was trained before the new `time_scale=1000` time
  embedding default. The code keeps old object checkpoints loadable by falling
  back to `time_scale=1.0` when the attribute is missing.
- A 1-epoch run is only a smoke-quality model. It can validate the plumbing and
  basic metrics, but not a research claim.
- Vanilla PushT baseline comparison is still needed before interpreting control
  or prediction improvements.
- Array `5521894` is the one-seed prediction gate, not a paper result. It must
  finish cleanly and pass exact-fork prediction metrics before three-seed
  control or transfer experiments are submitted.
- The current array was launched before `hidden_physics` was added directly to
  the task config, so its run metadata links the specification through the
  immutable dataset hash and published dataset metadata. Future runs record the
  strong profile inline as well.
- Fetch state-flow control remains a null after fresh-scene confirmation and
  validation-only solver calibration. Do not tune more objectives on these
  seed ranges; improve controller action ranking or nominal model accuracy
  before spending additional control evaluation budget.
