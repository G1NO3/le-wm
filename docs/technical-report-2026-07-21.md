# Technical Report: Flow-Matched Residual Kernels for a Deterministic Latent World Model

Date: 2026-07-21 · Branch: `latent-residual-flow` · Base commit: `60ea2b6`
Scope: full review of the repository, the mathematical approach, the experiment
design, and all results recorded to date. Written to be readable without prior
knowledge of the project's internal vocabulary; every symbol is defined where
it first appears.

---

## 1. Executive summary

This repository extends **LeWorldModel (LeWM)** — a deterministic model that
predicts how a scene will evolve, working in a learned compressed vector
representation of camera images rather than in pixels — with a **learned noise
model**. The deterministic model answers "what will most likely happen next";
the added component answers "how uncertain is that prediction, and in which
directions of the representation space can reality deviate". The combination
turns a single-outcome predictor into a **probabilistic** one: it can generate
many plausible futures, which a planner can then use to prefer actions that are
robust to bad outcomes.

Status in one paragraph: the prediction-level case is made at first-seed
strength — adding the noise model improves a proper probabilistic accuracy
score by **~30% over the deterministic baseline** on a hard stacking task,
with most of that gain (~23 points) coming from simply acknowledging noise at
the right per-dimension scale, and the remainder from learned conditioning.
The control-level case is **not yet made**: a 250-episode planning pilot showed
the stochastic planner slightly *reducing* success (36.8% → 34.0%) while
improving a worst-case distance metric, and a follow-up diagnostic revealed
why — the evaluation windows were placed at task stages where the true
environment is nearly deterministic over the planning horizon, so injected
model noise could only hurt. The corrected experiment (evaluate where the true
randomness lives, at the final placement stage) is the immediate next step.
Two preregistered success criteria are currently failing or at risk and are
flagged in §9.

---

## 2. Background: the deterministic base model

### 2.1 What the base model is, in plain terms

LeWM is a "world model": a neural network trained on logged robot experience
(camera images plus the actions the robot took) that learns to predict the
future. It does not predict future *images*. Instead it first compresses each
image into a short vector, and learns to predict the *next vector* from recent
vectors and actions. Planning then happens by searching over candidate action
sequences and asking the model where each sequence leads in vector space,
picking the sequence that lands closest to a goal image's vector. This family
is called a *joint-embedding predictive architecture* (JEPA); "joint-embedding"
just means both the current observation and the prediction target live in the
same learned vector space.

### 2.2 The base model, mathematically

Let $x_t \in \mathbb{R}^{3\times224\times224}$ be the camera image at model
step $t$ and $u_t \in \mathbb{R}^{k a}$ the block of $k$ consecutive raw
actions (each of dimension $a$) executed during that model step ($k$ is the
"frameskip"; e.g. $k=5$, $a=5$ for the manipulation tasks). The model has:

- an **encoder** $f_\psi$ (a small vision transformer) and projection head
  $P_\psi$ producing the latent state
  $z_t = P_\psi(f_\psi(x_t)) \in \mathbb{R}^{D}$ (e.g. $D=192$ on PushT);
- an **action embedder** $g_\psi(u_t)$;
- a **predictor** $\Phi_\psi$ that maps a history window of $H$ latents and
  action embeddings (default $H=3$) to a prediction of the next latent:

$$
\hat z_{t+1} \;=\; \Phi_\psi\!\big(z_{t-H+1:t},\; g_\psi(u_{t-H+1:t})\big).
$$

Training minimizes two terms over batches of length-$T$ clips:

$$
\mathcal{L}_{\text{LeWM}}
= \underbrace{\;\mathbb{E}\,\big\lVert \hat z_{t+1} - z_{t+1} \big\rVert_2^2\;}_{\text{prediction loss}}
\;+\; \lambda \, \underbrace{\mathcal{L}_{\text{SIGReg}}(z)}_{\text{anti-collapse regularizer}} .
$$

The regularizer pushes the distribution of latents toward an isotropic
Gaussian. Without it, the encoder could map every image to the same vector and
achieve zero prediction loss ("representation collapse"); the regularizer
makes that degenerate solution costly. $\lambda$ is the single tunable loss
weight — the base paper's selling point is that this two-term objective trains
stably end-to-end from pixels.

Both the prediction and the encoding of the *target* frame use the same
weights (no momentum copy of the encoder), and the model is small (~15M
parameters, single-GPU trainable).

### 2.3 Planning with the base model

Given a current image and a goal image, planning solves

$$
u^\star_{1:T} = \arg\min_{u_{1:T}} \; J(u_{1:T}),
\qquad
J = \big\lVert \hat z_T(u_{1:T}) - z_{\text{goal}} \big\rVert_2^2 ,
$$

where $\hat z_T$ is obtained by rolling the predictor forward
autoregressively (each predicted latent is fed back in as history). The outer
optimizer is the cross-entropy method (CEM): sample a population of candidate
action sequences from a Gaussian, score each with $J$, refit the Gaussian to
the lowest-cost "elite" candidates, repeat. The model only supplies the cost
$J$; everything stochastic about the search lives in CEM itself.

The important property for this project: **the transition model is
deterministic**. Given the same history and actions it always predicts the
same single future. It has no way to represent "with 25% probability the
object slips out of the gripper."

---

## 3. The proposed extension: a residual noise model

### 3.1 The core decomposition

Instead of replacing the deterministic model, keep it as the **nominal**
(best-guess) dynamics and learn a distribution over its *errors*. Define the
one-step **residual**

$$
r_t \;=\; z_{t+1} - \hat z_{t+1}
\;=\; z_{t+1} - \Phi_\psi(z_{t-H+1:t}, u_{t-H+1:t}) \in \mathbb{R}^D .
$$

If the nominal model is well trained, $r_t$ contains exactly the part of the
future the deterministic model *cannot* explain — ideally the genuine
randomness of the environment ("aleatoric" noise: slips, contact outcomes),
plus whatever systematic error ("bias") the nominal still has. The stochastic
transition model is then

$$
z_{t+1} \;=\; \underbrace{\Phi_\psi(z_{\le t}, u_{\le t})}_{\text{nominal}}
\;+\; \underbrace{\mu + \sigma \odot \tilde r}_{\text{sampled residual}},
\qquad
\tilde r \sim p_\theta(\,\cdot \mid c_t\,),
$$

where $\mu, \sigma \in \mathbb{R}^D$ are fixed per-dimension statistics of the
residuals (their mean and standard deviation over the training set, computed
against a frozen nominal checkpoint), $\odot$ is elementwise multiplication,
and $p_\theta$ is a learned distribution over **whitened** residuals
$\tilde r = (r - \mu)/\sigma$. Whitening matters: it makes "unit-scale noise"
a meaningful default and puts all $D$ dimensions on equal footing for the
learned model.

The **condition** $c_t$ tells the noise model what situation it is in:

$$
c_t = \big[\, z_t \,\Vert\, g_\psi(u_t) \,\Vert\, \hat z_{t+1} \,\big] \in \mathbb{R}^{3D}
\quad (\text{optionally } \Vert\, h_t \text{ — see §3.4}).
$$

An "unconditional" ablation replaces $c_t$ with zeros.

Design invariant enforced throughout the code: with the residual component
disabled, the model is byte-for-byte the vanilla LeWM (checkpoint
state-dictionaries stay compatible; residual buffers are non-persistent unless
a kernel is attached). The stochastic part is strictly additive and optional.

### 3.2 The flow-matching residual model

The main learned distribution $p_\theta$ is a **conditional flow**. Plain
description: instead of writing down a probability density, the model learns a
velocity field that continuously *transports* easy-to-sample noise into the
target distribution. To sample, draw a standard Gaussian vector and integrate
the velocity field for one unit of an artificial "transport time"
$\tau \in [0,1]$.

Training uses **flow matching** with the linear (rectified-flow) path. For a
whitened residual target $\tilde r$ and noise $\epsilon \sim \mathcal N(0, I_D)$:

$$
z_\tau = (1-\tau)\,\epsilon + \tau\,\tilde r,
\qquad \tau \sim \mathcal U[0,1],
$$

and the straight-line path's velocity is constant in $\tau$:
$\frac{d z_\tau}{d\tau} = \tilde r - \epsilon$. The network
$v_\theta(\tau, z_\tau, c)$ — an MLP (4 hidden layers of width 512, LayerNorm,
GELU, zero-initialized output layer) with a sinusoidal embedding of $\tau$ —
is trained by simple regression onto that velocity:

$$
\mathcal{L}_{\text{FM}}
= \mathbb{E}_{\tilde r,\,\epsilon,\,\tau}
\big\lVert v_\theta(\tau, z_\tau, c) - (\tilde r - \epsilon) \big\rVert_2^2 .
$$

Two implementation details worth recording:

- The zero-initialized output head makes the flow start as the identity map
  (samples = plain Gaussian noise), so turning it on cannot destabilize early
  training.
- $\tau$ is rescaled by `time_scale=1000` before the standard sinusoidal
  embedding with `max_period=10000`; without this, $\tau\in[0,1]$ never
  excites the high-frequency channels and the time embedding is effectively
  constant. (An early checkpoint predates this fix; a compatibility fallback
  preserves its original behavior on load.)

**Sampling** integrates the learned field with $K$ Euler steps evaluated at
interval midpoints, then un-whitens:

$$
z^{(0)} = \epsilon,\qquad
z^{(i+1)} = z^{(i)} + \tfrac{1}{K}\, v_\theta\!\Big(\tfrac{i+0.5}{K},\, z^{(i)},\, c\Big),
\qquad
r = \mu + \sigma \odot z^{(K)} .
$$

$K$ is the "number of function evaluations" (NFE): 8 for evaluation, 4 for
planning. Cost per sample is $K$ MLP forward passes — cheap relative to the
predictor, which preserves LeWM's fast-planning identity.

Why a flow rather than directly maximizing likelihood? A flow can represent
multi-peaked distributions (e.g. "either the grasp holds or the object slips"
= two distinct future clusters), the training objective is a stable regression
(no likelihood, no adversarial game), and sample cost is a controllable knob
(NFE).

### 3.3 Baseline residual distributions (the comparison ladder)

To attribute any gain honestly, three simpler alternatives for
$p_\theta(\tilde r \mid c)$ are implemented behind the same interface
(`residual_kernels.py`), trained on the same frozen nominal, same whitening
statistics, same data split:

1. **Unit Gaussian (no learning).** $\tilde r \sim \mathcal N(0, I_D)$ — i.e.
   the only "knowledge" is the whitening statistics. Any model must beat this
   to justify learned parameters.
2. **Global diagonal Gaussian.** $\tilde r \sim \mathcal N(l, \mathrm{diag}(s^2))$
   with learned $l, s \in \mathbb{R}^D$ (scale parameterized
   $s=\mathrm{softplus}(\rho)$, initialized so $s=1$), trained by Gaussian
   negative log-likelihood
   $\frac{1}{2}\big(\frac{(\tilde r - l)^2}{s^2} + 2\log s + \log 2\pi\big)$.
3. **Conditional diagonal Gaussian.** An MLP maps $c \mapsto (m(c), s(c))$;
   loss $\frac{1}{2}\big(\frac{\tilde r - m}{s}\big)^2 + \log s$ per
   dimension. This is the key baseline: if it matches the flow, the residual
   distribution is (conditionally) just a Gaussian and the flow's extra
   machinery is unjustified.
4. **Conditional mixture of Gaussians** ($M=2$ components; mixture weights,
   means and scales from one MLP head; negative log-likelihood via
   log-sum-exp). A cheap test for two-cluster structure (slip vs. no-slip).

All heads consume the identical condition and, for reproducible planning,
accept externally supplied Gaussian noise and component-selection uniforms
(this is what makes common-random-number planning possible, §5).

### 3.4 Temporal memory: a small recurrent state

The environment's randomness is not independent across time: each episode
draws hidden physical properties once (§6.1) and keeps them for the whole
episode. After watching a few transitions, an ideal model should *adapt* —
e.g. after seeing one slip, predict slips as more likely. A memoryless kernel
conditioned only on $c_t$ cannot do this.

The implemented mechanism is deliberately minimal: one GRU cell (a standard
gated recurrent unit — a small update rule that mixes new evidence into a
persistent hidden vector) with hidden size 128:

$$
h_{t+1} = \mathrm{GRU}\Big(h_t,\; \mathrm{SiLU}\big(W\,\mathrm{LN}([\,c_t \Vert \tilde r_t\,])\big)\Big),
\qquad h_0 = 0,
$$

fed by the condition and the *realized* whitened residual of each completed
transition, and appended to the condition of the next one:
$c_t \leftarrow [\,c_t \Vert h_t\,] \in \mathbb{R}^{3D + 128}$.

The state is **functional/explicit**: the module never stores $h$ internally;
callers create, thread, and update it. This is a correctness decision — during
planning there are separate axes for action candidates and noise particles,
and an implicit hidden state would silently mix them.

Training uses teacher forcing within the clip window: before predicting the
final (deployment-time) transition, up to `max_history_updates` earlier
*ground-truth* residuals are pushed through the GRU (the window permits at
most $H-1=2$); an optional scheduled-sampling probability replaces
teacher-forced residuals with the model's own samples to reduce
train/deploy mismatch. The final target never enters its own conditioning
state (no label leakage).

At deployment (`residual_policy.py`), the policy maintains one $h$ per
environment, updated once per completed model step from *real* observations:
it buffers per-step frame embeddings and raw action chunks (re-assembling the
$k\cdot a$-dimensional action block the encoder was trained on from $k$
individually normalized raw actions), computes the realized residual
$z_{t+1}^{\text{obs}} - \hat z_{t+1}$, and updates $h$. During imagined
rollouts, each particle updates its own copy of $h$ with its own sampled
residuals.

### 3.5 Joint vs. two-stage training

Two supported regimes:

- **Joint** (`train.py`): total loss
  $\mathcal L = \mathcal L_{\text{LeWM}} + \lambda_{\text{FM}}\,\mathcal L_{\text{kernel}}$,
  with whitening statistics tracked by an exponential moving average (EMA,
  decay 0.99) during training. Defaults detach both the residual target and
  the condition, so the kernel's gradients do not perturb the deterministic
  predictor (the non-detached variant is a planned ablation).
- **Two-stage / frozen-nominal** (the regime of all current science runs):
  train (or download) a vanilla nominal, freeze it (`freeze_nominal()` — also
  pins its BatchNorm statistics and disables dropout against Lightning's
  recursive `train()` calls), compute *exact* whitening statistics over all
  training clips once, freeze them, then train only the kernel head. This
  guarantees the deterministic baseline is untouched and makes every kernel
  comparable by construction.

The training loss is computed **only on the final history-conditioned
transition of each clip** ("deployment token"), because that is the only
conditional distribution rollouts ever consume; training all predictor tokens
would optimize a different (easier) mixture of conditionals.

An important supporting detail: the whitening statistics loader verifies
SHA-256 hashes of the nominal checkpoint, dataset, and split manifest before
accepting a statistics file — mismatched provenance is a hard error, not a
silent skew.

---

## 4. What "better" means: evaluation mathematics

Judging a *distributional* prediction requires proper scoring rules — metrics
minimized exactly when the model reports the true distribution. The suite
(`stochastic_metrics.py`) includes:

**Energy score** (primary). For model samples $X, X' \stackrel{iid}{\sim} P$
and observed outcome(s) $y$:

$$
\mathrm{ES}(P, y) = \mathbb{E}\,\lVert X - y \rVert_2 \;-\; \tfrac12\,\mathbb{E}\,\lVert X - X' \rVert_2 .
$$

The first term rewards putting mass near the truth; the second term rewards
honest spread (a model that hedges by spreading out pays for it, a model that
collapses to a point loses the discount). It is a strictly proper multivariate
generalization of the absolute error. Reported as **skill** against a paired
baseline:

$$
\mathrm{skill} = 1 - \frac{\sum_i \mathrm{ES}(P^{\text{model}}_i, y_i)}{\sum_i \mathrm{ES}(P^{\text{base}}_i, y_i)} ,
$$

summed over contexts before the ratio so near-zero baseline contexts cannot
blow up the statistic. Positive = better than baseline; the standing project
rule is that the reference baseline is **the deterministic model** (its
"sample set" being the single nominal prediction).

**Attribution controls.** Two companion scores separate *why* a model wins:
the mean-error-explained fraction (does the kernel's predictive *mean* correct
nominal bias?) and skill versus a residual-mean-only corrector (is the win
genuinely distributional, i.e. does sampling beat just shifting the mean?).

**Calibration.** Per-dimension 90% central-interval coverage (fraction of
held-out outcomes inside the sample-estimated 5%–95% interval; target 0.90),
quantile expected calibration error (ECE), and randomized probability integral
transform (PIT) values.

**Covariance match.** Relative Frobenius error
$\lVert \widehat\Sigma_{\text{model}} - \widehat\Sigma_{\text{emp}} \rVert_F / \lVert \widehat\Sigma_{\text{emp}} \rVert_F$
between model-sample and held-out empirical residual covariance — sensitive to
cross-dimension correlation that diagonal Gaussians cannot express.

**Temporal structure** (for multi-step rollouts): trajectory-level energy
score, lag-1 autocorrelation error, adjacent-step cross-covariance error —
these detect a model that has the right per-step spread but the wrong
*persistence* (memory should help exactly here).

**Distributional distances**: RBF maximum mean discrepancy and sliced
Wasserstein as secondary checks.

**Hidden-mode separation** (dataset difficulty gate). For fork realizations
with binary mode label $m$, grouped by context $g$:

$$
d = \frac{\big\lVert \overline{\Delta} \big\rVert_2}{s_{\text{pooled}}},
\qquad
\overline{\Delta} = \frac1{|G|}\sum_{g} \big(\bar y_{g,m=1} - \bar y_{g,m=0}\big),
$$

with $s_{\text{pooled}}$ the within-context, within-mode RMS deviation.
Pairing within context first is essential: pose differences *between* contexts
must not masquerade as a hidden-physics effect. Units are "pooled standard
deviations" — $d \ge 1$ means the hidden mode visibly moves the future.

**Uncertainty on comparisons**: paired bootstrap over contexts (resample
context indices, recompute the paired difference, report the 95% interval).
Pairing removes between-context variance from the comparison.

---

## 5. Planning under uncertainty

With a stochastic transition model, "the cost of an action sequence" becomes a
random variable. The planner (extended `get_cost` in `jepa.py`) draws $N$
**particles** — independent full-horizon rollouts $j = 1..N$, each with its
own noise sequence and its own memory state — per candidate, and aggregates:

- **Mean objective**: $\bar J = \frac1N \sum_j J_j$. Note that for the
  squared-distance cost this decomposes as
  $\mathbb{E}\lVert \hat z_T - z_g \rVert^2 = \lVert \mathbb{E}\hat z_T - z_g \rVert^2 + \mathrm{tr}\,\mathrm{Var}(\hat z_T)$:
  the mean objective equals the deterministic-style cost of the average
  endpoint *plus a penalty on predicted spread*. It therefore only ranks
  candidates differently from the deterministic planner insofar as the model's
  spread is **action-dependent** — and it actively penalizes actions the model
  is unsure about, which backfires if the model's spread is miscalibrated
  (this is exactly what the E3/E5 results found, §8.5–8.6).
- **CVaR objective** (conditional value-at-risk, i.e. "average of the worst
  cases"): with tail fraction $\alpha = 0.25$,
  $\mathrm{CVaR}_\alpha = \frac{1}{\lceil \alpha N \rceil} \sum_{j \in \text{worst } \lceil \alpha N \rceil} J_j$.
  This optimizes robustness: prefer plans whose bad outcomes are least bad.

Variance-control details that make small $N$ (default 8) usable:

- **Common random numbers**: the same $N$ noise sequences (Gaussian draws and
  mixture-component uniforms) are reused across all candidates in a CEM
  population, so candidate ranking differences reflect the actions, not
  sampling luck.
- A dedicated, seeded generator (`sampling_seed`) decouples planner noise from
  global process randomness; particle chunking bounds memory.

Optional **physical-probe cost**: a ridge-regression probe $W$ mapping latents
to object positions permits a collateral-damage penalty
$w \cdot \lVert (W\hat z_T - W\hat z_{\text{ctx}})_S \rVert^2$ on
non-target-object displacement (indices $S$) — a step toward costs aligned
with *physical* failure rather than raw latent distance.

A privileged **simulator-particle oracle** (`privileged_mpc.py`) provides the
planning ceiling: the same particle MPC but with the real simulator (under
resampled hidden modes) as the transition kernel. The deployable kernels are
never compared against anything privileged without that separation being
explicit.

---

## 6. The evaluation environment and ground-truth protocol

This is the project's second contribution and arguably its most novel: a way
to score stochastic world models against the environment's **true**
conditional distribution rather than against single observed outcomes only.

### 6.1 Controlled hidden randomness

`stochastic_physics.py` wraps MuJoCo manipulation environments (OGBench cube
stacking; RoboCasa planned) with two mechanisms, both invisible to the policy
and the model (exposed only under `privileged/*` keys that must never appear
in model inputs):

- **Episode-constant hidden modes.** At each reset, three physics multipliers
  are drawn independently from two-point sets — object mass
  $\in \{0.5, 1.5\}$, object/surface friction $\in \{0.3, 1.2\}$, gripper-pad
  friction $\in \{0.2, 1.0\}$ (the "strong" profile) — giving $2^3 = 8$ hidden
  modes held fixed for the episode. This creates exactly the structure the GRU
  memory is meant to exploit: persistent, inferable-from-experience latent
  causes.
- **Transient slips.** At each grasp onset (gripper commanded to close after
  being open), with probability $p_{\text{slip}} = 0.25$, pad friction is
  multiplied by 0.1 for exactly 10 simulator substeps (restored via a chained
  MuJoCo control callback). This creates discrete, unpredictable-in-principle
  events — irreducible aleatoric noise with a two-cluster outcome structure.

Weaker profiles ("medium", "mild") exist for difficulty calibration. Risk
metrics (drops, collateral displacement of non-target cubes, grasp retries,
final goal distance) and an exact copy of every commanded action are logged
into the dataset.

### 6.2 Exact simulator forks

For a saved context (full simulator state: `qpos`, `qvel`, actuator state,
mocap targets, warm-start accelerations, wrapper bookkeeping, RNG state), the
fork tool replays the **identical commanded action sequence** $R$ times, each
time resampling the hidden mode and slip randomness. The result is $R$ draws
from the *true* conditional distribution

$$
p^\star\big(s_{t+1:t+H} \,\big|\, s_t,\, u_{t:t+H-1}\big),
$$

against which model samples can be scored with the §4 metrics — a
ground-truth conditional test no real-robot dataset can provide. Preregistered
fork scale: 512 contexts × 128 realizations at horizons $H \in \{1,5,10,20\}$
(pilot-scale: 64 × 32 at $H=5$). Contexts are balanced across contact stages
so the evaluation is not dominated by free-space motion where nothing
interesting happens.

### 6.3 Physical probes

Frozen ridge-regression maps from latents to privileged physical state (cube
positions) allow model rollouts to be visualized and scored in interpretable
physical coordinates, and power the collateral cost in §5. Probes are fit once
on training data and never backpropagated through.

---

## 7. Experiment design

### 7.1 Design principles

The study (`config/studies/complex_stochastic.yaml` + `docs/paper-plan.md`) is
built around three disciplines that are worth naming because they shape every
run:

1. **Preregistered gates.** Each stage has numeric pass criteria written down
   *before* the run; a failing gate blocks the downstream compute (enforced
   mechanically — the Slurm submit helper verifies the previous gate's hashed
   evidence file before calling `sbatch`).
2. **Single-variable ladders.** Every comparison changes one thing: all
   kernels share one frozen nominal, one dataset split, one set of whitening
   statistics, one model seed (pilot) or three (paper), identical hardware.
3. **Provenance hashing.** Datasets, split manifests, checkpoints, statistics
   files, and fork evidence all carry SHA-256 hashes that are cross-checked at
   load time and recorded in run metadata; splits are episode-disjoint (no
   episode contributes to both train and validation) and immutable.

### 7.2 The gate sequence

| Gate | Question | Pass criteria (preregistered) | Status |
| --- | --- | --- | --- |
| 1 | Does the fork reproduce vanilla LeWM? | PushT regression ≤ 5 points | passed (implicitly; formal record thin — see §9) |
| 2 | Is the chosen task *hard for the model but doable by an expert, and genuinely stochastic*? | 100 episodes: expert success ∈ [0.5, 0.98]; deterministic LeWM ∈ [0.15, 0.6]; expert−model gap ≥ 0.2; fork mode separation ≥ 1 SD | **passed** on quadruple stacking (expert 0.95, LeWM 0.38/0.26, separation 5.18 SD). Double-stack **failed** (`fail_too_easy`: expert 1.0) and was demoted to a non-regression task — the gate did its job |
| 2b | Is the 1,000-episode training pilot calibrated? | expert ∈ [0.5, 0.95]; separation ≥ 1 SD | **passed** (expert 0.914; separation 10.09 SD; 383,766 transitions; all 8 modes; 781 slips) |
| 3 | Does the stochastic kernel beat deterministic on exact forks, and does memory beat memorylessness? | energy gain vs deterministic ≥ 0.10; memory gain vs memoryless flow ≥ 0.05; coverage₉₀ ∈ [0.87, 0.93] | partially measured; **two criteria at risk** (§9) |
| 4 | Does it help control? | 3 seeds; success gain ≥ 5 points or worst-decile distance gain ≥ 10% | first pilot negative on success, positive on risk metric (§8.5) |
| 5 | Does it transfer? | octuple stacking; RoboCasa pick-and-place | not started |

The difficulty gate (gate 2) deserves emphasis as good methodology: the first
task attempt (double stacking) was *rejected by its own gate* despite
substantial sunk collection cost, because a 100%-success expert leaves no
room to demonstrate stochastic benefit. The task was then made harder
(four-high stacking) rather than the gate being loosened.

### 7.3 The experiment matrix

**Tier 1 (required):**

- **E1 — Prediction on exact forks.** All kernels {conditional Gaussian, GMM,
  flow, GRU-flow} × 3 seeds, scored against 512-context × 128-realization
  forks at $H \in \{1,5,10,20\}$ with paired bootstrap CIs; memory ablations
  run each context with **correct** memory (built from the true prefix),
  **reset** ($h=0$), and **shuffled** (memory stolen from a different episode
  with a different hidden mode — constructed so the donor differs in *both*
  episode and mode, otherwise the ablation is contaminated). Correct ≫ reset
  establishes memory usefulness; correct ≈ shuffled would expose the memory as
  a generic prior rather than context inference.
- **E2 — Dual nominal.** Repeat the kernel ladder against two nominals: the
  official pretrained checkpoint (weak on this data) and a **task-adapted**
  nominal (vanilla LeWM trained to convergence on the pilot data). This is the
  design that separates "the kernel corrects the nominal's bias" from "the
  kernel models genuine randomness" — under a near-perfect nominal, whatever
  benefit survives must be aleatoric. (Result: §8.3–8.4.)
- **E3 — Control.** Particle MPC (8 particles, NFE 4, common random numbers),
  all kernels × {mean, CVaR} vs deterministic CEM, on 100 starts × 5 physics
  seeds with hash-pinned identical start contexts across arms; privileged
  simulator-particle oracle as ceiling; wall-clock budgets 1/4/10 s.
- **E4 — Non-regression.** Joint-trained vs vanilla LeWM on PushT: pred-loss
  within ±10%, planning unchanged. (Same-budget vanilla baseline still owed.)
- **E5 — Spread-growth diagnostic.** Model particle spread vs true fork spread
  at $H \in \{1,5,10,20\}$: the direct test of whether the kernel's noise
  *grows over the horizon* the way reality's does. (First half done — §8.6.)
- **E9 — Memory-filter probe** (runs before E3). Replays held-out episodes
  into the GRU and measures: (a) memory-vs-reset energy skill as a function of
  update depth 1–200; (b) hidden-mode decodability from $h$ (ridge and MLP
  probes vs the 0.18 majority baseline); (c) state norms/saturation
  (stability). Decides whether truncated backpropagation-through-time training
  is needed. (Result: §8.4.)

**Tier 2:** external baselines (PETS-style ensemble of 5 nominal predictors;
diffusion residual head at matched NFE; full-covariance Gaussian oracle,
analysis-only), ablations (detach variants, loss weight, EMA vs exact
statistics, unconditional kernel, scheduled-sampling memory, NFE ∈
{1,4,8,16}), surprise/violation-of-expectation study (energy-score surprise vs
deterministic MSE surprise on 7 event types), calibration→control correlation,
one-step distilled sampling.

**Tier 3:** RoboCasa transfer, octuple stacking, dataset-scale curve.

### 7.4 Statistical protocol

Three seeds for paper claims (single seed labeled as pilot everywhere it
appears); paired bootstrap 95% CIs over contexts for every model comparison;
identical, hash-pinned evaluation contexts across arms; energy skill computed
sum-then-ratio; stage-conditioned reporting (per contact stage) after the E3
pilot showed aggregate numbers hide opposite-signed stage effects. Reporting
rule in force for the whole project: **every claimed benefit is quantified
against the vanilla deterministic LeWM, rung by rung**, so no rung's gain is
silently attributed to a higher one.

---

## 8. Results to date

### 8.1 Infrastructure verification (PushT, sky1)

One-epoch joint training on PushT completed (A40, 1h42m): prediction loss
healthy (val 0.067), flow-matching loss well below its $\approx 2$ untrained
plateau (1.25). Held-out residual evaluation (6,144 targets, $D=192$, NFE 8):
flow vs diagonal Gaussian covariance error 0.476 vs 0.840, 90% coverage 0.863
vs 0.817, quantile ECE 0.0263 vs 0.0241 (slightly worse). Smoke-level only —
one epoch, pre-`time_scale`-fix — but it validated the entire train/eval
pipeline and showed the first real distributional signal.

### 8.2 Task screening (quadruple stacking)

Against the official `lewm-cube` checkpoint on four-high stacking with strong
hidden physics: scripted expert 0.95 vs deterministic LeWM 0.38 (ordinary
windows) and 0.26 (post-contact windows; per-stage 0.23/0.23/0.33) — a
69-point expert–model gap with fork mode separation 5.18 SD (screen) / 10.09
SD (pilot). The setting is exactly what the method needs: reality is
multi-modal where the deterministic model is weak.

### 8.3 The benefit ladder (E2, one-step, task-adapted nominal, single seed)

The centerpiece prediction result. First, the dual-nominal decomposition:
task-adapting the nominal (30 epochs on pilot data) dropped validation
prediction loss from 0.0559 to **0.00227** (25×) and the residual mean norm
from 1.110 to **0.048** (23×) with per-dimension scale 0.494 → 0.028. In
words: against the official nominal, the "residuals" were mostly the nominal's
own systematic error; against the task-adapted nominal they are close to pure
environment noise. All rungs below are on the clean (task-adapted) side.

Energy-score skill vs the deterministic baseline (2,048 held-out one-step
targets, 16 samples/context, NFE 8):

| Rung | Model | Skill vs deterministic | Increment |
| --- | --- | --- | --- |
| 0 | Deterministic LeWM | 0 (definition) | — |
| 1 | Unit Gaussian, whitened (no learning) | **+23.1%** | +23.1 |
| 2 | Conditional diagonal Gaussian | **+32.3%** | +9.2 |
| 3 | Memoryless flow | **+28.8%** | (−3.5 vs rung 2) |
| 4 | GRU-memory flow | **+30.3%** | +1.5 vs rung 3 |
| 5 | Control | unmeasured at this rung | — |

Supporting metrics: covariance error 0.906 / 0.874 / 0.774 / **0.769** (unit
G / cond. G / flow / GRU-flow); every kernel beats its own mean-only
correction by ~26% (the gain is genuinely distributional, not bias
correction); 90% coverage ≈ 0.81 for **all** heads.

Honest readings, recorded in the repo and endorsed here:

- The single largest contribution is rung 1 — *correctly scaled noise with no
  learning at all*. This is a finding, not an embarrassment, but it must be
  stated (and is).
- One-step, under a near-perfect nominal, **the flow has not separated from
  the conditional Gaussian** (the Gaussian actually wins raw energy score
  9.55 vs 10.04; the flow's edge is correlation structure). One-step residuals
  of a well-fit model being near-Gaussian is theoretically unsurprising; the
  flow's real case must come from *conditional, multi-step, contact-window*
  evaluation (the exact-fork gate) and the full-covariance oracle comparison.

### 8.4 What the memory actually is (E9, replicated across both nominals)

Memory-vs-reset energy skill by replay depth (task-adapted nominal):
+0.24/+0.27/+0.26/+0.24 at depths 1/2/5/12, decaying to +0.05..+0.16 at
30–200, all paired CIs above zero; hidden-state norms stable (4–7, ≤4%
saturated) despite training with at most 2 teacher-forced updates — so no
TBPTT fix is needed for stability.

But the hidden physics mode is **not decodable** from the GRU state at any
depth: ridge and MLP probes sit at the 0.18 majority-class baseline, per-axis
at chance, with no growth in observation count — under *both* nominals, which
kills the "nominal bias was masking the mode signal" explanation. Conclusion
adopted by the project: the memory is **adaptive recent-context
conditioning** (a short-horizon error-statistics tracker), not a Bayes filter
over hidden modes. The paper's central causal chain — calibration →
*mode inference via memory* → better control — is broken at its middle link
as originally worded and has been reframed. This is a genuinely valuable
negative result; §9 discusses the consequence for gate 3.

### 8.5 Control (E3): smoke, pilot, and diagnosis

Smoke ($n=10$, identical hash-pinned contexts): deterministic 40% success /
0.215 worst-decile distance; memoryless flow (8 particles, mean) 50% / 0.186;
GRU-flow with live online memory 40% / **0.138** (best risk arm; the online
action-chunking fix that enabled live memory is verified by tests asserting
the online path reproduces the training-time transition observation exactly).

Pilot (250 evaluations = 50 contexts × 5 physics seeds; first two arms):

| Arm | Success | Stage 1 | Stage 2 | Stage 3 | Worst-decile dist. |
| --- | --- | --- | --- | --- | --- |
| Deterministic CEM | **36.8%** | 26% | 29% | 56% | 0.130 |
| Flow, mean objective | 34.0% | 19% | 27% | 58% | **0.114** |

The smoke's success bump did not survive scale: mean-objective stochastic
planning does not raise success and slightly lowers it, *while* improving the
worst-case distance. Stage-conditioned, it **hurt the hardest stage**
(26%→19%). The mechanism (per §5's decomposition): the mean objective adds a
penalty on predicted spread, so if the model injects spread where reality has
none, good aggressive candidates are wrongly penalized.

### 8.6 The E5 ground-truth spread result — the key reprioritizing finding

True fork spread (64 contexts × 32 realizations, $H=5$), final-horizon
per-dimension SD of physical state **by contact stage**:

| Stage 1 | Stage 2 | Stage 3 | Stage 4 (final placement) |
| --- | --- | --- | --- |
| 0.00002 | 0.00005 | 0.00003 | **0.02024** |

The environment's aleatoric variance is concentrated 400–1000× at the final
placement stage; stages 1–3 are *nearly deterministic* over the 5-step
horizon. The E3 pilot evaluated only stages 1–3 (stage 4 had been excluded as
"mostly retries"). Therefore: (a) the control experiment so far was run where
stochastic planning cannot help by construction; (b) the stage-1 degradation
is explained — the kernel is over-dispersed exactly where reality is tight
(the miscalibration E5 exists to detect); (c) the decisive control experiment
is at stage 4 / full episodes / longer horizons, which is now the plan of
record, together with a model-side spread comparison and $H \in \{10,20\}$
fork regeneration, an action-dependence (heteroscedasticity) check of the
kernel's variance — the precondition for *any* risk objective to discriminate
candidates — and a physical-failure-aligned cost via the probes.

---

## 9. Critical assessment

### 9.1 Strengths

1. **Clean core decomposition.** Nominal + whitened residual is the right
   factorization: it preserves the deterministic model exactly (non-regression
   by construction in the frozen regime), makes every uncertainty head
   plug-compatible, and reduces the learning problem to a normalized,
   near-zero-mean target.
2. **Unusually strong experimental hygiene.** Preregistered numeric gates that
   have actually rejected work (double-stack; the too-easy pilot);
   mechanically enforced gate→compute dependencies; SHA-256 provenance on
   every artifact with load-time cross-checks; episode-disjoint immutable
   splits; paired bootstrap; hash-pinned identical contexts across control
   arms; common random numbers; privileged tooling explicitly quarantined.
   This is well above field-typical practice.
3. **The exact-fork protocol is a real contribution** independent of the
   method: proper scoring against the true conditional distribution, with a
   commanded-action audit and hidden-mode ground truth, is something the
   stochastic-world-model literature mostly lacks.
4. **Honest attribution.** The rung ladder, the dual-nominal design, the
   mean-only-correction control, and the willingness to headline "most of the
   gain is correctly scaled noise" and "the memory is not a mode filter" —
   the project is measurably resistant to self-deception.
5. **Sound engineering details**: zero-init flow head; explicit functional
   memory state (candidate/particle axes cannot mix); deployment-token loss;
   label-leakage-free memory updates; frozen-nominal BatchNorm/dropout
   pinning against Lightning's recursive `train()`; time-embedding scale fix
   with backward compatibility; online action re-chunking verified by
   equivalence tests.

### 9.2 Risks and gaps, in decreasing order of importance

1. **The control-level case is unproven and the first evidence is negative on
   the primary metric.** Success 36.8% → 34.0% under the mean objective. The
   E5 explanation (evaluated where there is no exploitable randomness) is
   convincing and the redesign is right, but until a stage-4/full-episode
   experiment shows a success or preregistered risk-metric gain, the method's
   headline justification is open. The worst-decile improvement (0.130 →
   0.114, and 0.138 with live memory in the smoke) is real but secondary.
2. **Gate 3's memory criterion is failing on current one-step evidence.** The
   preregistered threshold is memory-vs-memoryless energy gain ≥ 0.05; the
   measured one-step increment is +0.015 skill (9.83 vs 10.04 energy ≈ 0.021
   relative). The fork-based multi-step gate — where temporal persistence
   should actually matter (lag-1, cross-covariance metrics) — is the memory's
   remaining chance; if it also lands under 0.05, the preregistration commits
   the paper to demoting the memory from co-contribution to ablation. Plan
   for that branch now rather than after.
3. **The flow is not yet earning its complexity.** One-step, the conditional
   Gaussian matches or beats it on the primary score. The flow's
   differentiators — covariance structure and potential multimodality — need
   the full-covariance Gaussian oracle (E6) and slip-window fork evaluation to
   be demonstrated. If a full-covariance (or low-rank) Gaussian closes the
   covariance gap, the honest conclusion is that a Gaussian kernel suffices
   at one step, and the flow's case must rest entirely on multi-step/
   multimodal settings.
4. **Coverage is below the preregistered band for every head** (~0.81 vs
   [0.87, 0.93]). Two candidate causes should be separated before concluding
   the kernels under-disperse: (a) genuine under-dispersion; (b) small-sample
   interval bias — the 5%/95% quantiles are estimated from only 16 samples
   per context, and finite-sample central intervals systematically
   under-cover (the full range of $n$ samples covers only $(n-1)/(n+1)$;
   $n=16$ makes a ~0.85 practical ceiling for a nominal 0.90 interval).
   Cheap fix: raise samples per context (or apply a finite-sample coverage
   correction) in the evaluator before treating this as a model failure.
   Note the tension with item 1: E5 says the kernel *over*-disperses at
   stages 1–3 in physical-probe space while coverage says it *under*-covers
   marginally in latent space — both can be true (wrong shape, not wrong
   volume), which itself argues for conditional (per-stage) calibration
   reporting.
5. **One-step training vs multi-step deployment.** The kernel is trained
   exclusively on one-step residuals but deployed autoregressively over
   horizons of 10–50 model steps, feeding its own samples back through the
   nominal. Nothing in the objective controls the *compounded* distribution;
   the memory partially couples consecutive steps but is itself trained with
   ≤2 teacher-forced updates (exposure bias; the scheduled-sampling option
   exists but defaults off). E5's model-side half and the trajectory-level
   metrics will quantify the damage; multi-horizon consistency is correctly
   flagged as an open problem (M4), but it is the most likely technical
   reason for planning-time miscalibration and deserves priority over further
   one-step refinements.
6. **Reproducibility gap: science runs on uncommitted code.** Nearly every
   experiment-log entry since 2026-07-19 reads "commit `14da4a3` plus
   uncommitted implementation". Data, splits, checkpoints and statistics are
   hash-pinned, but the *code* that produced them is not — which undermines
   the otherwise-excellent provenance chain. Recommendation: commit (even to
   WIP branches) before every submitted job; record `git describe --dirty`
   in run metadata and treat `-dirty` as a gate failure for science runs.
7. **Latent-goal cost is a weak proxy for task success** (one-step latent MSE
   0.00097 coexists with 40% task success). The physical-probe cost work is
   the right response; consider also scoring control arms directly on the
   logged risk metrics (drops, collateral displacement) which are already in
   the dataset.
8. Smaller technical notes: (a) the EMA statistics path averages batch
   standard deviations rather than variances — a Jensen-inequality bias
   (harmless for plumbing runs; the exact-statistics path used for science
   avoids it; worth a comment); (b) `GlobalDiagonalGaussian`'s docstring says
   its location/scale are buffers, but they are trainable `nn.Parameter`s —
   stale doc, and the "unit Gaussian (no learning)" rung label should be
   double-checked against what the eval script actually instantiates;
   (c) `residual_condition` builds a zeroed full-width condition for the
   "none" ablation — fine, but wasteful width; (d) sampling evaluates the
   velocity at midpoint times while training draws $\tau$ uniformly — correct
   (midpoint integration of a field trained on all $\tau$), just worth a
   comment since it looks like an off-by-half at first read.

### 9.3 Verdict on the approach

The decomposition, the baseline ladder, and the evaluation protocol are sound
and in places exemplary. The two load-bearing open questions are empirical,
correctly identified by the project's own diagnostics, and have concrete next
experiments attached: **(1)** does stochastic planning help where the
environment is actually stochastic (stage-4/full-episode E3 with CVaR, a
physical-failure cost, and a verified action-dependent spread), and **(2)** do
the flow and the memory earn their complexity beyond a conditional Gaussian on
the multi-step exact-fork gate? A negative answer to (2) still leaves a
publishable, honest paper (calibrated Gaussian residual kernels + the fork
protocol + the stage-resolved control analysis); a negative answer to (1)
does not — which is why the reprioritization onto stage-4 control is right.

---

## 10. Recommended immediate sequence

1. Finish the three queued E3 arms; report all five stage-conditioned with
   paired CIs (analysis script already staged).
2. Regenerate forks at $H \in \{10, 20\}$; run E5's model-spread half; add
   the heteroscedasticity check (does kernel spread vary across candidate
   actions?). These are cheap and gate everything else.
3. Stage-4 / full-episode control comparison with CVaR and the physical-probe
   collateral cost — the experiment the whole project now hinges on.
4. Fix the coverage evaluator's sample count before re-judging the
   [0.87, 0.93] gate band.
5. Add the full-covariance Gaussian oracle (E6) to the one-step analysis to
   settle the flow-vs-Gaussian question at that rung.
6. Commit working-tree code before every science submission; add `--dirty`
   detection to run metadata and gates.
7. Then: three-seed E1/E2 arrays and the fork-based memory gate that decides
   the memory's status in the paper.
