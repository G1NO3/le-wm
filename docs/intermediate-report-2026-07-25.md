# Stochastic Residual Dynamics for LeWM: Intermediate Report

**Date:** 2026-07-25
**Branch:** `latent-residual-flow`
**Scope:** prediction, planning, and MPC results to date, with a focus on
whether adding a learned *stochastic residual* to a frozen deterministic LeWM
improves **control success rate** — the metric we ultimately care about.

---

## 0. Executive summary

We augment a frozen deterministic LeWM with a conditional **flow-matched
residual kernel** (optionally with a small GRU memory or two persistent
"mode" heads), turning the deterministic point predictor into a calibrated
stochastic transition kernel that is cheap enough for particle MPC. We evaluate
it with **exact simulator forks** so that stochastic predictions are graded
against the *true* conditional distribution using proper scoring rules, not just
marginal likelihood or downstream return.

Headline findings, each stated **relative to the deterministic LeWM baseline**:

| Level | Result | Verdict |
|---|---|---|
| **Prediction** | Flow residual improves exact-fork energy score by **+34.8%** vs deterministic LeWM (3 seeds, H=10; 95% CI [+32.0, +37.3]) and **+20.2%** vs its own predictive mean | **Confirmed win** |
| **Planning (commitment)** | Label-free persistent-mode residual raises simulator success **52.3% → 56.2%** (+3.8 pp; 95% CI [+1.8, +5.9]) over its deterministic mean | **Confirmed win (special case)** |
| **Planning (general MPC)** | Deployable ordinary-state receding-horizon MPC: prediction improves but success rate does **not** (20% vs 18%, CI [−3, +7]) | **Null** |
| **Ceiling (this report)** | Exact-fork geometry screen: on the commitment task a distribution-aware oracle beats a mean oracle **77% → 99%** context-level success (+22 pp; 95% CI [+14, +30]) | **Large, clean headroom** |

The through-line: **stochastic residuals reliably improve prediction, and they
improve control success *in tasks that force commitment under hidden
uncertainty*, but not in ordinary receding-horizon control where re-planning
hides the benefit.** This report sharpens exactly when and why the benefit
appears, and quantifies the ceiling a better learned kernel is chasing.

Companion video (exact-fork oracle mechanism):
[`fetch_push_commitment_oracle_demo_20260725.mp4`](results/fetch_push_commitment_oracle_demo_20260725.mp4)
(poster: [`.png`](results/fetch_push_commitment_oracle_demo_20260725.png)).

---

## 1. Notation and method

We follow LeWM notation (Maes & Le Lidec et al., 2026) and extend it.

### 1.1 The deterministic LeWM base

LeWM is a JEPA trained end-to-end from pixels. Let $o_t$ be the observation and
$a_t$ the action.

- **Encoder** $E_\theta$ maps an observation to a latent embedding
  $z_t = E_\theta(o_t) \in \mathbb R^{d}$ (here $d=192$ for the cube/PushT
  latent).
- **Predictor** $P_\theta$ maps the current embedding and (encoded) action to a
  point prediction of the next embedding,
  $$\hat z_{t+1} = P_\theta\bigl(z_t, \tilde a_t\bigr), \qquad \tilde a_t = A_\theta(a_t).$$
- **Training** uses LeWM's two-term objective: a next-embedding prediction loss
  plus a Gaussian (SIGReg) regularizer that prevents representation collapse,
  $$\mathcal L_{\text{LeWM}} = \underbrace{\bigl\| \hat z_{t+1} - \operatorname{sg}(z_{t+1}) \bigr\|_2^2}_{\text{prediction}} \;+\; \lambda\, \underbrace{\mathcal L_{\text{reg}}(z)}_{\text{Gaussian latent}},$$
  where $\operatorname{sg}$ is stop-gradient. Control is done by latent-space
  MPC: roll $P_\theta$ over a horizon and minimize a goal-embedding distance.

**The limitation we target.** $P_\theta$ is a *deterministic* map: for a given
$(z_t,a_t)$ it returns a single $\hat z_{t+1}$. When the environment has hidden
aleatoric structure — an episode-constant but unobserved physical mode $m$
(e.g. surface friction) — the true one-step conditional
$p(z_{t+1}\mid z_t,a_t)$ is **multimodal**. A point predictor can at best
recover the conditional mean $\mathbb E[z_{t+1}\mid z_t,a_t]$, which for a
bimodal target lies in the **low-density valley between the modes** — a state
the system never actually visits. This is the failure mode the whole method
attacks.

### 1.2 Residual stochastic kernel

We freeze the deterministic model (either the official LeWM checkpoint or a
**task-adapted nominal** — vanilla LeWM retrained on the task's own data) and
learn a residual *on top of it*. Let $x_t$ denote the dynamics variable: the
latent $z_t$ for the latent variant, or the ordinary $28$-D Fetch observation
for the **deployable state variant** (used in all FetchPush/FetchSlide control
experiments; no privileged simulator state or friction labels enter the model).

Define the frozen **nominal delta**
$$\Delta^{\mathrm{nom}}(x_t,a_t) \;=\; P_\phi(x_t,a_t) - x_t ,$$
and the realized delta $\Delta_t = x_{t+1}-x_t$. Using train-only statistics
(residual mean $\mu_r$, diagonal scale $\sigma$; both hashed and immutable), the
**whitened residual target** is
$$r_t \;=\; \sigma^{-1}\bigl(\Delta_t - \Delta^{\mathrm{nom}}(x_t,a_t) - \mu_r\bigr).$$
Task adaptation matters here: on the quadruple-stack pilot, the residual mean
norm dropped from $1.11$ (official nominal) to $0.05$ (task-adapted), so $r_t$
is close to **pure aleatoric noise** rather than systematic nominal bias.

**Conditional flow matching.** The kernel is a velocity field
$v_\psi(\tau, r, c)$ with a transition condition
$c = c(x_t, a_t, \Delta^{\mathrm{nom}})$. With linear interpolation between a
Gaussian base sample $\epsilon\sim\mathcal N(0,I)$ and the target,
$r_\tau = (1-\tau)\,\epsilon + \tau\, r$, the flow-matching loss is
$$\mathcal L_{\mathrm{FM}} = \mathbb E_{\tau\sim U[0,1],\,\epsilon,\,(x_t,a_t)} \Bigl\| v_\psi\bigl(\tau, r_\tau, c\bigr) - (r - \epsilon) \Bigr\|_2^2 .$$
Sampling integrates the probability-flow ODE $\dot r = v_\psi(\tau, r, c)$ from
$r_0=\epsilon$ over $\tau\in[0,1]$ with a small number of function evaluations
(NFE $\in\{4,8\}$), giving $\hat r = r_1$. The **stochastic transition** is
$$x_{t+1} \;=\; x_t + \Delta^{\mathrm{nom}}(x_t,a_t) + \sigma\,\hat r,\qquad \hat r \sim K_\psi(\cdot\mid x_t,a_t).$$
Setting $\hat r$ to the kernel mean recovers a *deterministic residual* baseline;
setting $\sigma=0$ recovers the frozen LeWM. This nesting is what lets us
attribute gains rung by rung.

**GRU memory variant.** A one-layer, 128-D GRU carries a hidden state $h_t$
updated online from observed transitions; the condition becomes
$c_t = (x_t, a_t, \Delta^{\mathrm{nom}}, h_t)$. Diagnostics (E9) show the memory
acts as **adaptive recent-context conditioning**, not a Bayes filter over the
hidden mode (the mode is not decodable from $h_t$ at any replay depth).

**Label-free persistent modes.** For the commitment experiments we replace the
single kernel with two residual heads $g_\psi^{(1)}, g_\psi^{(2)}$ producing
whitened residuals $r^{(k)}(c)$. Paired episodes that share an initial scene
(known from collection, but *without* friction labels) are assigned to opposite
heads by a within-pair winner-take-all on object-motion dimensions, and each
episode is trained under **one persistent head assignment** for its whole
length:
$$x_{t+1}^{(k)} = x_t + \Delta^{\mathrm{nom}}(x_t,a_t) + \sigma\, r^{(k)}(c),\qquad k\in\{1,2\}.$$
The deterministic-mean baseline for this model is the head average
$\bar r = \tfrac12\bigl(r^{(1)}+r^{(2)}\bigr)$. Crucially, one-step base-noise
reuse does **not** reproduce this — persistent per-episode outcome identity is
the ingredient that makes the residual useful for commitment (see §4.3).

### 1.3 What is different from LeWM, precisely

| Aspect | LeWM (baseline) | This work |
|---|---|---|
| Transition | Deterministic point map $\hat z_{t+1}=P_\theta(z_t,\tilde a_t)$ | Stochastic kernel $x_{t+1}=x_t+\Delta^{\mathrm{nom}}+\sigma\,\hat r$, $\hat r\sim K_\psi$ |
| Uncertainty | None (single trajectory) | Calibrated conditional distribution (flow) $\pm$ memory $\pm$ persistent modes |
| Training | Two-term JEPA loss, end-to-end | Frozen nominal **unchanged**; add $\mathcal L_{\mathrm{FM}}$ on whitened residuals (base model's prediction quality untouched by construction) |
| Planning | Latent MPC minimizing mean goal distance | Particle MPC with common random numbers; mean **or** CVaR objective over sampled futures |
| Evaluation | Task return / marginal metrics | Exact simulator forks + proper scores vs the true conditional, paired-bootstrap CIs, hashed provenance |

---

## 2. Evaluation protocol

The evaluation is a contribution in its own right. From any state we take an
**exact MuJoCo fork**, resample the hidden physical mode $m$, and roll $N$
independent realizations, yielding samples from the *true* conditional
$p(x_{t+H}\mid x_t, a_{t:t+H})$. Against this ground truth we report:

- **Energy score** (a strictly proper scoring rule for distributions):
  $$\mathrm{ES}(F, y) = \mathbb E_{X\sim F}\|X-y\| - \tfrac12\,\mathbb E_{X,X'\sim F}\|X-X'\|,$$
  and **energy-score skill** relative to deterministic LeWM,
  $\mathrm{skill} = 1 - \mathrm{ES}(\text{model})/\mathrm{ES}(\text{det})$.
- **Covariance relative Frobenius error**, **quantile ECE**, **central
  interval coverage** for calibration.
- **Sampling-attribution gate**: model vs its *own predictive mean*, to prove
  gains are distributional and not just bias correction.
- **Paired scene-cluster bootstrap** 95% CIs throughout; immutable
  episode-disjoint splits; SHA-256 provenance on every dataset, checkpoint, and
  statistics artifact.

---

## 3. Results — Prediction

### 3.1 The benefit ladder vs deterministic JEPA (quadruple-stack, E2)

Energy-score skill vs the deterministic baseline, decomposed so no rung's gain
is attributed to a higher one (task-adapted nominal, one seed):

| Rung | Model | Skill vs deterministic LeWM |
|---|---|---|
| 0 | Deterministic LeWM | 0 (by definition) |
| 1 | Correctly-scaled Gaussian noise (**no learning**) | **+23.1%** |
| 2 | Conditional diagonal Gaussian | **+32.3%** |
| 3 | Memoryless flow | +28.8% |
| 4 | GRU-memory flow | +30.3% |

Reading: **most of the one-step gain (23 of ~30 pts) comes from simply
acknowledging noise at the right scale**; learned conditioning adds the rest;
one-step flow does not yet separate from the conditional Gaussian on a
well-fit nominal (its edge is correlation structure: covariance error 0.77 vs
0.87). The decisive test for the flow is the multi-step, conditional exact-fork
gate below.

### 3.2 Three-seed FetchPush (the clean prediction win)

On exact balanced low/high friction forks at horizon $H=10$:

- Flow residual improves energy score **+34.8% vs deterministic LeWM**
  (95% CI [+32.0, +37.3]);
- **+20.2% vs its own predictive mean** (95% CI [+19.2, +21.3]) — the gain is
  genuinely distributional, not mean correction;
- **Memory is not justified** on the pre-contact prior mixture: GRU-flow ties
  memoryless flow (paired diff 0.0023, CI [−0.045, +0.047]).

At one seed the absolute picture: deterministic energy $2.676$ vs
$\{1.825, 1.706, 1.703\}$ for {cond. Gaussian, flow, GRU-flow}; flow's paired
absolute improvement is $0.971$ vs deterministic (CI [0.851, 1.098]).

### 3.3 PushT (non-regression / calibration smoke)

On a 1-epoch checkpoint (smoke quality), the flow residual improves covariance
matching (rel. Frobenius **0.476 vs 0.840** Gaussian) and 90% interval coverage
(**0.863 vs 0.817**), at a slightly worse quantile ECE (0.0263 vs 0.0241).

**Prediction verdict:** stochastic residuals deliver a real, reproducible
distributional improvement over deterministic LeWM, at **zero cost** to the base
model's point-prediction quality (the nominal is frozen).

---

## 4. Results — Planning and MPC

Particle MPC evaluates a candidate action sequence $a_{1:H}$ by sampling $N$
residual particles with common random numbers and scoring goal cost. Two
selectors are compared **using the same kernel**; the only difference is whether
they reason about the distribution:

$$
a^\star_{\mathrm{det}} = \arg\min_a\; \ell\!\bigl(\hat x^{\text{mean}}_{1:H}(a),\, g\bigr),
\qquad
a^\star_{\mathrm{sto}} = \arg\min_a\; \mathbb E_{m}\!\bigl[\,\ell\bigl(x^{(m)}_{1:H}(a),\, g\bigr)\,\bigr].
$$

The deterministic selector optimizes a single mean trajectory; the
distribution-aware selector minimizes expected failure (or CVaR) over sampled
outcomes.

### 4.1 General receding-horizon MPC — null

The deployable ordinary-state model (28-D observation, two raw steps per
transition, no labels/sim-state) predicts better than deterministic
(energy $0.335$ vs $0.353$, **+5.2%** skill; +11.0% vs its own mean, both CIs
above 0) — **yet control success does not move**: after validation-only CEM
calibration, flow reached **20% vs 18%** deterministic (95% CI [−3, +7]). The
full-task shared-scene latent controller was likewise null. *Prediction gains
do not automatically become success gains* when the controller re-plans every
step and both friction modes usually prefer the same action.

### 4.2 Why commitment tasks are different

The gain appears only when the task **forces an irreversible choice before the
hidden mode is revealed**. At push onset the controller commits one finite push
pulse $a$; the block's closest approach to the goal is $d(a,m)$ and success is
$\mathbf 1[d(a,m)\le\tau]$. Because outcomes are bimodal (slippery overshoots,
sticky undershoots), the **mean trajectory** $d(a,\bar m)$ can sit on the goal
while *both* real modes miss — the valley. The distribution-aware selector
never picks such an action. This is the mechanism the video visualizes.

### 4.3 The confirmed control win (label-free persistent modes)

On 1,024 fresh commitment contexts / 2,048 exact low-high episodes, with a model
that never reads friction labels:

- Residual **mean** selection: **52.34%** success.
- **Distribution-aware** selection: **56.15%** (+**3.81 pp**; paired
  scene-cluster 95% CI **[+1.76, +5.86]**; all 8 shards positive).
- Both modes improve: low friction 80.57% → 85.74%, high 24.12% → 26.56%.
- Exact-distribution oracle on the same contexts: **+17.09 pp** — the ceiling.

Contrasts that scope the claim:
- **Supervised upper bound** (two heads *with* friction labels at train time):
  +6.45 pp (CI [+4.59, +8.30]); oracle 60.79% → 77.73%.
- **One-step flow with base-noise reuse: null** (62.21% vs 62.45%,
  CI [−1.71, +1.22]). Persistent per-episode outcome identity — not one-step
  stochasticity — is the necessary ingredient.

Artifact:
[`fetch_push_paired_modes_confirmation_20260722.json`](results/fetch_push_paired_modes_confirmation_20260722.json).

### 4.4 The ceiling, quantified (this report's new experiment)

We ran a **pure exact-fork geometry screen** — no learned model, no training —
to map how much headroom the commitment task offers and to select the most
favorable regime. For each push-onset context we roll a dense push-pulse library
through exact forks under two friction multipliers and score two *oracles*
(both using the true physics; they differ only in reasoning):

- **Mean oracle** = deterministic planner: picks the push whose
  *averaged-position* trajectory is closest to goal (valley-prone).
- **Distribution oracle** = stochastic planner: picks the push minimizing
  expected 0/1 failure across the two modes (hedges or commits).

We grade at two levels. **Episode-level** scores each hidden mode separately.
**Context-level** is the honest commitment metric: one committed push must
succeed **under both frictions** (you commit before nature draws $m$).

Confirmed on 100 fresh scenes (script
[`screen_fetch_push_commitment_geometry.py`](../scripts/eval/screen_fetch_push_commitment_geometry.py);
data [`fetch_push_commitment_geometry_confirm_20260724.json`](results/fetch_push_commitment_geometry_confirm_20260724.json)):

| Friction | $\tau$ | metric | mean oracle | dist oracle | gain (95% CI) |
|---|---|---|---|---|---|
| 0.25× / 2.5× | 5 cm | episode | 88.5% | **99.5%** | +11.0 [+7.0, +15.0] |
| 0.25× / 2.5× | 5 cm | **context** | 77.0% | **99.0%** | **+22.0 [+14.0, +30.0]** |
| 0.25× / 2.5× | 3 cm | context | 69.0% | 94.0% | +25.0 [+17.0, +34.0] |
| 0.2× / 3.0× | 5 cm | context | 33.0% | 58.0% | +25.0 [+17.0, +34.0] |

**Interpretation.** At 0.25×/2.5×, τ=5 cm, a distribution-aware planner with
perfect physics solves **99 of 100** uncertain-friction scenes; a deterministic
mean planner solves only **77**, because it commits to a push that works for
only one of the two possible frictions. The gap is structural (+22 pp, CI well
clear of 0), and a both-friction-safe "hedge" push exists in **99%** of scenes,
so the task genuinely rewards distribution-awareness. A secondary sweep showed
this is a *sweet spot*: extreme friction (0.15×/4.0×) drives the two landing
points too far apart to hedge, and very mild friction saturates the task.

The video/poster (§0) shows one such scene: the mean-oracle push succeeds if the
table turns out slippery (0.9 cm) but misses if it is sticky (8.8 cm); the
distribution-oracle push succeeds either way (0.6 cm / 4.3 cm).

**This +22 pp context-level ceiling is the target the learned kernel is
chasing.** The current learned realization (§4.3) captured ~+3.8 pp of a
comparable ceiling — so the open problem is closing the learned-vs-oracle gap,
not finding task value.

---

## 5. Negative results and scope (kept explicit)

- **FetchSlide transfer is a clean negative.** The exact task value is huge
  (mean-state oracle 0% vs distribution oracle 47.7% on 64 scenes), but learned
  prediction fails at the ballistic horizon: 500-pair H=50 mode-mixture energy
  $1.127$ vs $0.309$ nominal; learned control 0% vs 3.91%. A matched **vanilla
  LeWM** baseline confirms the base model also degrades badly (point energy
  $0.856$ at H1 → $13.832$ at H50, zero interval coverage; native control 7.8%
  vs 43.8% distribution oracle). Diagnosis: one-step-to-ballistic-horizon
  inconsistency, not lack of task value — next needs a direct
  multi-horizon/terminal-outcome kernel.
- **Fixed-friction posterior adaptation is a null.** After 5 observed steps,
  kernel spread contracts <0.05% and memory decodes friction near chance, even
  though friction is 99.6% decodable from raw displacement — a *representation*
  bottleneck, not a signal one.
- **General MPC is a null** (§4.1). The confirmed control win is specific to
  commitment-under-uncertainty tasks.

---

## 6. What we have achieved vs LeWM — one-paragraph statement for the paper

Starting from a frozen deterministic LeWM, we add a conditional flow-matched
residual kernel that (i) improves the exact-fork energy score by **+34.8%**
(3 seeds) over the deterministic model at no cost to its point predictions,
(ii) is cheap enough for particle MPC with mean/CVaR objectives, and (iii) in an
uncertainty-sensitive commitment task raises **actual simulator success by
+3.8 pp** over its own deterministic mean using a label-free persistent-mode
residual. We further establish, via an exact-fork oracle, that this task class
offers a **+22 pp** context-level ceiling for distribution-aware control — a
regime where a deterministic world model is not merely worse but structurally
wrong, committing to actions that live in the valley between physical outcomes.

---

## 7. Next steps

1. **Close the learned-vs-oracle gap** at 0.2×/3.0× first (free — reuses the
   trained seed-26041 checkpoint): re-score the learned commitment eval at the
   context level and against **vanilla LeWM**, then improve the label-free mode
   assignment / flow steps to lift +3.8 pp toward the +22–25 pp ceiling.
2. **(Optional) 0.25×/2.5× data collection + retrain** for the picture-perfect
   oracle→99% figure, if the cheaper regime confirms the direction.
3. **Multi-horizon/terminal kernel** for FetchSlide to fix the ballistic-horizon
   inconsistency.

---

## 8. Artifacts and provenance

- Geometry screen (this report): [`screen_fetch_push_commitment_geometry.py`](../scripts/eval/screen_fetch_push_commitment_geometry.py);
  results [`fetch_push_commitment_geometry_screen_20260724.json`](results/fetch_push_commitment_geometry_screen_20260724.json),
  [`fetch_push_commitment_geometry_confirm_20260724.json`](results/fetch_push_commitment_geometry_confirm_20260724.json).
- Oracle demo: [`fetch_push_commitment_oracle_demo_20260725.mp4`](results/fetch_push_commitment_oracle_demo_20260725.mp4) / [`.png`](results/fetch_push_commitment_oracle_demo_20260725.png).
- Learned commitment: [`fetch_push_paired_modes_confirmation_20260722.json`](results/fetch_push_paired_modes_confirmation_20260722.json),
  [`fetch_push_commitment_mode_confirmation_20260722.json`](results/fetch_push_commitment_mode_confirmation_20260722.json),
  [`fetch_push_commitment_flow_confirmation_20260722.json`](results/fetch_push_commitment_flow_confirmation_20260722.json).
- Prediction ladder: [`residual-dynamics-benefit-20260720.md`](results/residual-dynamics-benefit-20260720.md) and `task_adapted_*_20260720.json`.
- FetchSlide: `fetch_slide_*_20260723.json`.
- Full engineering log and hashes: [`wiki/status.md`](../wiki/status.md), [`docs/paper-plan.md`](paper-plan.md).

All success-rate numbers are paired scene-cluster bootstraps on fresh held-out
contexts; datasets, checkpoints, and statistics carry SHA-256 provenance;
splits are immutable and episode-disjoint.
