# 14. Demonstrations, Imitation, and Offline Robot Learning

Much modern robot learning does not begin with random exploration and a reward.
It begins with demonstrations, logs, or a pretrained policy. This chapter shows
how those settings relate to—and differ from—reinforcement learning.

The family has several distinct questions:

| Setting | Data | Learns from | Main danger |
| --- | --- | --- | --- |
| supervised imitation | $(o,a)$ demonstrations | action labels | covariate shift/multimodality |
| inverse reward learning | expert trajectories | inferred objective | reward ambiguity |
| offline reinforcement learning | fixed $(s,a,r,s')$ data | reward and bootstrapping | out-of-support value error |
| offline-to-online | prior data plus new interaction | both | unsafe/unstable distribution transition |
| sequence policy pretraining | multi-task trajectory corpora | conditional prediction | data semantics and deployment mismatch |

Historically, behavior cloning imported supervised pattern recognition into
control; inverse reinforcement learning asked which reward could explain an
expert; Dataset Aggregation (DAgger) addressed learner-induced state shift;
adversarial imitation matched occupancy distributions; offline reinforcement
learning (RL) added pessimism or in-data constraints to value learning; and
modern generative/transformer policies model multimodal action sequences. These
are complementary tools, not a single ladder on which the newest name always
wins.

## 14.1 A transition dataset is an interface

A robot-learning dataset may contain

```math
D=\{(o_t,a_t,r_t,o_{t+1},d_t,m_t)\},
```

where $m_t$ is optional metadata such as task language, camera calibration,
timestamps, episode identifier (ID), or robot embodiment.

Before selecting an algorithm, audit:

- exact observation fields, units, frames, and timestamps;
- action semantics: torque, target, delta, end-effector pose, or action chunk;
- who/what generated each action;
- whether failed episodes are present;
- whether reward is measured, inferred, sparse, or absent;
- termination versus time-limit truncation;
- task and environment coverage;
- sensor/action rate and missing samples; and
- train/validation/test split by episode, scene, object, and operator.

A large dataset with inconsistent action semantics can be less useful than a
small coherent one.

### 14.1.1 Time alignment creates the supervised label

Suppose an image was exposed at $t^I$, joint state sampled at $t^q$, the
teleoperator chose a command using information available near $t^H$, and the
actuator applied it at $t^a$. Writing them in one row does not make them
simultaneous. The learning example is really

```math
(o(t^I,t^q,\ldots),a(t^a),\Delta t,\text{ages}).
```

If image latency is 100 ms and control runs at 20 hertz, a naive nearest-row
join can pair an image with an action two control steps into its future. The
policy then appears accurate offline but cannot reproduce that acausal mapping
online.

Use source and receive timestamps, document clock synchronization, and align by
the information the demonstrator actually had. Preserve raw timing so a later
learner can test alternative alignment. A dataset created only after resampling
to a neat fixed grid may erase the evidence of delays and drops.

Action semantics require the full control path. For a target position log,

```math
a_t=q^{target}_t
```

differs from a delta action

```math
a_t=q^{target}_t-q_t,
```

and both differ from the policy-normalized offset
$a_t=(q^{target}_t-q^{HOME})/s$. A learner trained on one and deployed as
another can remain numerically finite while commanding the wrong magnitude.

### 14.1.2 Split by causal unit, not shuffled frames

Neighboring frames from one trajectory share scene, object, operator, and
history. Random frame splitting puts nearly identical observations in train and
test, producing leakage. Split complete episodes first. To test stronger
generalization, hold out entire objects, scenes, operators, robot revisions, or
collection days. State the unit because each split answers a different
question.

Dataset size has several meanings:

```text
frames/transitions, control hours, episodes, tasks,
unique scenes/objects/operators, robots/embodiments
```

Millions of adjacent frames may contain fewer independent decisions than
thousands of diverse episodes. Report all relevant axes.

## 14.2 Behavior cloning

**Behavior cloning** (BC) treats demonstration actions as labels. For a
continuous action and deterministic policy, a simple objective is

```math
\min_\theta\ \mathbb{E}_{(o,a)\sim D}
\left[\|\pi_\theta(o)-a\|_2^2\right].
```

Plain language: make the policy predict the demonstrator's action for each
recorded observation.

The more general maximum-likelihood objective is

```math
\min_\theta
-\mathbb{E}_{(o,a)\sim D}[\log\pi_\theta(a\mid o)].
```

If $\pi_\theta$ is a Gaussian with fixed isotropic variance $\sigma^2I$ and
mean $\mu_\theta(o)$, then

```math
-\log\pi_\theta(a\mid o)
=\frac{1}{2\sigma^2}\lVert a-\mu_\theta(o)\rVert_2^2+C.
```

Thus mean-squared error is not arbitrary: it is maximum likelihood under a
single Gaussian noise model. Learning per-action variance, a mixture, discrete
codes, or diffusion changes the assumed conditional distribution.

Action normalization matters. If one gripper dimension spans 0–1 while an arm
joint spans only 0.02 radians, raw squared error gives the gripper far more
numerical weight. Standardize by a documented training-set statistic or choose
physical loss weights:

```math
L_{BC}=\frac{1}{d_a}\sum_j
\left(\frac{a_j-\hat a_j}{s_j}\right)^2.
```

Do not compute normalization statistics from test episodes. Save them with the
policy, just as Microduck saves observation normalization.

The core training loop maps directly to the equation:

```python
for observation, expert_action in loader:
    predicted_action = policy(observation)
    normalized_error = (predicted_action - expert_action) / action_scale
    loss = normalized_error.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Training loss measures prediction on demonstrated states. Closed-loop rollout
measures whether those predictions keep the robot near such states; both are
needed.

BC is supervised learning, not reinforcement learning. It does not need reward,
next-state values, or online interaction. It is often a strong baseline because
it avoids reward design and difficult exploration.

### Why mean-squared error can average incompatible actions

Suppose a demonstrator passes an obstacle equally often on the left and right.
At the decision point the action distribution has two modes. A deterministic
mean-squared-error policy may average them and drive straight toward the
obstacle.

For squared loss, the optimal deterministic predictor is the conditional mean:

```math
f^*(o)=\mathbb{E}[a\mid o].
```

To see why, write $a=\mathbb E[a\mid o]+\epsilon$ with
$\mathbb E[\epsilon\mid o]=0$; expanding
$\mathbb E[\lVert a-f(o)\rVert^2\mid o]$ leaves a constant noise term plus
$\lVert\mathbb E[a\mid o]-f(o)\rVert^2$. For equally likely scalar actions
$-1$ and $+1$, the optimum is zero even if zero never appeared in data.

Responses include:

- provide more context so the modes become distinguishable;
- predict a multimodal distribution;
- use a mixture model or discrete latent choice;
- predict an action sequence/chunk; or
- use a generative policy such as diffusion.

A mixture-density policy represents

```math
\pi(a\mid o)=\sum_{k=1}^{K}\rho_k(o)
\mathcal N(a;\mu_k(o),\Sigma_k(o)).
```

It can sample a coherent left or right mode, but component collapse and
unstable switching remain possible. A latent choice should persist long enough
to finish the maneuver; action chunks or a recurrent/skill-level latent can
provide that temporal consistency.

## 14.3 Covariate shift and compounding errors

Training observations come from the demonstrator distribution
$d_{expert}(o)$. Deployment observations come from the learned policy
$d_{\pi}(o)$. A small prediction error changes the robot state; the next
observation may never occur in demonstrations; another error follows.

This is **covariate shift**. In sequential control, errors can compound over
time.

In a simple worst-case argument, let the policy have error probability
$\epsilon$ on expert-distribution states over horizon $T$. By step $t$, the
chance that at least one earlier mistake occurred is at most about $t\epsilon$
by a union bound. Summing exposure to off-expert states across the horizon gives
order

```math
\sum_{t=1}^{T}t\epsilon
=\frac{T(T+1)}{2}\epsilon
=O(T^2\epsilon).
```

This is not a precise prediction for every robot; it explains why a small
one-step validation error can coexist with poor long-horizon control.

The Dataset Aggregation (DAgger) idea alternates:

1. run the current learner;
2. ask an expert for the correct action on states the learner visits;
3. aggregate those labeled states into the dataset; and
4. retrain.

The primary [DAgger paper](https://arxiv.org/abs/1011.0686) analyzes this
interactive reduction. On hardware, expert intervention and safe rollout
collection must be engineered; a human may not label a 1 kHz recovery action.

Because the aggregated dataset includes learner-visited states, the analysis
can reduce the horizon scaling toward $O(T\epsilon)$ under its assumptions. The
price is interaction and expert labels exactly where the learner may be unsafe.
Practical variants use intervention: a human or safety controller takes over,
and the pre-intervention/recovery segment becomes corrective data. This changes
the data distribution and must be logged; intervention-filtered datasets are
not ordinary expert demonstrations.

Run the dependency-free illustration:

```bash
python examples/behavior_cloning_shift.py
```

It shows both an invalid conditional mean for two action modes and quadratic
growth of a simple uncompensated drift cost. The toy model is not a robot
benchmark; it makes the failure mechanisms inspectable.

## 14.4 Inverse rewards and occupancy matching

Sometimes actions are available but the transferable object of interest is the
objective. **Inverse reinforcement learning** asks for a reward under which the
expert appears (near-)optimal. The problem is inherently ambiguous: many
rewards induce the same policy, including positive rescalings and some shaping
transformations. A recovered reward is an explanatory model under assumptions,
not the expert's uniquely true intent.

Maximum-entropy inverse learning assigns expert-like trajectories probability
roughly

```math
p_\theta(\tau)\mathrel{\propto}
\exp\left(\sum_t r_\theta(s_t,a_t)\right),
```

balancing reward fit with distributional uncertainty. It requires dynamics or
sampling machinery and careful feature/identifiability assumptions.

[Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/abs/1606.03476)
instead trains a discriminator to distinguish expert from policy occupancy and
updates the policy to confuse it. One convention is

```math
\min_\pi\max_D
\mathbb E_{(s,a)\sim\rho_E}[\log D(s,a)]
+\mathbb E_{(s,a)\sim\rho_\pi}[\log(1-D(s,a))]
-\lambda\mathcal H(\pi),
```

where $\rho_E$ and $\rho_\pi$ are discounted state–action occupancy measures
and $\mathcal H$ encourages policy entropy. Matching occupancy addresses more
than one-step action regression, but GAIL normally needs fresh policy rollouts
in a simulator/environment and can inherit adversarial-training instability.
It is imitation from demonstrations, not necessarily *offline* learning.

Use inverse/adversarial methods when reward discovery or distribution matching
is genuinely required. For a well-covered task with actions, BC is a simpler
baseline; for a real robot without safe interaction, a rollout-hungry method
may be unsuitable regardless of benchmark performance.

## 14.5 Action chunking

Instead of predicting one action, an **action-chunk** policy predicts a short
sequence:

```math
\pi(o_t)\rightarrow(a_t,a_{t+1},\ldots,a_{t+H-1}).
```

Benefits can include temporal consistency and fewer high-level inference calls.
But open-loop execution for the entire chunk delays correction. Practical
systems often use receding horizon: predict a chunk, execute a small prefix,
observe again, and replan.

Let prediction horizon be $H_p$, executed prefix $H_e$, and control period
$\Delta t$. Maximum open-loop time is $H_e\Delta t$. Larger $H_p$ gives the
model temporal structure, while smaller $H_e$ preserves feedback. These are
separate knobs; a policy can predict 100 steps but execute only one or several.

When a new chunk is predicted every step, several earlier chunks contain a
proposal for current action $a_t$. Temporal ensembling averages them with
recency weights:

```math
\bar a_t=
\frac{\sum_{i=0}^{K-1}w_i\hat a^{(t-i)}_{i}}
{\sum_{i=0}^{K-1}w_i},
\qquad w_i=\exp(-ki).
```

This can smooth inconsistent predictions but adds a stateful filter and delay.
Training, evaluation, and runtime must use the same rule. At a contact
transition, averaging an old “continue closing” proposal with a new “stop”
proposal may be unsafe, so inspect chunk age and discontinuities rather than
assuming smooth is always better.

The
[Action Chunking with Transformers (ACT) paper](https://arxiv.org/abs/2304.13705)
uses action chunking with a transformer-style conditional variational
autoencoder for low-cost bimanual manipulation. Its demonstration-based
success should not be generalized to fast balance loops without latency and
disturbance tests.

A conditional variational autoencoder (CVAE) uses an encoder distribution
$q_\phi(z\mid o,a_{t:t+H})$, a decoder
$p_\theta(a_{t:t+H}\mid o,z)$, and a prior $p(z)$. A common training objective
minimizes reconstruction plus regularization:

```math
L_{CVAE}=
-\mathbb E_{z\sim q_\phi}
[\log p_\theta(a_{t:t+H}\mid o,z)]
+\beta D_{KL}(q_\phi(z\mid o,a)\,\Vert\,p(z)).
```

The Kullback–Leibler (KL) term makes latent samples usable from the prior at
inference. Too much regularization can make the decoder ignore $z$; too little
can leave a latent distribution that does not match inference. ACT's exact
architecture and temporal ensembling should be read in its official code, not
reduced to “transformers solve manipulation.”

## 14.6 Diffusion Policy

A diffusion model learns to turn noise into a structured sample through
iterative denoising. Diffusion Policy conditions that process on robot
observations to generate action sequences.

Conceptually:

```text
random action sequence
       |
       v
denoise using image/state condition
       |
       v
more coherent action sequence
       |
       v
execute first action(s), observe, repeat
```

Why this can help:

- several valid action modes can be represented rather than averaged;
- an action horizon captures temporal structure;
- image-conditioned manipulation has genuinely multimodal choices.

Costs include several denoising evaluations, runtime latency, sensitivity to
data coverage, and no automatic recovery outside demonstrations.

Let $x_0$ denote a demonstrated action chunk. A forward diffusion process adds
Gaussian noise at level $k$:

```math
x_k=\sqrt{\bar\alpha_k}x_0
+\sqrt{1-\bar\alpha_k}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
```

A conditional network learns to predict the injected noise from noisy action,
noise level, and observation context $c$:

```math
L_{diff}=\mathbb E_{x_0,k,\epsilon}
\left[\lVert\epsilon-epsilon_\theta(x_k,k,c)\rVert_2^2\right].
```

At inference, begin from Gaussian $x_K$ and apply a scheduler's reverse steps
until a structured $x_0$ remains. The scheduler, number of steps, chunk horizon,
normalization, and executed prefix are part of the policy—not interchangeable
implementation details.

Why can diffusion represent two routes? Training does not force one
deterministic output to be the conditional mean; different initial noise can
denoise toward different high-density action sequences. But mode diversity is
useful only when samples remain temporally coherent and safe. Evaluate repeated
samples from the *same* observation and examine mode frequency, collision, and
latency.

The primary
[Diffusion Policy paper](https://arxiv.org/abs/2303.04137) reports evaluation
across 12 tasks in four manipulation benchmarks and public code/data. Those
results support the method in that protocol; they do not imply diffusion is
required for all robots.

Important alternatives illustrate the design space:

- [Vector-Quantized Behavior Transformer (VQ-BeT)](https://arxiv.org/abs/2403.03181)
  learns discrete latent action codes and a transformer, targeting multimodality
  with fewer iterative inference steps;
- [3D Diffusion Policy (DP3)](https://arxiv.org/abs/2403.03954) conditions a
  diffusion policy on compact point-cloud features for spatial generalization;
- [BAKU](https://arxiv.org/abs/2406.07539) studies modular observation trunks,
  transformer fusion, chunks, and several action heads for multi-task policies;
  and
- deterministic/recurrent BC remains the latency and implementation baseline.

These papers report different tasks, data counts, robots, rates, and protocols.
Their headline success rates are not a common leaderboard. Reproduce a simple
BC/chunk baseline on your dataset before attributing gains to a generative head.

## 14.7 Offline reinforcement learning: improvement without new interaction

Offline RL uses a fixed dataset but has rewards and sequential transitions. It
tries to find a policy better than the data-collection behavior while avoiding
unsupported actions.

This creates a tension:

```text
stay close to data -> reliable estimates but limited improvement
move beyond data   -> possible improvement but uncertain values
```

Ordinary off-policy algorithms can overestimate an unseen action, then train
the actor to select it. Because no online rollout corrects the mistake, the
error can reinforce itself.

Let dataset behavior be $\beta(a\mid s)$ and candidate policy be
$\pi(a\mid s)$. A standard critic target samples/evaluates the next policy
action:

```math
y=r+\gamma Q_{target}(s',a'),
\qquad a'\sim\pi(\cdot\mid s').
```

If $\beta(a'\mid s')\approx0$, the dataset provides little evidence for that
value. Function approximation may extrapolate a high number; the actor then
prefers it; bootstrapping propagates it backward. This feedback loop is the
offline form of extrapolation error.

One coverage diagnostic is a state–action occupancy ratio

```math
C=\sup_{s,a}\frac{d_\pi(s,a)}{d_\beta(s,a)}.
```

If the denominator is zero where the learned policy goes, $C$ is unbounded and
no finite-data guarantee based on behavior coverage can help. In continuous
high-dimensional robotics, estimating this ratio is itself hard, so practical
algorithms use proximity, uncertainty, pessimism, or in-sample backups as
proxies.

Offline data also cannot identify counterfactual dynamics from one action at a
state. Seeing an expert close a gripper does not reveal what would have happened
under a high-speed sideways motion. Generalization comes from model inductive
bias and nearby coverage, not information magically added by an offline
objective.

## 14.8 Conservative Q-Learning

Conservative Q-Learning (CQL) adds pressure for values of actions outside the
dataset to be lower than values of observed actions. One conceptual form is

```math
\min_Q\ 
\underbrace{L_{Bellman}(Q)}_{\text{fit transitions}}
+\alpha\left(
\underbrace{\mathbb{E}_{s,a\sim\mu}[Q(s,a)]}_{\text{candidate actions}}
-\underbrace{\mathbb{E}_{s,a\sim D}[Q(s,a)]}_{\text{dataset actions}}
\right).
```

$\mu$ is an action proposal distribution. The added term discourages the
critic from assigning unjustified high value broadly.

The [CQL paper](https://arxiv.org/abs/2006.04779) provides theoretical and
benchmark evidence for conservative value estimation. Conservative does not
mean physically safe; it describes value estimation relative to data support.

For a finite action set, a common conservative gap resembles

```math
L_{CQL}(Q)=
\alpha\mathbb E_{s\sim D}
\left[
\log\sum_a\exp Q(s,a)
-\mathbb E_{a\sim D(\cdot\mid s)}Q(s,a)
\right].
```

The log-sum-exp is a smooth maximum over candidate actions. Minimization pushes
down broadly high values while the dataset-action term pushes observed actions
up relative to them. Continuous-action implementations approximate the first
term using sampled actions and importance corrections; reading those sampling
details is essential for mapping paper to code.

Pseudocode:

```python
bellman = mse(q_data, reward + gamma * target_q_next)
q_candidates = q(states_repeated, sampled_candidate_actions)
conservative_gap = logmeanexp(q_candidates) - q_data.mean()
critic_loss = bellman + alpha * conservative_gap
```

Large $\alpha$ can become too pessimistic and prevent improvement; small
$\alpha$ may leave unsupported optimism. Some variants tune a Lagrange
multiplier toward a target gap. Treat it as a dataset/task-sensitive control,
not a universal safety coefficient.

## 14.9 Implicit Q-Learning

Implicit Q-Learning (IQL) avoids evaluating the Q-function at policy actions
outside the dataset during its main value-learning stage.

Its key stages are:

1. fit a state value using **expectile regression** toward the upper portion of
   dataset action values;
2. fit Q with a Bellman target using that state value; and
3. extract a policy with advantage-weighted behavior cloning.

An expectile parameter $\tau>0.5$ weights positive residuals more heavily,
making $V(s)$ reflect better actions present in the dataset without taking an
explicit max over unseen actions.

For residual $u=Q(s,a)-V(s)$, asymmetric squared loss is

```math
L_V=\mathbb E_{(s,a)\sim D}
\left[|\tau-\mathbf 1[u<0]|u^2\right].
```

If $\tau=0.7$, positive residuals ($Q>V$) receive weight 0.7 while negative
residuals receive 0.3, pulling $V$ toward the upper part of values among
*dataset actions*. It is an expectile, not a percentile/quantile: squared
residual magnitude matters.

The critic then uses

```math
L_Q=\mathbb E_D
\left[
Q(s,a)-\left(r+\gamma(1-d)V(s')\right)
\right]^2.
```

Policy extraction weights demonstrated actions approximately by

```math
w(s,a)=\exp(\beta(Q(s,a)-V(s))),
```

usually with clipping. Better-than-baseline dataset actions receive more
weight.

The actor loss maps directly to weighted likelihood:

```python
advantage = q_data.detach() - value.detach()
weight = exp(beta * advantage).clamp(max=max_weight)
actor_loss = -(weight * policy.log_prob(dataset_action)).mean()
```

`detach` expresses the algorithmic intent that this actor update not change the
critic through the weight. Clipping prevents a few estimated advantages from
dominating. Increasing $\beta$ makes extraction greedier and more sensitive to
critic error.

The [IQL paper](https://arxiv.org/abs/2110.06169) reports strong Datasets for
Deep Data-Driven Reinforcement Learning (D4RL) results and online fine-tuning.
A beginner should retain the design principle: improve
toward the best supported data without trusting arbitrary unseen actions.

The [official IQL implementation](https://github.com/ikostrikov/implicit_q_learning)
is compact enough to trace from `value_net.py`, critic updates, and
advantage-weighted policy code back to these equations.

## 14.10 Other offline and offline-to-online lenses

CQL and IQL are influential, not exhaustive. Several alternatives clarify what
an algorithm assumes.

### Behavior-regularized actor–critic

Twin Delayed Deep Deterministic Policy Gradient (TD3) plus behavior cloning,
usually written TD3+BC, uses an offline critic and an actor objective
schematically like

```math
\max_\theta
\mathbb E_{s\sim D}[\lambda Q(s,\pi_\theta(s))]
-\mathbb E_{(s,a)\sim D}
[\lVert\pi_\theta(s)-a\rVert_2^2].
```

The Q term seeks improvement; cloning keeps actions near data. Its importance
is methodological: a carefully normalized simple baseline can rival elaborate
methods, so algorithm comparisons must control implementation details.

### Return-conditioned sequence modeling

[Decision Transformer](https://arxiv.org/abs/2106.01345) represents a
trajectory as desired return-to-go, states, and actions, then autoregressively
predicts actions:

```text
(return-to-go, state, action, return-to-go, state, action, ...)
```

At step $t$, return-to-go updates as $R_{t+1}=R_t-r_t$ in the undiscounted
form. The model is trained by sequence likelihood rather than an explicit
Bellman backup. It can exploit long context and transformer scaling, but asking
for a return never represented by compatible trajectories does not create the
missing behavior. Return scale, context length, inference caching, action
semantics, and dataset quality remain critical. The
[official code](https://github.com/kzl/decision-transformer) is a useful
reference for the token/interleaving pipeline.

### Offline pretraining followed by controlled interaction

Pure offline conservatism can make a poor starting point for online improvement
if Q-values have an unsuitable scale. [Calibrated Q-Learning
(Cal-QL)](https://arxiv.org/abs/2303.05479) studies conservative but calibrated
pretraining for fine-tuning. [Reinforcement Learning with Prior Data
(RLPD)](https://proceedings.mlr.press/v202/ball23a.html) studies how an online
off-policy learner can mix prior and new data efficiently; its
[official repository](https://github.com/ikostrikov/rlpd) exposes the replay and
update design.

This phase change is risky on a robot:

```text
offline policy determines initial visited states
-> new data distribution enters replay
-> critic scale/uncertainty changes
-> actor may move beyond demonstrator support
```

Use bounded authority, human/local safety intervention, a rollback policy,
separate offline/online replay metrics, and staged interaction budgets. “Uses
offline data” does not make later exploration safe.

### Selection guide

| Method | Main inductive bias | Attractive when | Audit first |
| --- | --- | --- | --- |
| BC/recurrent BC | imitate conditional action | demonstrations strong | rollout shift, multimodality |
| chunk/generative BC | model temporal/multiple modes | action choices are structured | latency, mode safety |
| TD3+BC | improve while cloning | rewards exist; simple baseline desired | Q extrapolation and BC scale |
| CQL | pessimistic values | broad mixed-quality data | conservative penalty/sampling |
| IQL | select good in-data actions | useful actions already in data | expectile/weight sensitivity |
| Decision Transformer | conditional sequence modeling | long context and varied returns | unsupported return commands |
| RLPD/Cal-QL | prior data then interaction | safe online budget exists | transition and safety envelope |

## 14.11 Dataset coverage is the real boundary

Imagine a walking dataset containing only forward motion on high-friction
floor. No offline objective can identify the correct recovery action for an
unseen sideways slip purely from that data unless learned structure generalizes
correctly. The dataset does not contain counterfactual evidence.

Audit coverage along dimensions that matter physically:

- commands and task goals;
- initial poses and failure/recovery states;
- surfaces, payloads, voltage, and temperature;
- sensor noise/dropout and latency;
- successful and unsuccessful behavior;
- humans/operators and camera viewpoints; and
- action saturation and safety boundaries.

Build a coverage cube rather than one histogram. Example rows are command
buckets, columns are surface classes, and layers are reset/failure classes.
Store count, duration, success/failure, action extrema, and sensor validity per
cell. Empty cells are explicit unsupported claims.

For continuous observations, nearest-neighbor distance or learned density can
flag novelty, but neither proves safety. High-dimensional image distances may
reflect lighting rather than control relevance; a state can look familiar while
an unseen action is requested. Combine representation-based diagnostics with
semantic slices and closed-loop held-out evaluation.

Dataset quality is not identical to expert success rate:

- proficient demonstrations provide clean targets;
- varied suboptimal data can reveal alternatives and reward ranking;
- failures identify boundaries and recovery, if safely collected/labeled;
- interventions show where autonomy became unsafe, but censor what would have
  happened afterward; and
- duplicate or temporally dense data may overweight one operator/style.

For Microduck, a viewer log becomes useful offline data only if it records the
raw 61D observation before embedded normalization, 14D action and applied
target, command, reward/termination reconstruction, next observation,
timestamps/delays, randomization parameters, contacts, task/config/artifact
identity, and failure labels. A compressed video plus joint positions is not a
transition dataset.

## 14.12 D4RL and benchmark literacy

[D4RL](https://arxiv.org/abs/2004.07219) introduced standardized offline-RL
datasets with behavior mixtures and evaluation protocols. A normalized score
commonly has the form

```math
100\frac{J_\pi-J_{random}}{J_{expert}-J_{random}}.
```

This makes scores more comparable within a benchmark, but the reference
policies, environment version, termination handling, and dataset composition
remain part of the result. A score of 100 is not “100% physically safe” or
“solved for every robot.”

Normalization is affine. If $J_{expert}-J_{random}$ is small or reference
values change across versions, the normalized scale becomes unstable or
incomparable. Scores above 100 and below 0 are possible. Report raw return,
normalized score, environment/dataset version, evaluation episodes, seed-level
values, and uncertainty.

D4RL deliberately includes expert, medium-quality, replay, and mixed datasets,
plus navigation/manipulation domains. That variety tests different coverage
failures. An algorithm ranking on one dataset type need not carry to another,
and a simulated benchmark result does not establish hardware robustness.

Offline evaluation has a special model-selection problem: training loss and
estimated Q can improve while the actual policy worsens. Benchmarks permit
online simulator rollouts for validation, but a real deployment may not.
Reserve safe validation trials, use behavior/value diagnostics cautiously, and
avoid selecting hyperparameters on the final test environment.

## 14.13 Open robot-data ecosystems

[Hugging Face LeRobot](https://github.com/huggingface/lerobot) provides an
open-source ecosystem for robot datasets, policies, and real-hardware tools.
Its value for study is the full data-to-policy workflow and common dataset
format—not an assumption that every supported policy is RL.

[`robomimic`](https://github.com/ARISE-Initiative/robomimic) is a focused
framework for learning from robot demonstrations, with reproducible datasets,
BC/recurrent/transformer/generative and offline-RL baselines. Its versioned
dataset notes are also a lesson: simulator/binding changes can prevent exact
reproduction even when a dataset name looks unchanged.

[Distributed Robot Interaction Dataset
(DROID)](https://arxiv.org/abs/2403.12945) reported 76,000 demonstration
trajectories (350 hours) across 564 scenes and 84 tasks, collected by 50 people,
and released data, policy-learning code, and hardware documentation. This scale
supports studies of scene diversity; it does not eliminate action/calibration
or task-distribution boundaries.

[Open X-Embodiment](https://arxiv.org/abs/2310.08864) standardized data across
22 robot embodiments and studied Robotics Transformer X (RT-X) cross-robot
models. Cross-embodiment pooling creates a harder interface problem:

```text
different cameras/frames/rates/action spaces/grippers
  -> embodiment-specific decoding and normalization
  -> common semantic/task representation
  -> per-robot safety and evaluation
```

Data volume is valuable only after these meanings are aligned. A universal
tensor field named `action` can still mix Cartesian deltas, absolute poses, and
joint targets.

Generalist manipulation projects such as
[Octo](https://arxiv.org/abs/2405.12213) and
[OpenVLA](https://arxiv.org/abs/2406.09246) train on large collections of robot
demonstrations. They belong primarily to pretrained imitation/foundation-policy
research. Chapter 16 studies their architecture and safety boundary.

As of this book's September 2026 review, no single public dataset, format, or
policy family is a universal standard for locomotion, manipulation, mobile
navigation, and every embodiment. The durable skills are schema inspection,
split design, baseline reproduction, licensing/provenance, and deployment
contract testing. Treat ecosystem version and paper date as part of every
claim.

## 14.14 Choosing among BC, offline RL, and online RL

Start with behavior cloning when:

- demonstrations are strong and cover deployment;
- reward is missing or unreliable;
- a simple, auditable baseline is needed.

Consider offline RL when:

- transitions and meaningful rewards exist;
- the dataset contains a range of behavior quality;
- policy improvement beyond average demonstration behavior matters;
- you can evaluate conservatively before hardware.

Consider online fine-tuning when:

- a safe simulator or controlled hardware protocol exists;
- deployment states differ from logged data;
- reward is reliable; and
- rollback and exploration limits are enforced.

A common progression is

```text
demonstrations -> behavior cloning -> offline RL (optional)
               -> safe simulation/real fine-tuning -> frozen deployment
```

Each arrow needs its own evaluation gate.

## 14.15 A practical dataset card

Record at least:

```yaml
robot: exact hardware revision
task: operational success definition
episodes: total / success / failure
observation_schema: fields, shapes, units, frames
action_schema: meaning, range, rate, delay
sensors: models, calibration, timestamps
collection_policy: human / scripted / learned, versions
environment_distribution: objects, surfaces, lighting, payload
safety_interventions: what was filtered or stopped
splits: episode/scene/object/operator separation
known_gaps: unsupported conditions
license_and_consent: provenance and permitted use
```

Without provenance, “more data” can mean more untraceable error.

Automate dataset invariants before training:

```python
for episode in dataset:
    assert episode.observation.shape[1:] == observation_schema.shape
    assert episode.action.shape[1:] == action_schema.shape
    assert len(episode.next_observation) == len(episode.action)
    assert timestamps_are_monotonic(episode)
    assert terminal_and_truncation_are_consistent(episode)
    assert finite_or_explicitly_masked(episode)
    assert action_within_recorded_interface(episode)
```

Also compute per-field min/quantiles/max, missingness, sample intervals, action
saturation, episode length, task/operator counts, and near-duplicate hashes.
Review images for privacy/consent obligations and strip secrets or bystanders
according to the dataset license/governance plan. A public robot-data release
is both a machine-learning artifact and a record of real environments/people.

Version raw and derived data separately. A derived reward, resized image,
time-aligned action, or filtered episode is code-dependent transformation, so
store source dataset digest, transformation commit/config, and output digest.

## 14.16 Exercises

1. Give an example where squared-error BC averages two valid actions into an
   invalid one. Derive the optimal scalar prediction for equally likely actions
   `-1` and `+1`.
2. An image arrives 100 ms after exposure in a 20 hertz policy. How many nominal
   control periods old is it? Why can joining it to the action in its receive
   row create future-information leakage?
3. Explain covariate shift using a robot that drifts 2 cm left per step. What is
   drift after 25 uncompensated steps, and why is the real failure potentially
   worse than this linear arithmetic?
4. Explain the $O(T^2\epsilon)$ worst-case BC intuition and what additional
   resource DAgger uses to improve state-distribution coverage.
5. Is GAIL offline because it starts from expert demonstrations? Explain using
   its occupancy objective.
6. For a 20 hertz chunk policy that executes 6 predicted actions before
   replanning, calculate open-loop time. Give one task where this is acceptable
   and one where it is dangerous.
7. In the diffusion equation, what does $\bar\alpha_k\rightarrow0$ do to
   $x_k$? Distinguish the training noise-prediction pass from inference.
8. Distinguish ordinary off-policy online RL from offline RL. Why does the same
   Bellman target become more dangerous offline?
9. Explain the sign of CQL's dataset-action term. What would happen if both
   candidate and dataset Q-values were pushed down equally without a relative
   term?
10. For IQL with $\tau=0.8$, what weights apply to residuals $u>0$ and $u<0$?
    Why is this an expectile rather than a quantile?
11. A Decision Transformer dataset's best return is 50. Does conditioning on
    return 100 guarantee a twice-as-good policy? Why not?
12. Baseline and treatment each use 100,000 prior transitions, but treatment
    begins collecting online robot data. List five new safety/statistical facts
    the comparison must report.
13. Design train/validation/test splits for three tables, ten objects, and two
    human demonstrators. Which generalization question does each split answer?
14. A dataset contains only successful trials. What information about failure
    recovery and interventions is missing?
15. Build a coverage matrix for Microduck commands, surfaces, and reset classes.
    Which empty cell would invalidate a claim of general slip recovery?
16. Write a dataset card for a Microduck playback log. Which additional fields
    are needed before it could support offline learning?
17. Choose among deterministic BC, generative chunk BC, IQL, and offline-to-
    online RLPD for: (a) excellent demonstrations with no reward, (b) mixed
    reward-labeled logs, and (c) sparse demonstrations plus a safe simulator.
    Defend baselines and escalation gates.
18. Compare LeRobot, robomimic, DROID, and Open X-Embodiment as study resources.
    Do not rank them with one number; state the distinct question each enables.

Continue with [modern robot locomotion, adaptation, and sim-to-real research](15_modern_robot_locomotion_and_adaptation.md).

## 14.17 Folded solutions

<details>
<summary>Show solutions to Exercises 1–9</summary>

1. Left/right obstacle passing gives actions $-1,+1$. Expected squared error is
   $L(x)=\tfrac12(x+1)^2+\tfrac12(x-1)^2=x^2+1$, so
   $dL/dx=2x=0$ at $x=0$. Zero is the conditional mean and can be invalid even
   though both labels are safe modes. Add disambiguating context or a
   mode-preserving distribution/latent.
2. At 20 hertz, one period is 50 ms, so the image is two periods old. Joining
   by receive row pairs the old visual scene with an action selected from newer
   information. At deployment the policy cannot see that newer scene inside
   the old image, so offline accuracy contains acausal leakage.
3. Simple drift is $25(0.02)=0.50$ m. The real error can be worse because the
   learner enters unseen states, changes contact/viewpoint, and may no longer
   make the same 2 cm error; dynamics can amplify it nonlinearly or cause a fall.
4. Error probability accumulates over earlier steps, giving at most roughly
   $t\epsilon$ probability of prior mistake at step $t$; summing gives
   $\epsilon T(T+1)/2$. DAgger queries an expert on learner-visited states and
   aggregates labels, paying for interactive rollout, annotation, and safety.
5. Usually no. GAIL must sample the current policy occupancy $\rho_\pi$ to train
   its discriminator and policy. Unless those rollouts come from a fixed model
   or special offline reformulation, it interacts with an environment/simulator
   despite starting from expert trajectories.
6. Open-loop time is $6/20=0.30$ s. It may be acceptable for a slow free-space
   arm segment with local low-level protection; it is dangerous for biped
   balance, dynamic contact, or human-proximate motion where state changes
   within tens of milliseconds.
7. As $\bar\alpha_k\to0$, the signal coefficient vanishes and $x_k$ approaches
   Gaussian noise. Training samples a demonstrated chunk, noise level, and
   noise once and learns to predict that noise. Inference starts from noise and
   iterates multiple scheduler/model steps without knowing a demonstrated
   $x_0$.
8. Online off-policy learning reuses replay but can collect corrective data
   under its evolving policy. Offline RL never observes consequences of new
   actions. A target action outside behavior support can receive arbitrary
   extrapolated Q; bootstrapping and actor maximization then reinforce it
   without correction.
9. Minimizing `candidate log-sum-exp - dataset Q` pushes broad candidate values
   down *relative to observed actions*. If every value were lowered equally,
   action ranking/unsupported optimism could remain while Q scale simply
   shifted; the Bellman term and relative contrast establish useful values.

</details>

<details>
<summary>Show solutions to Exercises 10–18</summary>

10. For $u>0$, weight is $|0.8-0|=0.8$; for $u<0$, weight is
    $|0.8-1|=0.2$. Squared magnitude enters the objective, so this is asymmetric
    least squares/expectile regression. Quantile regression uses asymmetric
    absolute (pinball) loss.
11. No. The model learned conditional correlations within dataset support. A
    return token of 100 may be out of distribution and is not an optimizer that
    constructs missing twice-as-good trajectories. It may ignore, extrapolate,
    or fail under the unsupported condition.
12. Report online interaction count/time and schedule, initial/offline policy,
    authority/safety/interventions, replay mixing and update-to-data ratio,
    failures/severity, environment resets/changes, seed design, rollback, and
    which evaluation data remained untouched. Five of these is the minimum;
    all are useful.
13. Hold out a complete table for scene transfer, selected objects across seen
    tables for object transfer, and one operator for style/operator transfer.
    A combined held-out table/object/operator test asks compositional transfer.
    Validation holds separate examples for selection; never shuffle frames from
    the same episode across splits.
14. Missing evidence includes warning states, unsuccessful actions,
    recoverability boundaries, corrective actions, intervention policy,
    censored post-intervention outcomes, and severity. Success-only BC may have
    no label for states caused by its own mistakes.
15. Make rows `{idle, forward, lateral, turn, combined}`, columns
    `{high-friction, low-friction, uneven}`, and layers
    `{nominal, biased pose, push/slip}`. If all low-friction push/slip cells are
    empty, the dataset cannot directly support a general low-friction slip-
    recovery claim even if nominal forward data are abundant.
16. Record robot revision, task/commit/config/checkpoint, raw 61D schema and
    normalizer identity, 14D action/applied target, HOME/scale/order, commands,
    timestamps/delays, next observations, rewards, terminal versus truncation,
    resets/randomization/contacts, episode and failure/intervention labels,
    split units, transforms, hashes, license, and known gaps. Video alone lacks
    the causal transition fields.
17. (a) Start deterministic/recurrent BC; escalate to generative chunks only if
    measured multimodality/temporal inconsistency warrants latency. (b) Start BC
    and a simple reward-aware baseline, then IQL if mixed quality contains
    better supported actions and offline evaluation exists. (c) Pretrain BC or
    value/policy from demonstrations, validate in simulator, then consider
    RLPD with bounded online budget and safety. In all cases the escalation gate
    is a named baseline failure, not fashion.
18. LeRobot teaches an integrated open dataset/policy/hardware workflow;
    robomimic supports controlled learning-from-demonstration/offline baselines
    and reproducibility studies; DROID supports in-the-wild scene/diversity and
    standardized collection questions on one broad platform family; Open
    X-Embodiment supports cross-robot semantic/action alignment and transfer.
    Dataset/task/robot/protocol differences make one scalar ranking invalid.

</details>
