# 14. Demonstrations, Imitation, and Offline Robot Learning

Much modern robot learning does not begin with random exploration and a reward.
It begins with demonstrations, logs, or a pretrained policy. This chapter shows
how those settings relate to—and differ from—reinforcement learning.

## 14.1 A transition dataset is an interface

A robot-learning dataset may contain

$$
D=\{(o_t,a_t,r_t,o_{t+1},d_t,m_t)\},
$$

where $m_t$ is optional metadata such as task language, camera calibration,
timestamps, episode ID, or robot embodiment.

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

## 14.2 Behavior cloning

**Behavior cloning** (BC) treats demonstration actions as labels. For a
continuous action and deterministic policy, a simple objective is

$$
\min_\theta\ \mathbb{E}_{(o,a)\sim D}
\left[\|\pi_\theta(o)-a\|_2^2\right].
$$

Plain language: make the policy predict the demonstrator's action for each
recorded observation.

BC is supervised learning, not reinforcement learning. It does not need reward,
next-state values, or online interaction. It is often a strong baseline because
it avoids reward design and difficult exploration.

### Why mean-squared error can average incompatible actions

Suppose a demonstrator passes an obstacle equally often on the left and right.
At the decision point the action distribution has two modes. A deterministic
mean-squared-error policy may average them and drive straight toward the
obstacle.

Responses include:

- provide more context so the modes become distinguishable;
- predict a multimodal distribution;
- use a mixture model or discrete latent choice;
- predict an action sequence/chunk; or
- use a generative policy such as diffusion.

## 14.3 Covariate shift and compounding errors

Training observations come from the demonstrator distribution
$d_{expert}(o)$. Deployment observations come from the learned policy
$d_{\pi}(o)$. A small prediction error changes the robot state; the next
observation may never occur in demonstrations; another error follows.

This is **covariate shift**. In sequential control, errors can compound over
time.

The Dataset Aggregation (DAgger) idea alternates:

1. run the current learner;
2. ask an expert for the correct action on states the learner visits;
3. aggregate those labeled states into the dataset; and
4. retrain.

The primary [DAgger paper](https://arxiv.org/abs/1011.0686) analyzes this
interactive reduction. On hardware, expert intervention and safe rollout
collection must be engineered; a human may not label a 1 kHz recovery action.

## 14.4 Action chunking

Instead of predicting one action, an **action-chunk** policy predicts a short
sequence:

$$
\pi(o_t)\rightarrow(a_t,a_{t+1},\ldots,a_{t+H-1}).
$$

Benefits can include temporal consistency and fewer high-level inference calls.
But open-loop execution for the entire chunk delays correction. Practical
systems often use receding horizon: predict a chunk, execute a small prefix,
observe again, and replan.

The
[ACT paper](https://arxiv.org/abs/2304.13705) uses action chunking with a
transformer-style conditional variational autoencoder for low-cost bimanual
manipulation. Its demonstration-based success should not be generalized to
fast balance loops without latency and disturbance tests.

## 14.5 Diffusion Policy

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

The primary
[Diffusion Policy paper](https://arxiv.org/abs/2303.04137) reports evaluation
across 12 tasks in four manipulation benchmarks and public code/data. Those
results support the method in that protocol; they do not imply diffusion is
required for all robots.

## 14.6 Offline RL: improvement without new interaction

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

## 14.7 Conservative Q-Learning

Conservative Q-Learning (CQL) adds pressure for values of actions outside the
dataset to be lower than values of observed actions. One conceptual form is

$$
\min_Q\ 
\underbrace{L_{Bellman}(Q)}_{\text{fit transitions}}
+\alpha\left(
\underbrace{\mathbb{E}_{s,a\sim\mu}[Q(s,a)]}_{\text{candidate actions}}
-\underbrace{\mathbb{E}_{s,a\sim D}[Q(s,a)]}_{\text{dataset actions}}
\right).
$$

$\mu$ is an action proposal distribution. The added term discourages the
critic from assigning unjustified high value broadly.

The [CQL paper](https://arxiv.org/abs/2006.04779) provides theoretical and
benchmark evidence for conservative value estimation. Conservative does not
mean physically safe; it describes value estimation relative to data support.

## 14.8 Implicit Q-Learning

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

Policy extraction weights demonstrated actions approximately by

$$
w(s,a)=\exp(\beta(Q(s,a)-V(s))),
$$

usually with clipping. Better-than-baseline dataset actions receive more
weight.

The [IQL paper](https://arxiv.org/abs/2110.06169) reports strong D4RL results
and online fine-tuning. A beginner should retain the design principle: improve
toward the best supported data without trusting arbitrary unseen actions.

## 14.9 Dataset coverage is the real boundary

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

## 14.10 D4RL and benchmark literacy

[D4RL](https://arxiv.org/abs/2004.07219) introduced standardized offline-RL
datasets with behavior mixtures and evaluation protocols. A normalized score
commonly has the form

$$
100\frac{J_\pi-J_{random}}{J_{expert}-J_{random}}.
$$

This makes scores more comparable within a benchmark, but the reference
policies, environment version, termination handling, and dataset composition
remain part of the result. A score of 100 is not “100% physically safe” or
“solved for every robot.”

## 14.11 Open robot-data ecosystems

[Hugging Face LeRobot](https://github.com/huggingface/lerobot) provides an
open-source ecosystem for robot datasets, policies, and real-hardware tools.
Its value for study is the full data-to-policy workflow and common dataset
format—not an assumption that every supported policy is RL.

Generalist manipulation projects such as
[Octo](https://arxiv.org/abs/2405.12213) and
[OpenVLA](https://arxiv.org/abs/2406.09246) train on large collections of robot
demonstrations. They belong primarily to pretrained imitation/foundation-policy
research. Chapter 16 studies their architecture and safety boundary.

## 14.12 Choosing among BC, offline RL, and online RL

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

## 14.13 A practical dataset card

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

## 14.14 Exercises

1. Give an example where squared-error BC averages two valid actions into an
   invalid one.
2. Explain covariate shift using a robot that drifts 2 cm left per step.
3. Why can action chunking help smooth behavior, and when can it hurt reaction
   time?
4. Distinguish ordinary off-policy online RL from offline RL.
5. What kind of optimism do CQL and IQL try to avoid?
6. Design train/validation/test splits for three tables, ten objects, and two
   human demonstrators. Which generalization question does each split answer?
7. A dataset contains only successful trials. What information about failure
   recovery is missing?
8. Write a dataset card for a Microduck playback log. Which additional fields
   are needed before it could support offline learning?

Continue with [modern robot locomotion, adaptation, and sim-to-real research](15_modern_robot_locomotion_and_adaptation.md).
