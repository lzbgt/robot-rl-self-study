# 4. The Reinforcement-Learning Algorithm Map

Algorithm names can make reinforcement learning (RL) feel like a collection of
unrelated inventions. Most methods can be located along a few design axes.
Learning those axes is more durable than memorizing a leaderboard.

### A dated lineage, not a leaderboard

The major families answer recurring problems:

| Period | Representative origin or milestone | Problem it made explicit |
| --- | --- | --- |
| 1950s | Bellman dynamic programming | recursive planning with a known model |
| 1980s–1990s | actor-critic, Q-learning, REINFORCE | learn values or policies from sampled experience |
| 2000s | natural policy gradient | account for how parameters alter a probability distribution |
| 2013–2018 | Deep Q-Network, Deep Deterministic Policy Gradient, Trust Region Policy Optimization, Proximal Policy Optimization, Twin Delayed DDPG, Soft Actor-Critic | stabilize scalable neural discrete/continuous control |
| 2019–2023 | conservative/implicit offline RL, Dreamer, TD-MPC | learn from fixed data or learned latent dynamics |
| 2023–2026 | action generators, generalist policies, massive off-policy robot training | reuse diverse data and scale representations/control |

Here **Deep Deterministic Policy Gradient (DDPG)** is the actor-critic precursor
from which Twin Delayed DDPG addresses value-error failures; **Trust Region
Policy Optimization (TRPO)** is the constrained-update precursor that motivates
Proximal Policy Optimization's simpler surrogate. The dates locate ideas, not
declare old methods obsolete. Proximal Policy Optimization (PPO) remains a
strong robot baseline because an
algorithm's value depends on the data and system around it.

## 4.1 First ask what data interaction is allowed

### Online RL

In **online RL**, the agent collects new experience while learning. The data
distribution changes as the policy changes.

- Simulation makes online interaction cheap and safe enough for millions or
  billions of transitions.
- Physical online learning may consume robot time, wear actuators, or discover
  unsafe actions.

### Offline RL

In **offline RL**, learning uses a fixed logged dataset and cannot request new
transitions. It is attractive when robot data already exists or new exploration
is dangerous. It is difficult because value estimates for actions absent from
the dataset can be arbitrarily wrong.

### The behavior distribution is part of the mathematics

Let $\mu(a\mid s)$ be the **behavior policy** that collected data and
$\pi(a\mid s)$ the **target policy** being evaluated or improved. For an
expectation over actions at a covered state, importance sampling gives

```math
\mathbb{E}_{a\sim\pi}[f(a)]
=\mathbb{E}_{a\sim\mu}
\left[\frac{\pi(a\mid s)}{\mu(a\mid s)}f(a)\right].
```

Derivation for a discrete action set is substitution:

```math
\sum_a\mu(a\mid s)
\frac{\pi(a\mid s)}{\mu(a\mid s)}f(a)
=\sum_a\pi(a\mid s)f(a).
```

The ratio is valid only where $\mu(a\mid s)>0$ whenever $\pi(a\mid s)>0$.
Large ratios create high variance. Full trajectory ratios multiply across
time and can become unusably large or small. Modern off-policy algorithms
therefore combine bootstrapping, clipping, conservative objectives, learned
critics, and coverage assumptions rather than naively reweighting long robot
trajectories.

### Imitation learning

**Imitation learning** learns from demonstrations, often by treating expert
actions as supervised labels. It may not use a reward or Bellman backup at all.
Modern robot-policy research mixes imitation, offline RL, online fine-tuning,
and pretrained representations, so “robot learning” is broader than RL.

## 4.2 Model-free versus model-based

A **dynamics model** predicts how state changes:

```math
\hat{s}_{t+1}=f_\psi(s_t,a_t).
```

- **Model-free RL** learns a policy and/or value without using a learned
  transition model for planning.
- **Model-based RL** uses a known or learned model to evaluate or optimize
  future action sequences.

The labels do not mean model-free robot design has no physics knowledge. A
Proximal Policy Optimization (PPO) policy trained in MuJoCo relies heavily on a
simulator model to generate data; the learning algorithm itself simply does
not differentiate through or plan with a learned $f_\psi$ at runtime.

A second distinction is **planning at decision time** versus **amortized
control**. Model Predictive Control (MPC) optimizes an action sequence again at
each state. A feed-forward actor spends compute during training so that a
single network evaluation approximates a good decision later. Temporal
Difference Learning for Model Predictive Control, second generation (TD-MPC2)
uses both: a learned policy proposes actions and a learned model/value refines
a local plan.

Model-based methods can be more sample efficient because one transition helps
learn dynamics reusable by many plans. Their danger is **model bias**: planning
can exploit small prediction errors and imagine impossible success.

## 4.3 Value-based, policy-based, and actor-critic

### Value-based

Value-based methods learn $V$ or $Q$ and derive behavior by choosing a
high-value action. A Deep Q-Network (DQN) is the canonical deep discrete-action
example.

### Policy-based

Policy-gradient methods directly adjust a parameterized policy
$\pi_\theta(a\mid s)$ toward actions associated with higher return. Continuous
actions are natural because the policy can output distribution parameters.

### Actor-critic

An **actor-critic** has both:

- actor: produces actions;
- critic: evaluates states or state-action pairs to improve the actor.

PPO, Soft Actor-Critic (SAC), and Twin Delayed Deep Deterministic Policy
Gradient (TD3) are actor-critic methods, but their data use and objectives
differ substantially.

## 4.4 On-policy versus off-policy

An on-policy method updates from data collected by the current or very recent
policy. PPO is approximately on-policy. Old data becomes stale after a policy
change, so it is normally discarded.

An off-policy method can update a target policy using data from other behavior
policies. SAC, TD3, DQN, Implicit Q-Learning (IQL), and Conservative Q-Learning
(CQL) are off-policy in different settings. A replay buffer improves sample
reuse but makes the learning distribution less like current deployment.

The tradeoff is not simply “off-policy is better because it reuses data”:

| Property | On-policy | Off-policy |
| --- | --- | --- |
| data reuse | low | high |
| implementation/stability | often simpler | often more delicate |
| massively parallel simulation | excellent fit | possible, needs care |
| scarce real interaction | expensive | often preferable |
| distribution correction | limited by fresh data | central challenge |

## 4.5 Stochastic versus deterministic policies

A **stochastic policy** describes a distribution over actions. Exploration can
come from sampling that distribution. PPO and SAC train stochastic policies.

A **deterministic policy** maps a state to one action. TD3 learns a
deterministic actor and adds noise during data collection. A deployed
stochastic-policy actor may also use its mean deterministically.

Stochastic does not mean careless or random behavior. The distribution is
conditioned on the observation, and its spread is learned or configured.

## 4.6 Core families at a glance

**Temporal Difference Learning for Model Predictive Control, second generation
(TD-MPC2)** is the newer member of the TD-MPC method family. The long form
describes its combination of temporal-difference value learning and
model-predictive planning; the “2” identifies the later scalable method.

| Method | Action | Data | Learned objects | Characteristic idea | Typical fit |
| --- | --- | --- | --- | --- | --- |
| tabular Q-learning | discrete | online, off-policy | Q-table | greedy Bellman target | small teaching/control problem |
| DQN | discrete | online replay, off-policy | Q-network | replay + target network | games or finite skill choices |
| PPO | discrete or continuous | online, on-policy | actor + value critic | clipped policy-ratio objective | high-throughput simulation |
| TD3 | continuous | online replay, off-policy | deterministic actor + twin critics | clipped double-Q + delayed actor | sample-efficient state control |
| SAC | continuous; discrete variants exist | online replay, off-policy | stochastic actor + critics | maximize reward and entropy | continuous control with costly data |
| IQL/CQL | usually continuous | fixed offline data | values/critics + policy | avoid optimistic unseen actions | logged robot datasets |
| DreamerV3 | varied | online replay, model-based | latent world model + actor/critic | learn through imagined rollouts | pixels/general domains |
| TD-MPC2 | continuous | online replay, model-based | latent model + values/policy | local latent trajectory optimization | sample-efficient continuous control |
| behavior cloning | any represented in data | demonstrations | policy | supervised action prediction | strong expert data |
| diffusion policy | continuous action chunks | demonstrations | conditional diffusion model | model multimodal action sequences | visual manipulation |

This table is a map, not a ranking. Performance depends on implementation,
task, data, compute, observation/action choices, and evaluation.

### A unifying “what is fitted?” view

Many method differences become concrete by locating the regression or
optimization target:

```text
behavior cloning: observation -----------------> demonstrated action
value learning:   state/action ----------------> Bellman return target
policy gradient:  sampled log-probability -----> advantage-weighted objective
world model:      state/history + action ------> next latent/reward/continuation
offline RL:       fixed data -------------------> conservative value + extracted policy
```

This gives a code-reading strategy. Find the batch fields, construct the target,
find the loss, then find which parameters receive gradients. Algorithm names
matter less than these four concrete facts.

## 4.7 Four influential objectives in plain language

### DQN: make Q agree with a bootstrapped target

```math
\min_\theta
\left(r+\gamma\max_{a'}Q_{\bar\theta}(s',a')-Q_\theta(s,a)\right)^2.
```

Move the current action value toward immediate reward plus the target network's
best next value.

### PPO: improve sampled actions without moving probability too far

```math
\max_\theta\ \mathbb{E}
\left[\min(r_tA_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right].
```

Increase probability for better-than-expected sampled actions and decrease it
for worse ones, while clipping the incentive for a large probability-ratio
change. Chapter 5 derives every term.

### From natural gradient to TRPO to PPO

An ordinary parameter step measures distance in weight coordinates. But the
same weight change can barely alter one policy and radically alter another.
The Kullback–Leibler (KL) divergence measures change in action distributions.
Trust Region Policy Optimization poses a local problem of the form

```math
\max_\theta
\mathbb{E}_{(s,a)\sim\pi_{old}}
\left[
\frac{\pi_\theta(a\mid s)}{\pi_{old}(a\mid s)}A^{\pi_{old}}(s,a)
\right]
```

subject to

```math
\mathbb{E}_{s\sim\pi_{old}}
[D_{KL}(\pi_{old}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s))]
\leq\delta.
```

Near the old parameters, the KL curvature is approximated by the Fisher
information matrix $F$. The natural-gradient direction is

```math
\Delta\theta\propto F^{-1}\nabla_\theta J.
```

TRPO approximately solves this constrained problem using conjugate gradients
and a line search. PPO replaces that machinery with first-order clipped or KL-
penalized surrogates that are easier to implement and batch. PPO is therefore
best understood as a practical descendant of trust-region reasoning, not as a
proof that clipping alone bounds every policy change.

### TD3: trust the smaller of two learned critics

```math
y=r+\gamma\min_{i=1,2}Q_{\bar\phi_i}
(s',\pi_{\bar\theta}(s')+\text{clipped noise}).
```

Using the smaller critic reduces overestimation; delayed actor updates let the
critics settle between policy changes. See the primary
[TD3 paper](https://arxiv.org/abs/1802.09477).

### SAC: value both reward and action entropy

```math
J(\pi)=\mathbb{E}\left[\sum_t\gamma^t
\bigl(r_t+\alpha\mathcal{H}(\pi(\cdot\mid s_t))\bigr)\right].
```

Entropy $\mathcal{H}$ measures distributional spread. SAC rewards task success
and, weighted by temperature $\alpha$, retaining multiple plausible actions.
This often improves exploration and robustness. See the primary
[SAC paper](https://arxiv.org/abs/1801.01290).

## 4.8 Exploration is part of the system design

Exploration mechanisms include:

- $\epsilon$-greedy random discrete actions;
- action noise around a deterministic actor;
- sampling a stochastic actor;
- an entropy bonus;
- randomized initial states and goals;
- curricula that expose increasingly difficult regions;
- demonstrations or reverse-curriculum starts that place the agent near rare
  useful states; and
- intrinsic rewards for novelty or prediction error.

Robot exploration must respect hardware. “Let the agent discover it” is not a
safety plan. Most aggressive motor-skill exploration belongs in simulation,
with a separately enforced hardware envelope.

Exploration can also be reasoned about as uncertainty. For a simple bandit,
an upper confidence bound (UCB) action score is

```math
\mathrm{UCB}(a)=\hat Q(a)+c\sqrt{\frac{\log t}{N(a)}},
```

where the second term is large for rarely tried actions. Deep robotics rarely
uses this exact formula at every motor dimension, but the principle survives
in ensembles, uncertainty bonuses, active system identification, goal
sampling, and model-predictive exploration. Random action noise is only one
possible information-acquisition strategy.

## 4.9 The frontier as of 2026 is conditional

No defensible source supports one algorithm as “the state of the art for robot
intelligence.” The evaluated regimes are too different. A dated frontier map
is more useful:

| Regime | Strong current baselines or directions | Main unresolved pressure |
| --- | --- | --- |
| massively parallel proprioceptive locomotion | PPO/Robotic Systems Lab reinforcement learning (RSL-RL) recipes; emerging high-update off-policy SAC/TD3 variants | robustness and fair wall-clock/compute comparison |
| scarce online physical interaction | SAC-style replay, model-based world models, residual/hybrid control | safety, resets, nonstationary hardware |
| fixed logged robot data | behavior cloning, IQL, CQL, diffusion/action-chunk policies | coverage and out-of-distribution action error |
| pixel control with online learning | Dreamer-family latent models, TD-MPC-family planning | contact/model error and runtime compute |
| broad language-conditioned manipulation | transformer/diffusion/flow imitation plus selective RL fine-tuning | data provenance, embodiment transfer, calibrated evaluation |
| safety-constrained control | constrained Markov Decision Process (MDP) objectives, shields, control-barrier filters, MPC | expectation constraints do not equal hard guarantees |

The 2025 FastSAC/FastTD3 work is notable because it challenges the assumption
that on-policy PPO is inherently the fastest choice under massive simulation;
its training-time claim is tied to the reported simulator, update ratio,
hardware, and humanoid tasks. The right response is a matched experiment, not
automatic replacement. Chapters 6, 14–16, and 18 examine each regime using
primary papers and official artifacts.

## 4.10 A practical selection procedure

Ask in this order:

1. **Is the action discrete or continuous?** DQN does not naturally solve a
   14D continuous joint command.
2. **Can you collect fresh interaction cheaply?** If yes, on-policy simulation
   may be simple and effective. If no, consider replay, offline data, or a
   model.
3. **Is the observation low-dimensional state or high-dimensional vision?**
   Pixels increase representation and data demands.
4. **Do you have demonstrations?** A supervised initialization may remove the
   hardest exploration phase.
5. **Is a reliable model available?** Model-predictive control or model-based
   RL may exploit it.
6. **Does the task need memory/adaptation?** Select history, recurrence, or
   latent inference explicitly.
7. **What is the safety boundary?** Constraints and an independent runtime
   supervisor may matter more than algorithm choice.
8. **What baseline must learning beat?** Start with a scripted or classical
   controller when one exists.

## 4.11 Why PPO is sensible for Microduck

Microduck has:

- a 14D continuous action;
- thousands of graphics processing unit (GPU)-parallel simulated robots;
- inexpensive fresh rollouts;
- dense task/recovery signals; and
- a small actor needed for 50 Hz deployment.

These properties fit PPO. One iteration with 4,096 environments and 24 steps
collects 98,304 fresh transitions. Low sample reuse is less painful when
simulation generates this much data quickly.

PPO does not cause walking by itself. Robot model, observation, command
distribution, reward, reset distribution, actuator dynamics, randomization,
and curriculum define the experience from which PPO learns.

## 4.12 Why a different robot may need a different method

Consider three projects:

### A simulated biped locomotion controller

Continuous action, cheap massively parallel experience, 50–100 Hz inference.
PPO is a strong baseline.

### A real arm with 200 successful demonstrations

Physical trial cost is high and demonstrations already specify behavior.
Behavior cloning or a diffusion/action-chunk policy is a sensible first
baseline; offline RL may improve it if reward labels and dataset coverage are
credible.

### A wheeled robot choosing among “follow,” “dock,” and “stop”

The high-level choice is discrete and slow, but each skill has its own
continuous controller. A hierarchical architecture may use a planner or
discrete policy above classical/RL motor skills. One monolithic algorithm is
not required.

## 4.13 Common selection mistakes

- Choosing the newest paper before defining the observation and action.
- Comparing algorithms with different reward, model, environment count, or
  compute budgets.
- Calling behavior cloning “RL” because the policy is a neural network.
- Calling a simulator-trained model “model-based RL” merely because the
  simulator has physics.
- Assuming sample efficiency implies wall-clock efficiency.
- Reporting the best seed instead of a distribution.
- Using a generalist visual model in a hard realtime loop without measuring
  worst-case latency.

## 4.14 Exercises

For each scenario, select a starting algorithm and state what evidence would
change your mind:

1. Cart-pole with left/right discrete actions and unlimited simulation.
2. Microduck velocity tracking in 4,096 parallel environments.
3. A robot arm with a fixed dataset and no permission for online exploration.
4. An image-based manipulation task with several valid grasp trajectories.
5. A rover with a known dynamics model, strict constraints, and a 20 Hz local
   planner.
6. A legged robot whose payload changes during an episode.

Then answer these mechanism questions:

7. A behavior policy selects an action with probability 0.2 and the target
   policy assigns it probability 0.5. Compute its importance ratio. What makes
   a product of 100 such ratios dangerous?
8. Explain why a small Euclidean change in neural-network weights does not
   necessarily imply a small policy change. What quantity do trust-region
   methods measure instead?
9. For the same task, PPO uses 100 million transitions while SAC uses 10
   million but takes twice the wall-clock time and three times the GPU energy.
   Which is “more efficient”? Give a non-misleading report.
10. Locate each system on two axes—model-free/model-based and online/offline:
    PPO in simulation, behavior cloning from logs, Dreamer trained while
    interacting, and MPC with an identified fixed model.
11. A paper reports state-of-the-art mean reward on one simulator. List four
    facts needed before choosing it for a physical biped.

Then make a two-column list: what the learning algorithm decides versus what
the system architecture must decide. Continue with
[PPO from equations to code](05_ppo_from_equations_to_code.md).

## 4.15 Folded solutions

<details>
<summary>Show reference answers to Section 4.14</summary>

1. **Cart-pole:** begin with tabular Q-learning after discretization or DQN for
   a neural baseline. Unlimited simulation and two discrete actions fit value
   learning. A continuous-action variant or poor pixel sample efficiency would
   motivate an actor-critic or representation change.
2. **Microduck:** begin with PPO. Its continuous action and 4,096 cheap parallel
   simulators make fresh on-policy data practical. A matched-budget experiment
   showing better robustness or wall-clock cost from SAC/TD3 would change the
   choice.
3. **Fixed arm dataset:** start with behavior cloning, then compare IQL or CQL
   if rewards and coverage support offline improvement. Permission for safe
   online interaction would open a separate fine-tuning phase.
4. **Multimodal visual manipulation:** start with a diffusion or other
   multimodal imitation policy. Plain squared-error behavior cloning (BC) may
   average incompatible grasps. Additional reward-bearing data could justify
   offline RL.
5. **Known constrained rover:** start with Model Predictive Control (MPC), which
   can use the known model and express constraints. Learn a residual/model only
   if held-out traces reveal systematic error the baseline cannot handle.
6. **Changing payload:** start with a history-conditioned/adaptive locomotion
   actor, using privileged teacher information only during training. If the
   payload is measured directly, explicit estimation plus a robust controller
   may be simpler.

The learning algorithm decides how experience changes a policy, value, or
model. The architecture still decides sensor/calibration ownership, control
rate, action semantics, planner decomposition, hard limits, watchdog, E-stop,
compute placement, data privacy, and fallback. Algorithm selection cannot make
those system decisions disappear.

7. The ratio is $0.5/0.2=2.5$, so this sample receives greater target-policy
   weight. A trajectory product $2.5^{100}$ is enormous; other paths can yield
   products near zero. Such products create extreme variance and numerical
   instability, and support failure occurs if behavior probability is zero.
8. Neural policies are nonlinear and parameterized redundantly; local output
   sensitivity depends on state and current weights. The same weight-space norm
   can cause very different action-distribution changes. Trust-region methods
   use a distributional distance, commonly expected Kullback–Leibler
   divergence, and natural gradient incorporates its local curvature.
9. SAC is five times more transition-efficient; PPO is twice as wall-clock
   efficient and three times as energy/compute efficient under the stated
   measurements. Neither is simply “more efficient.” Report all three axes,
   final held-out performance, hardware, simulator throughput, update count,
   and uncertainty across seeds.
10. PPO in simulation is model-free at the learning/inference level and online
    in data collection. Behavior cloning is offline and does not learn/use a
    dynamics model. Interacting Dreamer is online and model-based because it
    learns a world model. MPC with a fixed identified model is model-based
    control but not necessarily RL; if it performs no learning from a dataset,
    online/offline RL is the wrong label.
11. At minimum: exact robot/task and action/observation contract; data and
    compute budget; baseline implementations and tuning; seeds/trials and
    uncertainty; simulator-to-real evidence; runtime latency/model size; safety
    protocol; and availability/license of code, configs, data, and checkpoints.
    Any four well-explained items expose why one benchmark mean is insufficient.

</details>
