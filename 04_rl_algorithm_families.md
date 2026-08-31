# 4. The Reinforcement-Learning Algorithm Map

Algorithm names can make RL feel like a collection of unrelated inventions.
Most methods can be located along a few design axes. Learning those axes is
more durable than memorizing a leaderboard.

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

The labels do not mean model-free robot design has no physics knowledge. A PPO
policy trained in MuJoCo relies heavily on a simulator model to generate data;
the learning algorithm itself simply does not differentiate through or plan
with a learned $f_\psi$ at runtime.

Model-based methods can be more sample efficient because one transition helps
learn dynamics reusable by many plans. Their danger is **model bias**: planning
can exploit small prediction errors and imagine impossible success.

## 4.3 Value-based, policy-based, and actor-critic

### Value-based

Value-based methods learn $V$ or $Q$ and derive behavior by choosing a
high-value action. DQN is the canonical deep discrete-action example.

### Policy-based

Policy-gradient methods directly adjust a parameterized policy
$\pi_\theta(a\mid s)$ toward actions associated with higher return. Continuous
actions are natural because the policy can output distribution parameters.

### Actor-critic

An **actor-critic** has both:

- actor: produces actions;
- critic: evaluates states or state-action pairs to improve the actor.

PPO, SAC, and TD3 are actor-critic methods, but their data use and objectives
differ substantially.

## 4.4 On-policy versus off-policy

An on-policy method updates from data collected by the current or very recent
policy. PPO is approximately on-policy. Old data becomes stale after a policy
change, so it is normally discarded.

An off-policy method can update a target policy using data from other behavior
policies. SAC, TD3, DQN, IQL, and CQL are off-policy in different settings.
A replay buffer improves sample reuse but makes the learning distribution less
like current deployment.

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

## 4.9 A practical selection procedure

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

## 4.10 Why PPO is sensible for Microduck

Microduck has:

- a 14D continuous action;
- thousands of GPU-parallel simulated robots;
- inexpensive fresh rollouts;
- dense task/recovery signals; and
- a small actor needed for 50 Hz deployment.

These properties fit PPO. One iteration with 4,096 environments and 24 steps
collects 98,304 fresh transitions. Low sample reuse is less painful when
simulation generates this much data quickly.

PPO does not cause walking by itself. Robot model, observation, command
distribution, reward, reset distribution, actuator dynamics, randomization,
and curriculum define the experience from which PPO learns.

## 4.11 Why a different robot may need a different method

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

## 4.12 Common selection mistakes

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

## 4.13 Exercises

For each scenario, select a starting algorithm and state what evidence would
change your mind:

1. Cart-pole with left/right discrete actions and unlimited simulation.
2. Microduck velocity tracking in 4,096 parallel environments.
3. A robot arm with a fixed dataset and no permission for online exploration.
4. An image-based manipulation task with several valid grasp trajectories.
5. A rover with a known dynamics model, strict constraints, and a 20 Hz local
   planner.
6. A legged robot whose payload changes during an episode.

Then make a two-column list: what the learning algorithm decides versus what
the system architecture must decide. Continue with
[PPO from equations to code](05_ppo_from_equations_to_code.md).

## 4.14 Folded solutions

<details>
<summary>Show reference answers to Section 4.13</summary>

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
   multimodal imitation policy. Plain squared-error BC may average incompatible
   grasps. Additional reward-bearing data could justify offline RL.
5. **Known constrained rover:** start with model-predictive control (MPC), which
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

</details>
