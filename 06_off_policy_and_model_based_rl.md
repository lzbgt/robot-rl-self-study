# 6. Off-Policy and Model-Based Reinforcement Learning

Proximal Policy Optimization (PPO) throws away old rollouts after a few update
epochs. That can be entirely reasonable in fast parallel simulation, but it
feels wasteful when one physical robot transition is expensive. Off-policy and
model-based methods try to learn more from each interaction.

This chapter explains the motivation and failure modes before the algorithms.

### Two historical routes to transition efficiency

Off-policy value learning descends from Q-learning: learn about one target
policy while data may come from another. Deep Deterministic Policy Gradient
(DDPG) extended this route to continuous actions using a learned actor; Twin
Delayed Deep Deterministic Policy Gradient (TD3) directly addressed its
overestimation and brittle actor
updates. Soft Actor-Critic (SAC) added a stochastic maximum-entropy policy and
became a widely used continuous-control baseline.

Model-based control follows a different route. System identification and Model
Predictive Control (MPC) plan through known or fitted dynamics. Probabilistic
Ensembles with Trajectory Sampling (PETS) represented uncertainty; model-based policy
optimization mixed short model rollouts with real replay; Dreamer learned and
acted in a recurrent latent world; TD-MPC learned a control-centered latent
model and planned locally. The 2026 frontier still contains both routes and
hybrids—no theorem makes replay or imagination universally cheaper than fresh
parallel simulation.

## 6.1 Replay buffers: experience as a reusable dataset

An off-policy learner commonly stores transitions

```math
(s_t,a_t,r_t,s_{t+1},d_t)
```

in a **replay buffer**, where $d_t$ indicates termination. Training samples
minibatches from the buffer rather than only the newest trajectory.

Replay provides:

- reuse of expensive experience;
- shuffled samples instead of strongly correlated time neighbors; and
- a mixture of behavior from several policy versions.

That mixture is also the difficulty. The current actor may visit states and
actions differently from the buffer. A critic can be asked about actions for
which the data provides weak evidence.

Two quantities describe the engineering regime:

- **replay ratio** or update-to-data ratio: gradient samples consumed divided
  by new environment transitions;
- **buffer age and coverage**: how old and behaviorally diverse sampled data
  is relative to the current policy.

A ratio of 32 means 32 sampled transition uses per new transition, not
necessarily 32 unique optimizer steps. Larger ratios can improve reuse until
the learner overfits a narrow buffer, targets drift, or compute becomes the
bottleneck. Report both environment steps and gradient updates.

## 6.2 State-action critics

PPO usually uses a state-value critic $V(s)$. Continuous off-policy
actor-critic methods often learn

```math
Q_\phi(s,a),
```

the expected return for taking action $a$ in state $s$ and following a policy
afterward.

A one-step critic target has the form

```math
y=r+\gamma(1-d)\,\text{next-value}.
```

The factor $(1-d)$ removes future value after a true terminal transition. A
time-limit truncation may need different handling because the underlying task
could have continued. Confusing termination with truncation biases targets.

The squared critic objective over replay distribution $D$ is

```math
L_Q(\phi)=
\mathbb{E}_{(s,a,r,s',d)\sim D}
[(Q_\phi(s,a)-y)^2].
```

This loss only constrains state-action pairs represented in $D$. An actor can
query $Q_\phi(s,a)$ at a different action $a=\pi_\theta(s)$, so low replay loss
does not imply accurate gradients at actor-proposed actions.

Target networks reduce rapid feedback. A **Polyak** or exponential moving
average update is

```math
\bar\phi\leftarrow\rho\bar\phi+(1-\rho)\phi,
```

with $\rho$ close to 1. The target follows slowly rather than being copied on
every critic step.

## 6.3 Why critics become overoptimistic

Suppose a critic has small random errors across actions. Selecting the maximum
tends to select not only a genuinely good action but also a positive estimation
error. Repeating this in bootstrapped targets can inflate values.

Robot consequence: the actor may exploit critic errors by proposing an
out-of-distribution action that the critic imagines is excellent, even though
the dataset never demonstrated it safely.

Two widely used responses are:

- learn two critics and use the smaller estimate; and
- constrain or regularize actions toward regions covered by data.

## 6.4 Twin Delayed Deep Deterministic Policy Gradient (TD3): deterministic continuous control

Twin Delayed Deep Deterministic Policy Gradient (TD3) learns:

- deterministic actor $a=\pi_\theta(s)$;
- two critics $Q_{\phi_1}$ and $Q_{\phi_2}$; and
- slowly moving target copies.

Its target is approximately

```math
y=r+\gamma(1-d)
\min_{i=1,2}Q_{\bar\phi_i}
\left(s',\pi_{\bar\theta}(s')+\epsilon\right),
```

where target noise $\epsilon$ is clipped.

The three ideas encoded in the name/paper are:

1. **clipped double Q**: use the smaller of two critics to reduce optimistic
   error;
2. **delayed policy update**: update the actor less often than critics; and
3. **target policy smoothing**: train the value of a neighborhood, not a narrow
   action spike.

The actor is optimized to choose actions its critic values:

```math
\max_\theta\ \mathbb{E}_{s\sim D}
[Q_{\phi_1}(s,\pi_\theta(s))].
```

The chain rule gives the deterministic actor gradient:

```math
\nabla_\theta J
\approx
\mathbb{E}_{s\sim D}
\left[
\left.\nabla_a Q_\phi(s,a)\right|_{a=\pi_\theta(s)}
\nabla_\theta\pi_\theta(s)
\right].
```

Read it from left to right: the critic says which small action change would
increase predicted value, then the actor Jacobian says how weights must change
to produce that action change. This makes critic smoothness and correctness at
actor actions crucial. A narrow erroneous Q peak can directly pull the actor
toward it.

Target policy smoothing adds bounded noise to the next action inside the
target. It approximates valuing a neighborhood:

```math
\tilde Q(s',a')
\approx\mathbb{E}_{\epsilon}[Q(s',a'+\epsilon)],
```

so an isolated critic spike is less attractive. Exploration noise on actions
collected into replay is a separate mechanism from this target noise.

The [TD3 paper](https://arxiv.org/abs/1802.09477) isolates these mechanisms.
Its benchmark findings support those mechanisms in the evaluated continuous
control tasks; they do not erase the need for robot-specific evaluation.

## 6.5 Soft Actor-Critic (SAC): maximum-entropy actor-critic

Soft Actor-Critic (SAC) learns a stochastic policy and adds entropy to the
return:

```math
J(\pi)=\mathbb{E}\left[\sum_{t=0}^{T-1}\gamma^t
\left(r_t+\alpha\mathcal{H}
(\pi(\cdot\mid s_t))\right)\right].
```

$\mathcal{H}$ is entropy and $\alpha$ is a temperature. In plain language, the
policy values reward while retaining action diversity when several actions are
plausible.

For continuous actions, implementations commonly sample an unconstrained
Gaussian variable using a differentiable reparameterization, then apply `tanh`
to bound it:

```math
u=\mu_\theta(s)+\sigma_\theta(s)\odot\xi,
\quad \xi\sim\mathcal{N}(0,I),
\quad a=\tanh(u).
```

$\odot$ means elementwise multiplication. Writing randomness as an input
$\xi$ allows gradients to flow through $\mu$ and $\sigma$.

A soft critic target includes the next action's entropy term:

```math
y=r+\gamma(1-d)
\left[
\min_i Q_{\bar\phi_i}(s',a')
-\alpha\log\pi_\theta(a'\mid s')
\right].
```

The actor minimizes approximately

```math
J_\pi=\mathbb{E}
\left[\alpha\log\pi_\theta(a\mid s)
-\min_iQ_{\phi_i}(s,a)\right].
```

Interpretation: prefer actions the critics value, while paying a cost for
collapsing the action distribution too sharply. Modern SAC often tunes
$\alpha$ toward a target entropy instead of fixing it.

### Soft value and temperature, step by step

The maximum-entropy state value is

```math
V(s)=\mathbb{E}_{a\sim\pi}
[Q(s,a)-\alpha\log\pi(a\mid s)].
```

Because entropy is $-\mathbb{E}[\log\pi]$, subtracting
$\alpha\log\pi$ adds an entropy bonus. Insert this value into the Bellman
target to obtain the earlier SAC target. The actor loss is the negative of the
same soft value under reparameterized samples.

Automatic temperature tuning can minimize

```math
J(\alpha)=
\mathbb{E}_{a\sim\pi}
[-\alpha(\log\pi(a\mid s)+\mathcal{H}_{target})].
```

If entropy is below the chosen target, the update tends to increase $\alpha$
and put more weight on diversity; if entropy is above target, it reduces that
pressure. The target entropy is still a design choice, not discovered task
intent.

The `tanh` action transform requires a probability correction. If
$a=\tanh(u)$, change of variables gives, componentwise,

```math
\log\pi_A(a\mid s)=
\log\pi_U(u\mid s)-\sum_j\log(1-\tanh^2(u_j)).
```

Omitting the Jacobian term makes the actor optimize the wrong bounded-action
density, especially near $-1$ and $1$. Mature implementations clamp or
rearrange this calculation for numerical stability.

The primary [SAC paper](https://arxiv.org/abs/1801.01290) motivated stable,
sample-efficient continuous control. “Sample efficient” still depends on what
counts as a sample, update-to-data ratio, environment, and compute.

## 6.6 PPO versus SAC as an engineering decision

| Question | PPO tendency | SAC tendency |
| --- | --- | --- |
| Fresh simulation is cheap? | excellent | also possible |
| Each transition is expensive? | wastes more data | replay is attractive |
| Thousands of synchronous environments? | natural fit | replay/update design needs care |
| Simplicity of distributed rollout? | relatively simple | buffer and target networks add state |
| Old data from prior policies? | mostly discarded | reusable if relevant |
| Primary instability concern | policy update size | critic error/distribution shift |

Do not compare only final reward. Match environment steps, wall time, compute,
network size, evaluation protocol, and number of seeds.

## 6.7 What a model adds

A dynamics model predicts the next state distribution and possibly reward:

```math
p_\psi(s_{t+1},r_t\mid s_t,a_t).
```

If the model is known, Model Predictive Control (MPC) can repeatedly:

1. observe the current state;
2. simulate candidate action sequences;
3. score predicted futures;
4. execute only the first action; and
5. replan from the next observation.

Executing only the first action is **receding-horizon control**. Replanning
limits damage from prediction error because reality corrects the plan at every
step.

For horizon $H$, a deterministic planner can solve

```math
\max_{a_{t:t+H-1}}
\left[
\sum_{k=0}^{H-1}\gamma^k
\hat r(\hat s_{t+k},a_{t+k})
+\gamma^H\hat V(\hat s_{t+H})
\right]
```

subject to

```math
\hat s_{t+k+1}=\hat f(\hat s_{t+k},a_{t+k}).
```

The terminal value $\hat V$ summarizes reward beyond the short planning
horizon. Without it, a short planner can be myopic; with a bad learned value,
it can inherit critic error.

The cross-entropy method (CEM) is a gradient-free sequence optimizer:

```text
initialize a Gaussian over H-step action sequences
repeat:
    sample candidate sequences
    roll each through the model and score it
    keep the lowest-cost elite fraction
    refit each time step's mean and standard deviation to elites
execute only the first mean action; observe reality; replan
```

Run
[`examples/cem_mpc_point_mass.py`](examples/cem_mpc_point_mass.py) to see the
equations control a one-dimensional point mass. It deliberately contains no
learning: replace its exact `dynamics` with a fitted model and it becomes the
planning core of a simple model-based learning experiment.

Model-based reinforcement learning (RL) learns some or all of the model and
uses it to improve behavior.

## 6.8 Model error compounds through imagination

Let a learned model have a small one-step error. Rolling it forward for 100
steps feeds predictions back as inputs, so error can compound. The policy or
planner may also seek regions where the model is wrong in a favorable way.

A simple bound makes the pressure visible. Suppose true dynamics $f$ and
learned dynamics $\hat f$ differ by at most $\epsilon$ per step, and both are
$L$-Lipschitz in state. If $e_k=\lVert s_k-\hat s_k\rVert$, then roughly

```math
e_{k+1}\leq Le_k+\epsilon.
```

Starting from the same state, unrolling gives

```math
e_H\leq\epsilon\sum_{i=0}^{H-1}L^i.
```

For $L=1$, the bound grows as $H\epsilon$; for $L>1$, it can grow
geometrically. This is a worst-case bound, not a forecast, but it explains why
longer imagination is not free and why unstable/contact-rich dynamics are
difficult.

Mitigations include:

- short planning horizons;
- model ensembles to estimate disagreement;
- uncertainty penalties;
- frequent replanning from real observations;
- training the model on the policy's current distribution;
- latent representations that focus on control-relevant information; and
- value functions to approximate outcomes beyond the short model horizon.

## 6.9 Latent world models

Images contain far more pixels than control-relevant state. A **latent model**
compresses observations into a learned representation $z_t$, predicts latent
dynamics, and learns reward/value in that space.

Conceptually:

```text
camera/history -> encoder -> latent state z_t
                                  |
                           learned dynamics
                     (z_t, a_t) -> z_{t+1}
                                  |
                         imagined trajectories
                                  |
                           actor/value updates
```

The latent state is not guaranteed to correspond to human-named variables. It
is optimized to support reconstruction, prediction, reward, value, or a
combination.

A recurrent state-space model separates deterministic memory $h_t$ and a
stochastic latent $z_t$:

```math
h_t=f_\psi(h_{t-1},z_{t-1},a_{t-1}),
```

```math
z_t\sim q_\psi(z_t\mid h_t,o_t)
\quad\text{during observation},
```

```math
z_t\sim p_\psi(z_t\mid h_t)
\quad\text{during imagination}.
```

The posterior $q$ may inspect the real observation; the prior $p$ must predict
without it. Training balances prediction terms and a Kullback–Leibler (KL)
divergence that makes prior and posterior compatible:

```math
L_{model}\approx
-\log p(o_t\mid h_t,z_t)
-\log p(r_t\mid h_t,z_t)
-\log p(c_t\mid h_t,z_t)
+\beta D_{KL}(q(z_t\mid h_t,o_t)\,\|\,p(z_t\mid h_t)).
```

$c_t$ represents continuation. Exact DreamerV3 losses include additional
balancing, free-bit, distribution, and normalization details; the equation is
a reading map, not replacement code.

## 6.10 DreamerV3

[DreamerV3](https://arxiv.org/abs/2301.04104) learns a world model from replay,
then trains actor and critic from trajectories imagined in its latent space.
The paper evaluates a single configuration across more than 150 tasks in
several domains and emphasizes normalization and scale-robust objectives.

The important conceptual contribution for a beginner is not “Dreamer replaces
PPO.” It is the separation:

```text
real/sim experience -> learn predictive latent world
learned world        -> generate imagined futures
imagined futures     -> improve policy/value
```

During imagination, the actor samples actions from latent states and the
critic estimates a $\lambda$-return. Gradients can then improve many imagined
steps per real transition. This sample reuse is valuable only insofar as the
learned latent dynamics and reward remain decision-correct where the actor
goes.

Questions to ask before applying it to a robot:

- Does the latent model predict contacts and discontinuities well enough?
- What observation history resolves partial observability?
- Does imagined success correlate with real rollout success?
- What is the runtime actor size and latency?
- How is uncertainty outside the replay distribution handled?

## 6.11 Temporal Difference Learning for Model Predictive Control, second generation (TD-MPC2)

[TD-MPC2](https://arxiv.org/abs/2310.16828) combines a learned latent model,
temporal-difference values, a policy prior, and local trajectory optimization.
Rather than decode future images, it learns task-relevant latent predictions
for control. The paper reports results across 104 online RL tasks and explores
multi-task scaling.

At each decision, planning conceptually samples/refines short action sequences
in latent space, uses learned reward and terminal value to score them, and
executes the first action. The policy helps propose actions and can also serve
as a fast behavior prior.

TD-MPC2 trains a representation so that predicted next latent, reward, and
value agree with replay-derived targets. A schematic joint loss is

```math
L=
c_z\lVert\hat z_{t+1}-\mathrm{stopgrad}(z_{t+1})\rVert^2
+c_r\ell(\hat r_t,r_t)
+c_q\ell(\hat Q_t,y_t)
+c_\pi L_{policy}.
```

`stopgrad` means the target representation is treated as a constant on that
loss branch. The actual method uses distributional/value-normalization and
planning details documented by its paper and
[official implementation](https://github.com/nicklashansen/tdmpc2). The mapping
to look for is: encoder, latent dynamics, reward/value heads, policy prior,
replay update, and trajectory optimizer.

This offers a different runtime tradeoff from a small feed-forward PPO actor:
planning may improve data efficiency or adaptation but consumes more per-step
compute and introduces planner/model failure modes.

### Model-based alternatives through 2026

- **PETS** uses probabilistic ensembles and trajectory sampling to represent
  epistemic uncertainty, then plans without learning a separate actor.
- **Model-Based Policy Optimization (MBPO)** adds short learned-model rollouts
  to real replay, deliberately limiting horizon to control model bias.
- **DreamerV3** learns a generative recurrent latent world and amortized actor;
  it is attractive when observations are rich and online samples are scarce.
- **TD-MPC2** learns control-centered latents and retains decision-time
  planning; it trades actor simplicity for planner compute.
- **Known-physics MPC plus learned residual** preserves explicit constraints
  and asks learning to model only what nominal dynamics miss.

These methods answer different questions. Compare realized task success per
real transition, total compute, planning deadline misses, model calibration,
reset burden, and failure severity—not only a benchmark mean.

## 6.12 Known physics, learned residuals, and hybrid control

Robot learning need not choose between “all classical” and “all neural.” A
hybrid can use:

- known rigid-body dynamics for nominal prediction;
- a learned residual for unmodeled friction or compliance;
- MPC for explicit constraints;
- an RL policy for fast local correction; and
- a hard safety supervisor outside both.

A **residual policy** outputs a bounded correction:

```math
u=u_{nominal}+\Delta u_\theta,
\qquad |\Delta u_\theta|\le u_{residual,max}.
```

This can reduce exploration scope and preserve an interpretable baseline.
However, a poorly defined residual can still destabilize the nominal loop.

## 6.13 A fair comparison experiment

To compare PPO and SAC on one simulated robot:

1. freeze robot model, observations, actions, reward, resets, randomization,
   control rate, and evaluator;
2. give both enough network capacity but report parameter counts;
3. report environment transitions, wall-clock time, and graphics processing
   unit (GPU) time;
4. evaluate deterministic policies on identical held-out seeds and physics;
5. show median and uncertainty across training seeds;
6. include failure categories, not only average return; and
7. preserve resolved configs and checkpoints.

Changing algorithm and reward together answers no clean question.

## 6.14 Exercises

1. Why can a replay buffer improve sample efficiency but worsen distribution
   mismatch?
2. Explain why taking the minimum of two noisy critics can reduce optimistic
   targets. What new pessimistic bias can it introduce?
3. In SAC, what happens qualitatively as $\alpha$ approaches zero?
4. A true terminal transition has $d=1$. Evaluate
   $y=r+\gamma(1-d)V(s')$.
5. Why does planning farther into a learned model sometimes reduce real-world
   performance?
6. Compare actor-only inference with MPC through a learned model for a 100 Hz
   loop. What must be measured?
7. Design a bounded residual action for a wheeled-leg robot with a classical
   balance controller.
8. A learner performs 64 gradient sample uses for each new environment
   transition. What is its update-to-data ratio? Name two reasons increasing it
   further can hurt even though no new robot wear is incurred.
9. With current critic parameter 10, target parameter 4, and Polyak
   $\rho=0.995$, compute the new target parameter.
10. Use the deterministic policy-gradient equation to explain how a false
    local Q peak can move an actor even if that action never appears in replay.
11. In SAC, why must a `tanh`-squashed Gaussian subtract a log-Jacobian term?
12. For model one-step error $\epsilon=0.01$, Lipschitz factor $L=1.2$, and
    horizon 3, evaluate the derived upper bound on state error.
13. Explain why CEM executes only the first planned action. What information is
    thrown away and why can discarding it be beneficial?
14. Compare PETS, DreamerV3, TD-MPC2, and known-physics MPC along: learned
    representation, decision-time planning, actor, and uncertainty/model risk.

Continue with [robot dynamics, control, and estimation](07_robotics_control_and_estimation.md).

## 6.15 Folded solutions

<details>
<summary>Show answers to Section 6.14</summary>

1. Replay reuses each expensive transition and decorrelates adjacent samples,
   improving transition efficiency. But old data came from older policies and
   possibly older task/robot distributions, so its state-action coverage can
   differ from the policy currently being optimized.
2. If critic noise is sometimes optimistically high, taking the smaller of two
   estimates makes that error less likely to enter the target. The minimum can
   instead be systematically low, creating conservative or pessimistic bias.
3. As SAC temperature $\alpha$ approaches zero, entropy contributes less and
   the objective approaches ordinary reward-maximizing actor-critic behavior.
   Exploration/diversity pressure weakens.
4. With $d=1$, the bootstrap multiplier is zero, so $y=r$. Bootstrapping beyond
   a true terminal state would incorrectly assign value after the episode has
   ended.
5. Small one-step model errors compound as predicted states become inputs to
   later predictions. Farther plans can exploit model mistakes or cross contact
   events the model predicts poorly; shorter receding horizons refresh from
   reality more often.
6. Measure worst-case—not just average—inference/planning time, jitter,
   deadline misses, horizon/sample count, model error, memory, observation age,
   action continuity, and fallback behavior. A 100 Hz loop has only 10 ms for
   the entire sensor-to-command path.
7. Let a classical balance controller output wheel target $u_c$ and constrain
   the learned correction:

   ```python
   residual_limit_rad_s = 1.0
   residual = max(-residual_limit_rad_s,
                  min(residual_limit_rad_s, actor_output))
   wheel_target = classical_target + residual
   ```

   Expire the residual on stale input or deadline miss, rate-limit the combined
   target, preserve microcontroller unit (MCU) current/speed/tilt limits, and
   prove that residual zero returns to the accepted baseline.
8. The ratio is 64 sampled transition uses per new transition. More reuse can
   overfit recent/narrow coverage, amplify bootstrapped critic error, make
   targets chase a fast learner, bias training toward stale behavior, or make
   optimizer compute dominate wall time. Robot wear is only one resource.
9. Polyak averaging gives

   ```math
   \bar\phi_{new}=0.995(4)+0.005(10)=3.98+0.05=4.03.
   ```

   The target moves only 0.03 toward the current critic.
10. The actor queries $\nabla_aQ(s,a)$ specifically at
    $a=\pi_\theta(s)$, not only at replay actions. If approximation creates a
    false rising slope/peak there, the chain rule turns that slope into an actor
    parameter update toward the unsupported action. Low error on stored actions
    does not constrain every queried gradient.
11. Probability density changes under a nonlinear coordinate transform. `tanh`
    compresses an unbounded interval near action limits, so equal intervals in
    pre-squash $u$ do not map to equal intervals in $a$. The Jacobian correction
    accounts for that volume change; without it, entropy and actor objectives
    describe the wrong action density.
12. The bound is

    ```math
    e_3\leq0.01(1+1.2+1.2^2)=0.01(3.64)=0.0364.
    ```

    It is a worst-case norm bound under the stated uniform/Lipschitz assumptions,
    not an expected empirical error.
13. After one action, the system obtains a new real observation. Replanning
    replaces predicted state with measured/estimated state and can adapt to
    disturbances/model error. The remaining $H-1$ candidate actions are
    discarded (or sometimes warm-start the next distribution); blindly
    executing them open-loop would accumulate prediction error.
14. PETS learns a probabilistic ensemble and plans at every decision without a
    required actor; ensemble disagreement exposes one kind of model uncertainty.
    DreamerV3 learns a recurrent stochastic latent world plus actor/critic and
    normally deploys the amortized actor; imagination error affects training.
    TD-MPC2 learns a control-centered latent, policy/value, and dynamics and
    retains online trajectory optimization. Known-physics MPC may fit only
    parameters/residuals, plans online, and can express constraints, but wrong
    nominal physics and state estimates remain risks. Runtime and data costs
    differ, so the comparison is conditional.

</details>
