# 6. Off-Policy and Model-Based Reinforcement Learning

Proximal Policy Optimization (PPO) throws away old rollouts after a few update
epochs. That can be entirely reasonable in fast parallel simulation, but it
feels wasteful when one physical robot transition is expensive. Off-policy and
model-based methods try to learn more from each interaction.

This chapter explains the motivation and failure modes before the algorithms.

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

Model-based reinforcement learning (RL) learns some or all of the model and
uses it to improve behavior.

## 6.8 Model error compounds through imagination

Let a learned model have a small one-step error. Rolling it forward for 100
steps feeds predictions back as inputs, so error can compound. The policy or
planner may also seek regions where the model is wrong in a favorable way.

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

This offers a different runtime tradeoff from a small feed-forward PPO actor:
planning may improve data efficiency or adaptation but consumes more per-step
compute and introduces planner/model failure modes.

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

</details>
