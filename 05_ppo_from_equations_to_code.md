# 5. Proximal Policy Optimization (PPO) from Equations to Code

This chapter explains how Proximal Policy Optimization turns a batch of
Microduck rollouts into improved actor and critic networks.

### Origin and scope

Williams's 1992 REINFORCE algorithm supplied a general sampled policy-gradient
estimator. Actor-critic methods learned a baseline to reduce its variance.
Natural policy gradient and Trust Region Policy Optimization (TRPO) measured
steps in policy-distribution space. The 2017 PPO paper proposed clipped and
Kullback–Leibler-penalty surrogates that retain this “do not move too far on one
batch” motivation with ordinary first-order optimization.

PPO is an algorithm for updating a policy from on-policy trajectories. It does
not define robot observations, action semantics, reward, actuator physics,
reset distribution, or safety. Those surrounding choices are why two projects
that both say “PPO” can solve very different problems.

## 5.1 Actor and critic

The actor chooses actions. The critic estimates how much discounted reward is
still available from the current situation.

```math
\text{actor: } \pi_\theta(a_t\mid o_t)
```

```math
\text{critic: } V_\phi(x_t) \approx \mathbb{E}[G_t\mid x_t]
```

The actor must use observations available on the real robot. The critic is
needed only during training and may receive privileged simulator information.
In the main velocity task:

| Network | Input | Hidden layers | Output |
| --- | ---: | --- | ---: |
| Actor | 61 | 512 → 256 → 128, exponential linear unit (ELU) | 14 action means |
| Critic | 76 | 512 → 256 → 128, ELU | 1 value estimate |

The critic's extra information includes true base linear velocity and contact
features. This asymmetric actor-critic pattern can make training easier without
creating a deployment dependency.

The actor uses a Gaussian distribution during training. Its network predicts
the mean action, and an exploration standard deviation allows nearby actions
to be sampled. Deployment uses the inference policy rather than deliberately
sampling exploratory motor commands.

The layer widths, learning rate, discount, and similar choices are
**hyperparameters**: settings chosen by the engineer rather than weights
learned by gradient descent. A **gradient** describes how a small parameter
change would change the loss; backpropagation efficiently computes those
gradients through the network.

## 5.2 Why a critic helps

Suppose a robot receives reward 1 in a state. Is that good? If similar states
normally produce reward 5, it was worse than expected. If they normally
produce reward 0, it was better than expected.

The advantage measures this relative quality:

```math
A_t = Q(o_t,a_t) - V(o_t)
```

A positive advantage means the sampled action led to a better outcome than the
critic expected. A negative advantage means it was worse. Policy-gradient
learning increases the probability of positive-advantage actions and decreases
the probability of negative-advantage actions.

### Deriving the likelihood-ratio policy gradient

Start with expected trajectory return:

```math
J(\theta)=\sum_\tau p_\theta(\tau)R(\tau).
```

The trajectory probability factors into initial-state probability, policy
probabilities, and environment transitions:

```math
p_\theta(\tau)=p(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
```

Only the policy factors depend on $\theta$. Differentiate $J$ and multiply and
divide by $p_\theta(\tau)$:

```math
\begin{aligned}
\nabla_\theta J
&=\sum_\tau \nabla_\theta p_\theta(\tau)R(\tau)\\
&=\sum_\tau p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)R(\tau)\\
&=\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)R(\tau)
\right].
\end{aligned}
```

The identity $\nabla p=p\nabla\log p$ is the likelihood-ratio or score-
function trick. It lets the environment be nondifferentiable: contacts and
resets are sampled, while gradients pass through the actor's log-probability.

Rewards before action $a_t$ cannot be caused by that action, so they may be
removed, replacing whole-trajectory $R(\tau)$ with reward-to-go $G_t$. Then
subtract a state baseline $b(s_t)$:

```math
\nabla_\theta J
=\mathbb{E}
\left[
\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
(G_t-b(s_t))
\right].
```

Why does a baseline not bias the expectation? Conditional on $s$:

```math
\begin{aligned}
\mathbb{E}_{a\sim\pi}
[\nabla_\theta\log\pi_\theta(a\mid s)b(s)]
&=b(s)\sum_a\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)\\
&=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
```

Choosing $b=V(s)$ turns $G_t-b(s)$ into an advantage estimate. A better critic
reduces variance; an inaccurate critic can add bias when bootstrapped targets
are used, which is why critic learning and advantage diagnostics matter.

## 5.3 Temporal-difference error and Generalized Advantage Estimation (GAE)

The one-step temporal-difference residual is:

```math
\delta_t = r_t + \gamma V_\phi(x_{t+1}) - V_\phi(x_t)
```

Generalized Advantage Estimation (GAE) combines a sequence of these residuals:

```math
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1}
+ (\gamma\lambda)^2\delta_{t+2} + \cdots
```

Microduck uses $\gamma=0.99$ and $\lambda=0.95$. Lower $\lambda$ relies more
on the critic and usually lowers variance; higher $\lambda$ uses longer sampled
returns and usually lowers bias. The chosen values are a common compromise,
not a law of robotics.

The value target is commonly formed from the advantage and old value estimate:

```math
\hat{G}_t = \hat{A}_t + V_{\text{old}}(x_t)
```

The critic learns by reducing error between $V_\phi(x_t)$ and this target.

### Deriving the efficient backward recurrence

Let $\rho=\gamma\lambda$. Starting from the infinite-looking sum:

```math
\hat A_t=\delta_t+\rho\delta_{t+1}
+\rho^2\delta_{t+2}+\cdots,
```

factor one $\rho$ from the tail:

```math
\hat A_t=\delta_t+\rho
(\delta_{t+1}+\rho\delta_{t+2}+\cdots)
=\delta_t+\gamma\lambda\hat A_{t+1}.
```

That recurrence gives all advantages in one reverse loop. A true terminal mask
$m_t\in\{0,1\}$ prevents both value bootstrapping and propagation across an
episode boundary:

```math
\delta_t=r_t+\gamma m_t V(x_{t+1})-V(x_t),
```

```math
\hat A_t=\delta_t+\gamma\lambda m_t\hat A_{t+1}.
```

For a time-limit truncation of an underlying continuing task, $V(x_{t+1})$
should normally be retained even though the rollout buffer ends. Libraries may
represent masks and timeout bootstrapping differently, so inspect code rather
than infer semantics from a field named `done`.

Run [`examples/gae_walkthrough.py`](examples/gae_walkthrough.py). Its
`generalized_advantage_estimation` function is the equation translated line by
line, including the terminal mask and value targets. Change the last flag from
`True` to `False` and observe why the provided bootstrap value suddenly
matters.

### What $\lambda$ is mixing

Expanding each $\delta$ shows that Generalized Advantage Estimation mixes
one-step, two-step, and longer value targets. With a perfect critic, a shorter
target can already assign useful credit with low variance. With a biased critic,
longer sampled returns reduce dependence on its error. Therefore changing
$\lambda$ is not just “more or less smoothing”; it changes how much the actor's
credit signal trusts the learned value function.

Practical PPO implementations commonly normalize advantages within the batch:

```math
\tilde A_i=\frac{\hat A_i-\overline A}
{\sqrt{\mathrm{Var}_{batch}(\hat A)+\varepsilon}}.
```

This stabilizes gradient scale. It also makes the update depend on batch
composition: a sample's normalized advantage is relative to other commands,
terrains, and episodes in that batch. Preserve this detail when comparing
implementations.

## 5.4 The policy ratio

After collecting a rollout with policy $\pi_{\theta_{old}}$, PPO asks how the
new policy changes the probability of each sampled action:

```math
r_t(\theta) =
\frac{\pi_\theta(a_t\mid o_t)}
     {\pi_{\theta_{old}}(a_t\mid o_t)}
```

- $r_t=1$ means no probability change.
- $r_t>1$ means the sampled action became more likely.
- $r_t<1$ means it became less likely.

For a 14-dimensional diagonal Gaussian, the joint log-probability is the sum
of per-joint log densities:

```math
\log\pi_\theta(a\mid o)=
\sum_{j=1}^{14}
\log\mathcal{N}(a_j;\mu_{\theta,j}(o),\sigma_j^2).
```

The ratio is calculated stably as

```math
r_t(\theta)=
\exp\left[
\log\pi_\theta(a_t\mid o_t)
-\log\pi_{old}(a_t\mid o_t)
\right].
```

The old log-probability must be stored when collecting the action. Recomputing
it after updating the actor would make numerator and denominator describe the
same new policy and erase the intended comparison.

A basic policy-gradient objective would maximize $r_t\hat{A}_t$. Reusing the
same rollout for aggressive updates can move the policy so far that the data
no longer represents it. PPO limits the incentive for a large move.

## 5.5 The clipped PPO objective

The clipped surrogate objective is:

```math
L^{CLIP}(\theta) = \mathbb{E}_t\left[
\min\left(
r_t(\theta)\hat{A}_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right]
```

Microduck uses $\epsilon=0.2$, so the clipped interval is $[0.8,1.2]$.

Example: an action has advantage $+2$, its old probability was 0.10, and its
new probability is 0.13. The ratio is 1.3. The unclipped objective is 2.6, but
the clipped positive-advantage objective is $1.2\times2=2.4$. Increasing its
probability further offers no clipped benefit for this sample.

The sign of advantage changes which side clips. With $\epsilon=0.2$:

| Advantage | Ratio direction that would help the surrogate | Clipped plateau |
| --- | --- | --- |
| $A>0$ | increase probability, $r>1$ | no extra benefit above $1.2$ |
| $A<0$ | decrease probability, $r<1$ | no extra benefit below $0.8$ |

For a negative advantage and $r=1.5$, clipping does **not** protect the sample:
the minimum selects $1.5A$, which is more negative than $1.2A$. PPO blocks
updates only when the probability change would improve the surrogate beyond
the clip boundary. It still penalizes moves in the wrong direction.

Run [`examples/ppo_clip_demo.py`](examples/ppo_clip_demo.py) and inspect both
advantage signs. This four-line scalar function is the direct mapping of the
paper objective:

```python
unclipped = ratio * advantage
clipped = clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
objective = min(unclipped, clipped)
```

Clipping does not mathematically guarantee a small update. The Robotic Systems
Lab reinforcement learning (RSL-RL) library also tracks the approximate
Kullback–Leibler (KL) divergence, a measure of how different the old and new
action distributions are, and uses an adaptive learning-rate schedule with
desired KL `0.01`. Gradient norm is capped at `1.0`.

## 5.6 Value, entropy, and the combined update

The optimizer conceptually combines three goals:

```text
maximize clipped policy improvement
minimize critic value error
preserve enough action-distribution entropy to explore
```

One common minimization form is:

```math
L = -L^{CLIP} + c_v L^{value} - c_e H[\pi_\theta]
```

The main configuration uses value-loss coefficient `1.0` and entropy
coefficient `0.01`. Entropy is not “randomness is always good.” Early in
training it prevents the policy from collapsing before discovering useful
motion. Too much entropy can keep behavior noisy; too little can freeze a poor
strategy.

For an unbounded scalar Gaussian, entropy is

```math
H[\mathcal{N}(\mu,\sigma^2)]
=\frac{1}{2}\log(2\pi e\sigma^2).
```

It depends on $\sigma$, not $\mu$: translating the distribution does not
change its spread. For a diagonal Gaussian, sum this term over action
dimensions. If policy outputs are transformed or clipped, the actual action
distribution may differ; entropy and log-probability must follow the same
distribution semantics.

A simple critic loss is

```math
L^{value}=\frac{1}{B}\sum_{i=1}^{B}
\left(V_\phi(x_i)-\hat G_i\right)^2.
```

Some PPO implementations also clip the value prediction change. This is an
implementation option, not part of the one universal PPO definition. Record
it when reproducing results.

The signs in the combined loss follow the optimizer: gradient descent
minimizes, so the actor objective and entropy bonus receive minus signs, while
value error receives a plus sign. A useful code audit starts by writing the
desired maximize/minimize direction beside every scalar before interpreting a
logged loss.

## 5.7 What one Microduck iteration contains

The default runner uses:

```text
parallel environments       4096 in a normal full run
steps per environment         24
rollout transitions       98,304 per iteration
minibatches                    4
learning epochs                5
initial learning rate       0.001
```

The same rollout is shuffled into four **minibatches** (smaller parts of the
full collected batch) and visited for five optimization **epochs** (complete
passes over that rollout). Then the updated policy collects a new rollout.

At a 50 Hz policy rate, 24 steps represent 0.48 seconds per environment. The
large parallel batch contains many commands, initial conditions, contacts, and
randomized robot instances even though each individual fragment is short.

The logical loop is:

```python
for iteration in range(max_iterations):
    rollout = collect_current_policy(num_envs=4096, steps=24)
    advantages, returns = generalized_advantage_estimation(rollout)

    for epoch in range(5):
        for minibatch in split_and_shuffle(rollout, count=4):
            update_actor_with_clipped_ppo(minibatch)
            update_critic(minibatch)

    log_metrics_and_periodically_save_checkpoint()
```

This is explanatory pseudocode. RSL-RL owns the actual buffers, distribution,
losses, optimizer, and checkpoint format.

The official
[RSL-RL PPO source](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py)
is the implementation mapping to read after the pseudocode. Trace in this
order:

1. rollout storage retains observations, actions, values, rewards, masks, and
   old action log-probabilities;
2. return computation performs the reverse GAE recurrence;
3. the minibatch generator yields stored old quantities and current inputs;
4. the actor evaluates the stored actions under its current distribution;
5. ratio, clipped surrogate, value loss, entropy, gradient norm, and optimizer
   step implement the equations above.

Pin the exact RSL-RL revision from the Microduck lockfile before comparing line
numbers; `main` can evolve after a training run was produced.

## 5.8 Observation normalization

Joint angles, angular velocities, gravity components, and commands have
different numerical scales. The actor and critic use empirical observation
normalizers so one large-magnitude feature does not dominate merely because of
its units.

Conceptually, each feature becomes:

```math
\tilde{o}_i = \frac{o_i - \mu_i}{\sqrt{\sigma_i^2 + \varepsilon}}
```

The running mean and variance are learned from training observations and saved
in the checkpoint. The Open Neural Network Exchange (ONNX) exporter includes
the actor's normalizer in the graph. A hand-converted network that omits it
receives a different numerical problem and may fail on hardware even if
playback from the checkpoint looked correct.

## 5.9 The important hyperparameters in this repository

| Setting | Main value | Practical effect |
| --- | ---: | --- |
| `gamma` | 0.99 | how strongly later rewards count |
| `lam` | 0.95 | GAE bias/variance tradeoff |
| `clip_param` | 0.2 | clips the policy-ratio incentive |
| `learning_rate` | 0.001 | initial optimizer step scale |
| `desired_kl` | 0.01 | target used by adaptive schedule |
| `entropy_coef` | 0.01 | exploration pressure |
| `num_learning_epochs` | 5 | reuse passes over each rollout |
| `num_mini_batches` | 4 | subdivisions per epoch |
| `max_grad_norm` | 1.0 | gradient clipping threshold |
| actor/critic normalization | on | normalizes each observation stream |

Do not tune these merely because a reward curve looks noisy. First verify the
environment, reward signs, resets, observation validity, and actual rollout.
Most project failures have been specification or physics failures rather than
a need for exotic PPO settings.

## 5.10 Reading a PPO update scientifically

Four diagnostics describe different mechanisms:

| Diagnostic | What it measures | Suspicious pattern |
| --- | --- | --- |
| approximate KL | old/new policy distribution change | persistent overshoot beyond target |
| clip fraction | fraction of samples whose improving direction is clipped | near 0 with no learning, or near 1 with violent updates |
| explained variance | how much return-target variation the critic predicts | negative: worse than predicting the batch mean |
| entropy/action standard deviation | exploration spread | early collapse or uncontrolled growth |

Explained variance is commonly

```math
1-\frac{\mathrm{Var}(\hat G-V_\phi)}
{\mathrm{Var}(\hat G)}.
```

A value near 1 indicates predictions explain most target variation; 0 is no
better than predicting a constant mean; a negative value is worse. It is not a
policy-success metric: a critic can accurately predict returns for a bad
policy.

For scalar Gaussians $p=\mathcal{N}(\mu_0,\sigma_0^2)$ and
$q=\mathcal{N}(\mu_1,\sigma_1^2)$:

```math
D_{KL}(p\,\|\,q)=
\log\frac{\sigma_1}{\sigma_0}
+\frac{\sigma_0^2+(\mu_0-\mu_1)^2}{2\sigma_1^2}
-\frac{1}{2}.
```

This shows that changing either action means or exploration scales changes the
policy. Summing across 14 independent dimensions means modest per-joint shifts
can create a large joint policy change.

PPO alternatives do not remove the environment problem. TRPO enforces a more
explicit approximate trust region but costs additional linear algebra. Soft
Actor-Critic reuses replay and optimizes a maximum-entropy objective but relies
on state-action critics. Model-based methods may use fewer real transitions
but introduce model error and planning compute. As of 2026, PPO remains a
reference for massively parallel locomotion, while high-update off-policy
methods are credible matched-budget challengers rather than automatic drop-in
replacements.

## 5.11 Why PPO does not understand intent

PPO only sees sampled data and scalar objectives. It does not know that a
forward roll should be sagittal rather than a shoulder roll, or that “stand”
should not mean balancing on the head. Hard state-based gates and correct task
distributions translate that intent into the optimization problem.

This is also why total reward can improve while the task fails. The agent may
learn cheaper regularizers without improving the main skill. Always inspect
individual weighted reward terms and rollouts.

## 5.12 Check your understanding

1. Why can the critic use information that is unavailable on the real robot?
2. What does an advantage of zero mean for a sampled action?
3. What problem does PPO clipping reduce?
4. How many transitions are collected in one 4,096-environment iteration?
5. Why must observation normalization be included in the deployed ONNX graph?
6. In the policy-gradient derivation, why do simulator transition
   probabilities disappear from $\nabla_\theta\log p_\theta(\tau)$?
7. Prove in one line why a state-only baseline has zero expected score term.
   Why may it still reduce variance?
8. Given rewards $[1,2]$, values $[0.5,0.25]$, a terminal final transition,
   $\gamma=0.9$, and $\lambda=0.8$, compute both GAE advantages backward.
9. The old joint log-probability is $-8.0$ and the new one is $-7.7$. Compute
   the PPO ratio. With $A=2$ and $\epsilon=0.2$, compute the clipped surrogate.
10. With $A=-2$ and ratio 0.5, which branch of the minimum is selected? Explain
    why this is the correct clipping direction.
11. Why must `old_log_probability` be stored before optimization rather than
    recomputed from the updated actor?
12. A run has explained variance 0.95, increasing total reward, and a video of
    the robot farming reward while fallen. What has and has not succeeded?

Continue with
[off-policy and model-based reinforcement learning](06_off_policy_and_model_based_rl.md).

## 5.13 Folded solutions

<details>
<summary>Show answers to Section 5.12</summary>

1. The critic is used to estimate returns/advantages during training and is
   not part of the deployed actor. Simulator-only truth can therefore reduce
   critic noise while the actor remains restricted to reproducible sensors.
   Leakage into actor inputs would break this argument.
2. Zero advantage means the sampled result matched the critic's baseline for
   that observation. The policy-gradient term has no first-order reason to
   make that action more or less likely, though entropy and other loss terms
   may still update the policy.
3. Clipping reduces the incentive for one batch to move action probabilities
   too far from the behavior policy that collected it. It is a practical trust-
   region surrogate, not a proof that every update improves the real task.
4. One iteration collects $4096\times24=98{,}304$ transitions.
5. Normalization changes the function's actual input. Omitting the saved mean
   and variance makes the ONNX actor see a different numeric distribution than
   the network optimized in training.
6. The environment transition kernel and initial-state distribution do not
   depend on actor parameters $\theta$ in the standard derivation. Their log
   derivatives are therefore zero. If the environment itself is jointly
   parameterized by $\theta$, that assumption must be revisited.
7. Conditional on state $s$:

   ```math
   \mathbb{E}_{a\sim\pi}
   [b(s)\nabla\log\pi(a\mid s)]
   =b(s)\nabla\sum_a\pi(a\mid s)=b(s)\nabla1=0.
   ```

   A baseline centered near expected return makes the remaining advantage
   smaller and less variable, so finite-batch gradient estimates fluctuate
   less even though their expectation is unchanged.
8. At the final step, continuation is zero:

   ```math
   \delta_1=2-0.25=1.75,
   \qquad \hat A_1=1.75.
   ```

   At the first step:

   ```math
   \delta_0=1+0.9(0.25)-0.5=0.725,
   ```

   ```math
   \hat A_0=0.725+0.9(0.8)(1.75)=1.985.
   ```

9. The ratio is $\exp(-7.7-(-8.0))=e^{0.3}\approx1.350$. The improving
   positive-advantage branch clips at 1.2, so the surrogate is
   $1.2\times2=2.4$ rather than about 2.700.
10. Unclipped is $0.5(-2)=-1$; clipped is $0.8(-2)=-1.6$. The minimum selects
    $-1.6$, preventing extra surrogate benefit from reducing the probability
    of a negative-advantage action below the lower boundary.
11. The denominator must describe the behavior policy that generated the
    stored action. Recomputing after updates would replace historical evidence
    with the current distribution, often making the ratio artificially near
    one and breaking the importance-ratio interpretation.
12. The critic has succeeded at explaining the return targets of the sampled
    (reward-hacking) behavior, and the optimizer has raised its specified
    objective. The environment specification and intended-task acceptance have
    failed. High explained variance says nothing about whether the reward
    represents the desired skill.

A minimal transition-count check is:

```python
num_envs = 4096
steps_per_env = 24
assert num_envs * steps_per_env == 98_304
```

</details>
