# 5. Proximal Policy Optimization (PPO) from Equations to Code

This chapter explains how Proximal Policy Optimization turns a batch of
Microduck rollouts into improved actor and critic networks.

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

## 5.10 Why PPO does not understand intent

PPO only sees sampled data and scalar objectives. It does not know that a
forward roll should be sagittal rather than a shoulder roll, or that “stand”
should not mean balancing on the head. Hard state-based gates and correct task
distributions translate that intent into the optimization problem.

This is also why total reward can improve while the task fails. The agent may
learn cheaper regularizers without improving the main skill. Always inspect
individual weighted reward terms and rollouts.

## 5.11 Check your understanding

1. Why can the critic use information that is unavailable on the real robot?
2. What does an advantage of zero mean for a sampled action?
3. What problem does PPO clipping reduce?
4. How many transitions are collected in one 4,096-environment iteration?
5. Why must observation normalization be included in the deployed ONNX graph?

Continue with
[off-policy and model-based reinforcement learning](06_off_policy_and_model_based_rl.md).

## 5.12 Folded solutions

<details>
<summary>Show answers to Section 5.11</summary>

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

A minimal transition-count check is:

```python
num_envs = 4096
steps_per_env = 24
assert num_envs * steps_per_env == 98_304
```

</details>
