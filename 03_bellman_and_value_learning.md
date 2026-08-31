# 3. Bellman Methods: From Tables to Deep Q-Learning

Before Proximal Policy Optimization (PPO), learn the simpler idea that
organizes much of reinforcement learning (RL): the value of a decision is its
immediate reward plus the value of what follows. This is the **Bellman
principle**.

The tabular setting in this chapter is deliberately small. When every state and
action can be listed, the algorithms expose their logic without a neural
network hiding mistakes.

### Historical thread: one recursive idea, several data regimes

Bellman's 1957 dynamic programming assumed a known model and decomposed a long
optimization into one-step subproblems. Monte Carlo methods estimated outcomes
from complete samples. Temporal-difference methods learned a Bellman fixed
point from incomplete experience. Watkins's 1989 thesis and the 1992
Watkins–Dayan paper established Q-learning's off-policy control rule in the
tabular setting. The 2013 Deep Q-Network (DQN) work combined Q-learning with a
convolutional network, replay, and a target network at influential scale.

The changes are not just “use a bigger network.” Each step relaxes an
assumption—known model, enumerable state, on-policy data—while introducing a
new source of approximation or instability. The primary records are indexed in
[SOURCES.md](SOURCES.md#foundations-and-policy-optimization).

## 3.1 Start with a bandit: action without state transition

A **multi-armed bandit** has several actions (“arms”). Each arm produces reward
from an unknown distribution. There is no changing state and no delayed
consequence. The learner must balance:

- **exploration**: try uncertain actions to gain information; and
- **exploitation**: choose the action currently estimated to be best.

If action $a$ has been selected $N(a)$ times, an incremental sample-average
estimate is

```math
Q_{new}(a)=Q_{old}(a)+\frac{1}{N(a)}
            \left(r-Q_{old}(a)\right).
```

Read the term in parentheses as **prediction error**: observed reward minus
the old prediction. The $1/N(a)$ step size shrinks as evidence accumulates.

An $\epsilon$-greedy policy chooses a random action with probability
$\epsilon$ and the estimated best action otherwise. If $\epsilon=0.1$, about
10% of decisions explore.

Robotics connection: choosing among calibration routines or high-level skills
can resemble a contextual bandit, but balancing a moving robot is sequential.
An action changes the next state and therefore future choices.

## 3.2 State value and action value

For a policy $\pi$, the **state-value function** is the expected discounted
return starting from state $s$:

```math
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s].
```

The **action-value function**, usually called the Q-function, also fixes the
first action:

```math
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
```

Plain language:

- $V^\pi(s)$ asks, “How good is this situation if I continue with policy
  $\pi$?”
- $Q^\pi(s,a)$ asks, “How good is taking this action here, then continuing
  with $\pi$?”

These are expectations over future transition randomness and future policy
sampling. They are not promises for one rollout.

## 3.3 The Bellman expectation equation

Split the return into the next reward and the remaining future:

```math
G_t=R_{t+1}+\gamma G_{t+1}.
```

Taking expectations gives

```math
V^\pi(s)=
\sum_a\pi(a\mid s)
\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma V^\pi(s')\right].
```

Do not let the sums obscure the idea:

1. consider each action the policy might take;
2. consider each next state and reward the environment might produce;
3. add immediate reward to discounted next-state value; and
4. average using their probabilities.

This is a self-consistency equation: a correct value estimate agrees with a
one-step lookahead using itself.

### Matrix form: policy evaluation is a linear system

For a finite Markov Decision Process (MDP) and a fixed policy, collect all
state values into vector $v$, expected one-step rewards into $r_\pi$, and
policy-induced transition probabilities into matrix $P_\pi$. Then

```math
v=r_\pi+\gamma P_\pi v.
```

Move the value term to the left:

```math
(I-\gamma P_\pi)v=r_\pi,
```

and, when the inverse exists,

```math
v=(I-\gamma P_\pi)^{-1}r_\pi.
```

$I$ is the identity matrix. This exact solve is practical only for small
problems, but it reveals what iterative methods approximate: a fixed point of
a linear operator. In a million-dimensional or continuous robot state, storing
$P_\pi$ is impossible, while sampling one transition is easy.

### Numerical example

Suppose a state has one action. It gives reward 2 and moves deterministically
to a state worth 5. With $\gamma=0.9$:

```math
V(s)=2+0.9\times5=6.5.
```

If the next state were terminal, its future value would be zero and the answer
would be 2.

## 3.4 Bellman optimality

The optimal action-value function chooses the best next action:

```math
Q^*(s,a)=
\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\max_{a'}Q^*(s',a')\right].
```

Once $Q^*$ is known, a greedy optimal policy chooses

```math
\pi^*(s)=\arg\max_a Q^*(s,a).
```

$\arg\max$ returns the action producing the largest value, not the value
itself.

## 3.5 Dynamic programming: when the model is known

**Dynamic programming** (DP) assumes the transition/reward model is known.
Value iteration repeatedly applies an optimal Bellman backup:

```math
V_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma V_k(s')\right].
```

Repeated backups propagate information backward from goals and hazards.

### Why discounted Bellman iteration converges in the tabular case

Define the Bellman optimality operator

```math
(TV)(s)=\max_a\mathbb{E}[R_{t+1}+\gamma V(S_{t+1})\mid s,a].
```

For two bounded value functions $V$ and $W$, the maximum and expectation cannot
amplify their largest difference beyond the discount:

```math
\lVert TV-TW\rVert_\infty
\leq\gamma\lVert V-W\rVert_\infty.
```

This makes $T$ a **contraction** when $\gamma<1$. After $k$ exact sweeps,

```math
\lVert V_k-V^*\rVert_\infty
\leq\gamma^k\lVert V_0-V^*\rVert_\infty.
```

The error shrinks geometrically toward a unique fixed point. This clean result
depends on exact tabular backups. A neural network update changes many values
at once and may use off-policy sampled targets, so the proof does not transfer
unchanged to deep Q-learning.

Run:

```bash
python examples/gridworld_value_iteration.py
```

The program has the full grid transition model, so it can evaluate every
one-step outcome without collecting experience. In most real robotics tasks,
the exact stochastic model is unknown and continuous state makes a table
impossible. DP still provides the conceptual reference.

## 3.6 Monte Carlo learning: wait for the outcome

**Monte Carlo** (MC) methods estimate value from complete sampled returns. If a
state is visited and the later rewards are $1,0,3$, with $\gamma=0.9$:

```math
G_t=1+0.9(0)+0.9^2(3)=3.43.
```

The estimate can move toward 3.43 after the episode finishes. MC does not need
a transition model and does not bootstrap from its own current value estimate.
But it must wait for an outcome and can have high variance.

The sample return is unbiased for the value of the policy under the sampled
start condition, but a single rollout mixes every future source of randomness.
Long robot episodes can therefore give extremely noisy credit. Truncating an
episode and incorrectly setting the remaining return to zero adds bias of a
different kind.

## 3.7 Temporal-difference learning: learn before the episode ends

**Temporal-difference** (TD) learning updates from one transition:

```math
V(S_t)\leftarrow V(S_t)+\alpha\delta_t,
```

where

```math
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t).
```

$\delta_t$ is the **TD error**:

```text
new one-step target - old prediction
```

The next state's current estimate supplies the unfinished future. Reusing an
estimate inside its own target is called **bootstrapping**.

### Numerical example

Let $V(S_t)=4$, reward $R_{t+1}=1$, $V(S_{t+1})=6$,
$\gamma=0.9$, and $\alpha=0.1$.

```math
\delta_t=1+0.9(6)-4=2.4,
```

```math
V(S_t)\leftarrow4+0.1(2.4)=4.24.
```

The experience was better than predicted, so the estimate rises.

### The spectrum between one-step TD and Monte Carlo

An $n$-step target uses $n$ observed rewards and then bootstraps:

```math
G_t^{(n)}=
\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+\gamma^nV(S_{t+n}).
```

- $n=1$ is the one-step temporal-difference (TD) target.
- $n$ reaching the true terminal state is a Monte Carlo target.
- Intermediate $n$ trades a longer piece of real outcome against a later
  learned bootstrap.

The forward-view $\lambda$-return mixes all $n$-step returns:

```math
G_t^\lambda=(1-\lambda)
\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}.
```

Eligibility traces compute an equivalent backward credit signal online in the
tabular continuing case. Generalized Advantage Estimation in Chapter 5 uses
the same geometric weighting idea on TD residuals. This is an example of an
old mechanism surviving inside a newer deep algorithm.

## 3.8 State–Action–Reward–State–Action (SARSA) and Q-learning

SARSA is an on-policy TD control method. Its name lists the transition tuple:
$S_t,A_t,R_{t+1},S_{t+1},A_{t+1}$. Its update target follows the action the
behavior policy actually selects:

```math
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\right].
```

Q-learning is off-policy. Its target assumes the greedy next action even if
the behavior policy explored:

```math
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha
\left[R_{t+1}+\gamma\max_aQ(S_{t+1},a)-Q(S_t,A_t)\right].
```

**On-policy** means learning about the behavior policy generating the data.
**Off-policy** means the policy being learned can differ from the behavior
policy. Off-policy learning enables data reuse but introduces harder stability
and distribution-shift problems.

Run:

```bash
python examples/tabular_q_learning.py
```

Try three values of $\epsilon$. Record both final success and the number of
episodes needed. Exploration is not free, and no single setting is universally
best.

The implementation maps directly to the equation:

```python
next_best = 0.0 if done else max(q[(next_state, a)] for a in ACTIONS)
td_error = reward + GAMMA * next_best - q[(state, action)]
q[(state, action)] += ALPHA * td_error
```

The first line implements the terminal-aware bootstrap. The second constructs
the bracketed Q-learning error. The third applies the step size $\alpha$. Read
the complete runnable mapping in
[`examples/tabular_q_learning.py`](examples/tabular_q_learning.py), then alter
only the target to use the actually selected next action to obtain SARSA.

In the classic cliff example, that difference matters. Q-learning learns the
greedy target path even while exploratory behavior sometimes falls from it;
SARSA evaluates its exploratory behavior and can prefer a safer path farther
from the cliff. Neither label means “safe”: the result depends on reward,
exploration, and environment.

## 3.9 From Q-table to Deep Q-Network

A robot camera produces too many possible observations for a table. A Deep
Q-Network (DQN) approximates all discrete action values with a neural network:

```math
Q_\theta(o,a).
```

The squared TD loss for one sample is

```math
L(\theta)=
\left(
r+\gamma\max_{a'}Q_{\bar\theta}(o',a')
-Q_\theta(o,a)
\right)^2.
```

$\bar\theta$ denotes a slowly updated **target network**. Two important DQN
mechanisms are:

- **experience replay**: store transitions and sample shuffled minibatches,
  which reuses data and reduces adjacent-sample correlation;
- **target network**: hold the target parameters fixed temporarily so the
  prediction does not chase a target changing on every optimizer step.

The original
[DQN paper](https://arxiv.org/abs/1312.5602) demonstrated value learning from
pixels on Atari. That is historically important evidence for discrete-action
domains; it is not evidence that DQN is the natural choice for 14 simultaneous
continuous joint targets.

### One DQN update, from tensor to optimizer

For a replay minibatch of size $B$, a practical update is:

```python
with no_gradient():
    next_q = target_network(next_observation).max(axis=1)
    target = reward + gamma * (1.0 - terminated) * next_q

all_q = online_network(observation)           # shape (B, number_of_actions)
chosen_q = gather(all_q, action_index)        # shape (B,)
loss = mean(huber_loss(chosen_q, target))
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`gather` is the code counterpart of selecting $Q(s,a)$ for the action stored in
each transition. The maximum selects the target action. `no_gradient` and the
target network prevent the target branch from being optimized in the same
step. A Huber loss is quadratic near zero and linear for large errors, reducing
the influence of rare huge temporal-difference targets relative to pure
squared loss.

The official [CleanRL DQN implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py)
is a compact executable reference; the original DeepMind DQN result should be
read from the paper because later repositories contain many extensions.

### What followed DQN, and which problem each change addresses

Several influential extensions are often bundled into “modern DQN”:

| Extension | Problem targeted | Core change |
| --- | --- | --- |
| Double DQN | maximization overestimation | select next action online, evaluate it with target network |
| prioritized replay | many stored transitions have little current learning signal | sample larger-error transitions more often and importance-correct |
| dueling network | action values share a large state-value component | estimate value and action advantage streams |
| multi-step return | reward propagates slowly through one-step backups | include several observed rewards before bootstrapping |
| distributional RL | an expectation hides outcome structure | predict a return distribution, not only its mean |
| noisy networks | hand-designed $\epsilon$ schedule | learn parameter-space exploration noise |

Rainbow combined six such components and tested their interaction on Atari.
These are alternatives within deep **discrete** value learning, not a universal
frontier for continuous robotics. As of 2026, value distributions,
representation learning, large-scale replay, and offline value learning remain
active ingredients, while continuous motor control usually uses an actor,
planning, or action-generation model to avoid enumerating actions.

## 3.10 Why continuous robot action changes the algorithm

For four discrete actions, computing $\max_a Q(s,a)$ is easy: evaluate all
four. A 14D continuous action space contains infinitely many possible vectors.
One cannot enumerate them.

Continuous-control families handle this differently:

- policy-gradient methods directly learn an action distribution;
- deterministic actor-critic methods learn an actor that approximately
  maximizes the critic;
- stochastic off-policy methods such as Soft Actor-Critic (SAC) optimize value
  plus entropy; and
- model-based methods plan through known or learned dynamics.

Microduck uses PPO because its simulator can generate large fresh on-policy
batches and its action is continuous. This is an engineering fit, not a theorem
that PPO is always the best robot algorithm.

## 3.11 The “deadly triad” warning

Learning can become unstable when three ingredients combine:

1. **function approximation**: a neural network generalizes across states;
2. **bootstrapping**: targets depend on current estimates; and
3. **off-policy learning**: training distribution differs from the target
   policy's distribution.

This combination is called the **deadly triad**. It does not mean every such
algorithm fails. It explains why replay buffers, target networks, double
critics, conservative objectives, and careful evaluation exist.

The danger can be understood as a feedback loop:

```text
slightly wrong value at an unsupported next action
        -> enters a bootstrap target
        -> shared network generalizes the error to other states
        -> greedy/off-policy selection seeks the inflated region
        -> new targets amplify the same error
```

A lower training loss does not prove the values are correct: predictions and
targets can move together. Held-out Bellman error is also incomplete because
the target contains estimates. Policy return, calibration against realized
returns, target statistics, Q-scale, and coverage all provide complementary
evidence.

## 3.12 Partial observability and memory

Bellman equations are written in terms of Markov state: all information needed
to predict the future distribution. A deployed robot usually receives an
observation $o_t$, not full state $s_t$. Hidden terrain compliance, actuator
temperature, backlash, payload, and delayed contacts can make the problem a
Partially Observable Markov Decision Process (POMDP).

Common responses are:

- stack recent observations;
- add an explicit estimator;
- use a recurrent network with internal memory;
- train an adaptation module that infers latent environment properties; or
- add sensors.

No optimizer can reconstruct information that leaves no trace in the available
history.

## 3.13 Exercises

1. With rewards $2,1,4$ and $\gamma=0.5$, compute the three-step return.
2. Calculate the TD error when $V(s)=3$, $r=-1$, $V(s')=5$, and
   $\gamma=0.9$. Does the value rise or fall for positive $\alpha$?
3. Explain the difference between a model, a value function, and a policy.
4. Why can Q-learning learn a greedy target policy while collecting with an
   $\epsilon$-greedy behavior policy?
5. Name the three parts of the deadly triad and one mitigation used by DQN.
6. Why is ordinary DQN inconvenient for a 14D continuous action vector?
7. Give one hidden physical property that can make a robot observation
   non-Markov and one way to expose or infer it.
8. For a two-state fixed policy with
   $P_\pi=\begin{bmatrix}0.5&0.5\\0&1\end{bmatrix}$,
   $r_\pi=[1,0]^T$, $\gamma=0.8$, and the second state terminal with value
   zero, solve the first state's Bellman equation without a matrix inverse.
9. If $\gamma=0.9$ and the initial maximum value error is 10, what upper bound
   does the contraction result give after 20 exact Bellman sweeps?
10. Compute the three-step target for rewards $[1,2,3]$, $\gamma=0.9$, and
    bootstrap value $V(S_{t+3})=4$. Identify the observed and estimated parts.
11. In the DQN pseudocode, explain why `gather` and `max` act on different
    action choices.
12. Write the Double DQN target using an online network to select and a target
    network to evaluate the next action. Which bias is it intended to reduce?
13. Why can a critic's training loss fall while its induced robot policy gets
    worse?
14. Choose between a return expectation and a return distribution for a robot
    whose two outcomes are “normal landing” and “rare destructive impact.”
    State what the richer prediction still cannot guarantee.

Continue with the [reinforcement-learning algorithm
map](04_rl_algorithm_families.md), where these dimensions become a practical
selection framework.

## 3.14 Folded solutions

<details>
<summary>Show answers to Section 3.13</summary>

1. The three-step return is
   $2+0.5(1)+0.5^2(4)=2+0.5+1=3.5$.
2. The TD error is
   $-1+0.9(5)-3=0.5$. Because it is positive, an update with positive learning
   rate raises $V(s)$.
3. A **model** predicts what the environment will do after an action. A
   **value function** predicts accumulated future reward. A **policy** selects
   or distributes actions. A system may learn any one, two, or all three.
4. Q-learning's target uses the greedy next value, $\max_a Q(s',a)$, regardless
   of which exploratory action the behavior policy actually selects next. The
   behavior policy supplies coverage; the backup defines the target policy.
5. The deadly triad is function approximation, bootstrapping, and off-policy
   learning. DQN mitigates instability with a replay buffer and a temporarily
   fixed target network; neither is a universal convergence guarantee.
6. A 14D continuous vector has infinitely many candidates, so ordinary DQN
   cannot enumerate every action to compute a maximum. Continuous actor-critic
   methods learn an actor that proposes the maximizing action instead.
7. Hidden ground friction is one example. Recent wheel/contact history or a
   learned adaptation latent can infer its effect; an additional force/slip
   sensor can expose it more directly. Motor temperature, payload, and backlash
   are other valid examples.
8. Let the first value be $v$. Its equation is
   $v=1+0.8(0.5v+0.5\times0)=1+0.4v$. Thus
   $0.6v=1$ and $v=5/3\approx1.667$. The self-loop repeatedly earns reward 1;
   transition to the terminal state ends future reward.
9. The bound is $10(0.9)^{20}\approx1.216$. It is an upper bound in the
   maximum norm, not a claim that every instance attains exactly that error.
10. The target is

    ```math
    1+0.9(2)+0.9^2(3)+0.9^3(4)
    =1+1.8+2.43+2.916=8.146.
    ```

    The first three terms are observed rewards; the final term estimates all
    value beyond the sampled three-step segment.
11. `gather` selects the value of the action that was actually stored in each
    replay transition, producing the prediction being trained. `max` chooses
    the greedy next action used by the Q-learning target. They refer to current
    behavior evidence and next-state target behavior respectively.
12. With online parameters $\theta$ and target parameters $\bar\theta$:

    ```math
    a^*=\arg\max_a Q_\theta(s',a),
    \qquad
    y=r+\gamma(1-d)Q_{\bar\theta}(s',a^*).
    ```

    Separating selection from evaluation reduces the positive bias caused by
    maximizing noisy estimates from the same estimator.
13. Bootstrapped targets contain critic predictions, so prediction and target
    can become mutually consistent at wrong values. Loss is measured on the
    replay distribution, while the actor may seek unsupported high-value
    actions elsewhere. Scale collapse, distribution shift, or exploitation of
    approximation error can therefore reduce loss and harm realized return.
14. A return distribution can expose probability mass on destructive impact
    that the same expected value would hide. It supports quantile or risk-
    sensitive decisions. It still depends on correct data coverage/modeling and
    cannot replace hard hardware protection or guarantee safety outside the
    evaluated distribution.

The arithmetic behind answers 1 and 2 can be checked with:

```python
discount = 0.5
three_step_return = 2 + discount * 1 + discount**2 * 4
td_error = -1 + 0.9 * 5 - 3
assert three_step_return == 3.5
assert td_error == 0.5
```

</details>
