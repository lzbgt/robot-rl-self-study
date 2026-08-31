# 3. Bellman Methods: From Tables to Deep Q-Learning

Before PPO, learn the simpler idea that organizes much of reinforcement
learning: the value of a decision is its immediate reward plus the value of
what follows. This is the **Bellman principle**.

The tabular setting in this chapter is deliberately small. When every state and
action can be listed, the algorithms expose their logic without a neural
network hiding mistakes.

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

## 3.8 SARSA and Q-learning

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

## 3.10 Why continuous robot action changes the algorithm

For four discrete actions, computing $\max_a Q(s,a)$ is easy: evaluate all
four. A 14D continuous action space contains infinitely many possible vectors.
One cannot enumerate them.

Continuous-control families handle this differently:

- policy-gradient methods directly learn an action distribution;
- deterministic actor-critic methods learn an actor that approximately
  maximizes the critic;
- stochastic off-policy methods such as SAC optimize value plus entropy; and
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

## 3.12 Partial observability and memory

Bellman equations are written in terms of Markov state: all information needed
to predict the future distribution. A deployed robot usually receives an
observation $o_t$, not full state $s_t$. Hidden terrain compliance, actuator
temperature, backlash, payload, and delayed contacts can make the problem a
Partially Observable MDP (POMDP).

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

Continue with [the RL algorithm map](04_rl_algorithm_families.md), where these
dimensions become a practical selection framework.

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

The arithmetic behind answers 1 and 2 can be checked with:

```python
discount = 0.5
three_step_return = 2 + discount * 1 + discount**2 * 4
td_error = -1 + 0.9 * 5 - 3
assert three_step_return == 3.5
assert td_error == 0.5
```

</details>
