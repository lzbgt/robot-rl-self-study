# 1. Reinforcement Learning Foundations

## 1.1 The central idea

Reinforcement learning trains a decision-making rule by letting it interact
with an environment. The learner is not given the correct motor command for
every possible situation. Instead, it tries actions, observes consequences,
and adjusts its policy so that useful outcomes become more likely.

For Microduck, one interaction looks like this:

```text
simulated robot and command
        |
        | observation: orientation, joint state, previous action, command
        v
neural-network policy
        |
        | action: 14 normalized joint-position targets
        v
Better Actuator Models (BAM) motor model + MuJoCo physics
        |
        | next robot state, reward, termination flag
        +----------------------------------------------> repeat
```

An **actuator** is the component that turns a command into physical force or
motion; here it is a Dynamixel servo represented by Better Actuator Models
(BAM). During training, thousands of simulated robots execute this loop in
parallel. During deployment, one real robot executes only the learned policy;
Proximal Policy Optimization (PPO) and the critic are no longer updating
weights.

## 1.2 Reinforcement learning (RL) compared with nearby ideas

Supervised learning starts with labeled examples such as “for this image, the
answer is cat.” RL usually has no label saying “the correct 14 motor targets
are these.” It has an objective, such as tracking velocity while staying
upright, and must discover a sequence of actions.

Classical control starts from an explicit model or error law, such as a
proportional–integral–derivative (PID) controller. RL can learn a nonlinear
controller where writing the full law by hand is difficult. Classical
controllers remain valuable as baselines, safety layers, and inner loops.

Planning chooses goals or sequences over a longer horizon. A walking policy is
normally below planning: a planner asks for a local velocity, and the policy
produces stable joint targets. Asking a cloud model for raw motor torques would
collapse these layers and remove the timing and safety boundary.

### Why reinforcement learning became a separate field

The distinctive difficulty is **credit assignment**: deciding which earlier
actions deserve credit or blame for a later result. A supervised label can
say that one image is a cat. It usually cannot say which ankle target 1.3
seconds ago prevented a fall after three later contacts. The action also
changes what data becomes available next, so examples are neither independent
nor drawn from a fixed distribution.

The modern formal thread has several roots rather than one birthday:

- Richard Bellman's 1950s dynamic programming made a long decision problem
  recursive: solve the future subproblem, then choose the present action.
- Trial-and-error psychology and optimal-control research contributed ideas
  about reward, feedback, and sequential behavior.
- Temporal-difference learning joined sampled experience with Bellman-style
  bootstrapping; the 1983 actor-critic work of Barto, Sutton, and Anderson is
  an early recognizable learning controller.
- Watkins's Q-learning, Williams's REINFORCE estimator, neural function
  approximation, and later deep-learning scale produced today's major
  value-learning and policy-gradient branches.

The [annotated primary-source index](SOURCES.md#foundations-and-policy-optimization)
links the original or stable records. This history matters because current
methods recombine old answers to four questions: what should be predicted,
how should delayed credit be assigned, which data may be reused, and how far
may the policy change at once?

## 1.3 A small guide to the notation

Math notation is compact language. Read the symbols rather than trying to
memorize the visual shape:

| Notation | Read it as | Example meaning here |
| --- | --- | --- |
| $x_t$ | “x at time step t” | robot state now |
| $x_{t+1}$ | “x at the next step” | robot state 20 ms later |
| $a \in \mathcal{A}$ | “a is an element of set A” | this action is allowed |
| $p(x\mid y)$ | “probability of x given y” | next-state probability given state/action |
| $\sum_i x_i$ | “sum x over index i” | add reward terms or future rewards |
| $\mathbb{E}[X]$ | “expected or long-run average X” | average return across sampled rollouts |
| $x\sim p$ | “sample x from distribution p” | sample a randomized mass |
| $\theta$ | “theta,” a parameter collection | all actor network weights |
| $\approx$ | approximately equal | critic estimate versus unknown true value |

A **vector** is an ordered list of numbers. A **tensor** is the general
multidimensional form used by PyTorch: a scalar is 0D, a vector is 1D, a table
or batch is 2D, and images/batched histories use more dimensions. In a batch
of 4,096 actor observations, the observation tensor has shape `(4096, 61)`.

## 1.4 The Markov Decision Process

The standard mathematical model is a Markov Decision Process (MDP):

```math
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
```

| Symbol | General meaning | Microduck example |
| --- | --- | --- |
| $s_t \in \mathcal{S}$ | complete state at time $t$ | all simulated positions, velocities, contacts, and randomized physics |
| $a_t \in \mathcal{A}$ | action selected at time $t$ | 14 joint-position commands |
| $P(s_{t+1}\mid s_t,a_t)$ | transition dynamics | MuJoCo plus the BAM actuator, contacts, delay, and noise |
| $R(s_t,a_t,s_{t+1})$ | reward | velocity tracking, uprightness, foot behavior, and regularization |
| $\gamma$ | discount factor | `0.99` in the main PPO configuration |

The Markov property says the current state contains enough information to
model the distribution of the next state. The policy does not receive the
complete simulator state, however. It receives an observation $o_t$.

More precisely, Markov means that after conditioning on the present state and
action, the earlier history adds no predictive information:

```math
P(S_{t+1}\mid S_0,A_0,\ldots,S_t,A_t)
=P(S_{t+1}\mid S_t,A_t).
```

This does **not** mean the next state is deterministic. A randomized push,
noisy sensor, or uncertain contact can still produce a distribution of next
states. It means that a correctly defined state carries the information needed
to specify that distribution.

The main actor observation has 61 values:

```text
base angular velocity          3
projected gravity              3
joint position                14
joint velocity                14
previous action               14
twist command                  3
head-pose command              4
body-pose command              6
                              --
total                         61
```

This makes the real problem partially observable. For example, the actor does
not directly receive true base linear velocity. Previous actions and physical
dynamics provide some short-term context without making the policy recurrent.

The formal extension is a Partially Observable Markov Decision Process
(POMDP). It adds an observation model $O(o_t\mid s_t)$: the hidden state emits
what the agent can measure. An ideal decision maker could maintain a **belief
state**—a probability distribution over hidden states given the history:

```math
b_t(s)=P(S_t=s\mid o_0,a_0,\ldots,o_t).
```

A filter, recurrent network, or history encoder approximates this idea. Merely
renaming $o_t$ as “state” does not restore missing velocity, friction, delay,
or terrain information.

## 1.5 Policy, trajectory, and return

A policy is a conditional distribution over actions:

```math
\pi_\theta(a_t \mid o_t)
```

$\theta$ represents all neural-network weights. The vertical bar means
“conditioned on,” so the formula reads: “the probability of action $a_t$ when
the observation is $o_t$.” During training, the actor
defines a Gaussian distribution so it can explore nearby actions. A rollout is
a trajectory:

```math
\tau = (o_0,a_0,r_0,o_1,a_1,r_1,\ldots)
```

The discounted return from time $t$ is:

```math
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
```

The capital sigma means “add all the following terms.” With $\gamma=0.99$, a
reward one step later counts almost as much as a reward
now, while very distant rewards gradually matter less. Discounting helps make
long sums finite and expresses a preference for reliable, sooner outcomes.

The same definition gives a recurrence used in real rollout code. Peel off
the first reward:

```math
\begin{aligned}
G_t
&=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots\\
&=r_t+\gamma(r_{t+1}+\gamma r_{t+2}+\cdots)\\
&=r_t+\gamma G_{t+1}.
\end{aligned}
```

This is not a new assumption; it is the same sum regrouped. Implementations
therefore compute all returns in one backward pass:

```python
future = bootstrap_value
for t in reversed(range(len(rewards))):
    future = rewards[t] + gamma * future
    returns[t] = future
```

Run [`examples/returns_and_occupancy.py`](examples/returns_and_occupancy.py)
to see this recurrence produce $[3.5,3.0,4.0]$ from rewards $[2,1,4]$ at
$\gamma=0.5$.

### Discounting, horizon, and physical time

If reward is a constant 1 forever and $0\leq\gamma<1$, the return is a
geometric series:

```math
1+\gamma+\gamma^2+\cdots=\frac{1}{1-\gamma}.
```

At $\gamma=0.99$, the total is 100. This motivates the rough **effective
horizon** $1/(1-\gamma)$: about 100 policy steps. At 50 Hz that is about two
seconds. It is only a rule of thumb—the weight after $k$ steps is exactly
$\gamma^k$—but it exposes a common error: copying the same $\gamma$ between
20 Hz and 200 Hz changes the amount of physical future being valued.

To preserve an approximate continuous-time discount time constant $\tau$, use

```math
\gamma=e^{-\Delta t/\tau},
```

where $\Delta t$ is the policy period. If $\tau=2$ s and $\Delta t=0.02$ s,
$\gamma\approx0.99005$. This derivation explains why `0.99` is plausible for
a 50 Hz controller; it does not prove it is optimal for every reward design.

The learning objective is to find weights with high expected return:

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}
\gamma^t r_t\right]
```

Read this as: “adjust network weights $\theta$ to maximize the average
discounted reward over trajectories sampled from the policy.” “Expected” is
important. Initial poses, commands, contact events, actuator
parameters, and sampled actions vary. A policy should work across that
distribution, not memorize one rollout.

### The policy changes its own training distribution

Let $d_t^\pi(s)=P(S_t=s\mid\pi)$ be the probability of visiting state $s$ at
time $t$ under policy $\pi$. A discounted state-occupancy distribution is

```math
d_\gamma^\pi(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t d_t^\pi(s).
```

The leading factor normalizes the weights to sum to one. With this notation,
the continuing objective can be viewed as reward averaged over the states and
actions the policy itself visits:

```math
J(\pi)=\frac{1}{1-\gamma}
\mathbb{E}_{s\sim d_\gamma^\pi,\,a\sim\pi(\cdot\mid s)}[r(s,a)].
```

This equation explains several practical facts at once:

- changing the policy changes the data distribution;
- a rare recovery state supplies almost no gradient if resets and failures
  almost never reach it;
- behavior cloning can fail after one mistake because it leaves the expert's
  state distribution; and
- reverse-curriculum starts work by deliberately adding occupancy near the
  missing part of a maneuver.

The occupancy demo in the runnable example estimates this distribution in a
two-state balance process. It is a small programmatic bridge between the
probability definition and the state histograms used in robot evaluation.

## 1.6 Reward is a specification, not a feeling

The robot has no concept of “walk naturally” unless the measurable reward and
termination conditions make natural walking the best available strategy. If a
policy can earn more reward by shuffling, leaning, falling into a stable pose,
or repeating a violent motion, PPO may discover exactly that.

A typical per-step reward is a weighted sum:

```math
r_t = \sum_i w_i f_i(s_t,a_t,s_{t+1})
```

For example, a velocity-tracking term may be positive while action-rate and
foot-slip terms are costs. The sign convention is a software contract, not a
mathematical universal: some functions return a nonnegative cost and therefore
need a negative weight; some Microduck penalty helpers already return a
negative value and therefore need a positive weight. The observable invariant
is that every logged penalty contribution should be at most zero.

The most common beginner mistake is to optimize a proxy and then judge the
policy by the intended goal. Judge both:

- Does the logged reward term improve?
- Does the rollout visibly perform the desired behavior?
- Does it work across commands, resets, and randomized conditions?

### Return-equivalent reward shaping—and its boundary

A shaping signal can make learning easier without changing which policy is
optimal, but only under specific conditions. Potential-based shaping adds the
difference of a scalar potential $\Phi$ between consecutive states:

```math
F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t),
```

```math
r'_t=r_t+F(s_t,s_{t+1}).
```

Summing its discounted contribution causes the intermediate potential terms
to cancel:

```math
\sum_{t=0}^{T-1}\gamma^tF_t
=-\Phi(s_0)+\gamma^T\Phi(s_T).
```

For an infinite discounted task with bounded $\Phi$, the last term vanishes.
For a fixed start state, every policy's return shifts by the same constant, so
the optimal policy is preserved. In finite episodes, terminal potentials and
truncation handling must be defined carefully.

This gives a mathematical reason that “progress since last step” is often less
farmable than paying for merely *being* in an intermediate pose. Arbitrary
weighted bonuses do not receive this guarantee. The policy-invariance result
originated in Ng, Harada, and Russell's 1999 reward-shaping analysis, linked in
the source index.

### Reward, constraint, and acceptance test are different

Keep three mechanisms separate:

| Mechanism | Role | Example |
| --- | --- | --- |
| reward | ranks behavior during optimization | track commanded velocity |
| constraint/supervisor | restricts allowed behavior | motor-current and tilt cutoff |
| acceptance metric | decides whether an artifact may advance | zero falls in a fixed test battery |

A large negative reward for overcurrent is still something the optimizer can
trade against positive reward. It cannot replace a current limiter. Conversely,
a hard limiter alone gives no learning gradient until it activates.

## 1.7 Commands make one policy represent many behaviors

The velocity policy does not learn one fixed walk. A command $c_t$ is part of
the observation, so the policy is more accurately written as:

```math
\pi_\theta(a_t \mid o_t, c_t)
```

Training samples many forward, lateral, turn, stand, and head-pose commands.
Playback also resamples commands, which can look like a “random walk.” The
actions are not random at deployment: an application can deliberately set the
command.

This distinction separates skill from intention:

```text
planner or operator: where/why to move
          |
          | local velocity and pose command
          v
RL locomotion policy: how to move the joints stably
```

The current velocity task has no global waypoint, obstacle observation, or
navigation memory. Those belong in an upper layer unless a new observation
contract and task are intentionally designed.

## 1.8 Episodes and termination

Training divides experience into episodes. An episode resets when it reaches a
time limit or a terminal failure condition. Resetting gives the policy fresh
initial states and bounds the return.

Termination design changes the problem. Ending immediately on a fall teaches
the locomotion policy to avoid falling. A stand-up policy must not terminate
merely because it begins on the ground; the ground state is its task. This is
why task templates differ even when they use the same robot.

Time limits need special care. A **termination** says the modeled task reached
a true absorbing end such as failure or success. A **truncation** says data
collection stopped for an external reason such as a fixed horizon. For a true
terminal transition, future value is zero. For a time-limit truncation in a
continuing task, a critic should usually bootstrap the value that would have
followed. Treating every time limit as death systematically undervalues states
near the horizon.

## 1.9 On-policy learning

PPO is on-policy: each update uses recent data sampled by the current or very
recent policy. After several optimization epochs, that rollout is discarded
and new behavior is collected.

Consequences:

- simulation throughput matters;
- a rare state may receive too little training data;
- curricula and reset distributions control what data the policy sees; and
- old demonstrations or checkpoints are not automatically replayed as an
  off-policy dataset.

Reverse-curriculum resets are useful when a maneuver learns its beginning but
never reaches its final state: starting some episodes near completion creates
on-policy data at the missing frontier.

## 1.10 A complete two-state derivation

Consider a toy balance controller with states `upright` ($U$) and `fallen`
($F$). `fallen` is terminal. In $U$, action `careful` gives reward 1 and stays
upright with probability 0.9; action `fast` gives reward 2 but stays upright
with probability 0.5. Let $\gamma=0.9$.

If the policy always chooses `careful`, Bellman self-consistency gives

```math
V^{careful}(U)=1+0.9[0.9V^{careful}(U)+0.1V(F)].
```

Since $V(F)=0$:

```math
V^{careful}(U)=1+0.81V^{careful}(U),
```

```math
(1-0.81)V^{careful}(U)=1,
```

```math
V^{careful}(U)=\frac{1}{0.19}\approx5.263.
```

For always `fast`:

```math
V^{fast}(U)=2+0.9[0.5V^{fast}(U)],
```

```math
V^{fast}(U)=\frac{2}{1-0.45}\approx3.636.
```

The larger immediate reward loses because it causes earlier termination.
This tiny example contains the full sequential problem: transition dynamics,
discounted future, policy-dependent occupancy, and a reward that cannot be
judged one step at a time.

An implementation does not solve these two equations symbolically for a
continuous biped. It samples transitions and trains a critic to approximate
the same self-consistency relationship. Chapter 3 turns that statement into
Bellman and temporal-difference updates; Chapter 5 turns it into PPO.

## 1.11 From definition to environment code

Every mathematical object should have an identifiable software owner:

| Mathematical object | Typical environment implementation | Evidence to inspect |
| --- | --- | --- |
| $\mathcal{S}$ | simulator state and hidden randomized parameters | model, scene, event configuration |
| $o_t$ | observation-term functions and concatenation order | resolved shape, units, ranges |
| $\mathcal{A}$ | action transform and limits | normalized-to-physical mapping |
| $P$ | physics, actuator, delay, contact, randomization | timestep and identified parameters |
| $R$ | reward-term functions and weights | per-term logs and sign tests |
| initial distribution | reset events | spawn histograms |
| terminal/truncated flags | termination manager and time limit | bootstrapping behavior |
| $\pi_\theta$ | actor network and action distribution | checkpoint plus resolved config |

Minimal environment pseudocode makes the mapping explicit:

```python
def step(normalized_action):
    physical_target = home_pose + action_scale * normalized_action
    for _ in range(decimation):
        torque = actuator_model(physical_target, measured_joint_state)
        simulator.integrate(torque, physics_dt)

    observation = build_actor_observation(simulator, command)
    reward_terms = compute_reward_terms(simulator, command, normalized_action)
    terminated = unsafe_tilt_or_forbidden_contact(simulator)
    truncated = episode_steps >= time_limit_steps
    return observation, sum(reward_terms), terminated, truncated
```

This is not Microduck's exact application programming interface (API). It is a
reading guide: if one line has no concrete counterpart in a project, the task
definition is incomplete or hidden elsewhere.

## 1.12 Check your understanding

1. Why is the 3D velocity command part of the observation rather than part of
   the action?
2. If an obstacle is visible in the renderer but absent from the actor
   observation, can the policy deliberately avoid it?
3. What is the difference between a high instantaneous reward and a high
   expected return?
4. Why can a reward function produce behavior its author did not intend?
5. Which state should terminate a walking episode but be a valid initial state
   for stand-up training?
6. At 50 Hz, roughly how much physical time does the heuristic horizon
   $1/(1-\gamma)$ represent for $\gamma=0.99$?
7. Derive $G_t=r_t+\gamma G_{t+1}$ from the discounted sum in one line of
   algebra. Why is this useful in code?
8. A continuing task is cut into 10-second log files. Is the file boundary a
   termination or a truncation? Should a value target normally bootstrap?
9. Why does a policy change the distribution of observations used to train
   its next update?
10. Design a potential $\Phi(s)$ for standing upright. State one way an
    incorrectly handled terminal potential could still change behavior.
11. In the two-state example, find the immediate reward $r_{fast}$ at which
    the always-fast and always-careful policies have equal value.
12. Map “joint command arrives one cycle late” to the MDP/POMDP vocabulary.

Continue with the
[math and neural-network toolkit](02_math_and_neural_network_toolkit.md).

## 1.13 Folded solutions

<details>
<summary>Show answers to Section 1.12</summary>

1. The velocity command describes the behavior requested *of* the policy, so
   it must be an input. The action is the policy's response: fourteen immediate
   joint targets. If forward velocity were an action, the network would merely
   repeat the request instead of deciding how to move the mechanism.
2. No—not deliberately before contact. Pixels in a viewer are not numbers in
   the actor observation. The policy could learn a generic cautious gait or a
   post-contact reaction, but pre-contact avoidance needs an obstacle feature,
   map, range sensor, or an upper planner that changes the local command.
3. Instantaneous reward evaluates one transition. Expected return averages a
   discounted sequence of rewards over trajectories and uncertainty. A policy
   can accept one small immediate cost when it produces a much better future,
   or exploit a repeated small reward whose long-run sum is large.
4. A reward is a proxy written in code. Any unspecified orientation, contact,
   timing, or terminal condition may provide a cheaper way to score. The policy
   follows the mathematical signal, not the author's unstated visual intent.
5. A fallen body should normally terminate walking because continued fallen
   transitions are outside the skill. The same fallen pose is a legitimate
   reset state for stand-up, where recovery from it is the task.
6. The heuristic horizon is $1/(1-0.99)=100$ steps. At 50 steps per second,
   that is roughly two seconds. The exact weight two seconds ahead is
   $0.99^{100}\approx0.366$, so “horizon” is not a hard cutoff.
7. Factor one $\gamma$ from every term after $r_t$:

   ```math
   G_t=r_t+\gamma(r_{t+1}+\gamma r_{t+2}+\cdots)
      =r_t+\gamma G_{t+1}.
   ```

   A backward loop then obtains every return in linear time without repeatedly
   summing the same suffix of rewards.
8. It is a truncation: the underlying control task could continue. The critic
   should normally bootstrap from the final observation. A true fall or task
   completion would instead set future value to zero.
9. Actions alter future states. A stable policy spends more samples upright;
   an unstable one creates more fall/recovery observations. Updating the
   policy therefore changes the occupancy distribution that generates the
   next training batch.
10. One potential is upright progress, such as
    $\Phi(s)=k\,\hat z_{body}\cdot\hat z_{world}$, possibly combined with a
    bounded height term. If a failure terminal is assigned a nonzero potential
    but the final $\gamma^T\Phi(s_T)$ term is silently dropped or treated
    inconsistently, terminating can receive an unintended bonus or penalty and
    policy invariance no longer follows from the telescoping argument.
11. Always-fast has value $r_{fast}/(1-0.9\times0.5)$.
    Set it equal to $1/(1-0.9\times0.9)$:

    ```math
    \frac{r_{fast}}{0.55}=\frac{1}{0.19}
    \quad\Rightarrow\quad
    r_{fast}=\frac{0.55}{0.19}\approx2.895.
    ```

    Below about 2.895, careful has greater return; above it, the immediate
    reward compensates for the greater fall probability under this model.
12. The delay belongs to transition dynamics and observation history. If the
    current state includes the queued prior action, the augmented process can
    be Markov. If the actor cannot observe the queue/delay, it faces partial
    observability. Adding previous action and history can expose evidence, while
    randomizing delay trains robustness rather than revealing its exact value.

</details>
