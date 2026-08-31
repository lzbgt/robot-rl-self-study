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

## 1.10 Check your understanding

1. Why is the 3D velocity command part of the observation rather than part of
   the action?
2. If an obstacle is visible in the renderer but absent from the actor
   observation, can the policy deliberately avoid it?
3. What is the difference between a high instantaneous reward and a high
   expected return?
4. Why can a reward function produce behavior its author did not intend?
5. Which state should terminate a walking episode but be a valid initial state
   for stand-up training?

Continue with the
[math and neural-network toolkit](02_math_and_neural_network_toolkit.md).

## 1.11 Folded solutions

<details>
<summary>Show answers to Section 1.10</summary>

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

</details>
