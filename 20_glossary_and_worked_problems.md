# 20. Glossary and Worked Problems

Use this chapter as both a reference and a self-test. Try each problem before
reading its solution. The point is not to perform arithmetic quickly; it is to
connect the arithmetic to a real robot decision.

## 20.1 Plain-language glossary

### Action

The numeric decision produced by a policy at one time step. Microduck's main
policy outputs 14 relative joint-position targets.

### Actor

The neural network that selects actions. It is the part exported for
deployment.

### Actuator

Hardware that produces motion or force, plus its control electronics. A
Dynamixel XL330 servo is an actuator. In simulation, an actuator model converts
the policy's target into realistic force/torque behavior.

### Advantage

An estimate of how much better or worse a sampled action was than the critic
expected in that situation. Positive advantage encourages the action; negative
advantage discourages it.

### BAM

Better Actuator Models, the library/model used here to represent XL330 motor
electrical behavior, firmware position control, friction, voltage, load sag,
and delay more realistically than an ideal position actuator.

### Behavior cloning (BC)

Supervised learning from demonstration pairs: given the observation, predict
the demonstrator's action. BC does not require a reward or Bellman update and
is therefore imitation learning, not by itself reinforcement learning.

### Bellman backup

An update based on “immediate reward plus discounted value of what follows.”
Dynamic programming, TD learning, Q-learning, and many critics use different
forms of this recursive idea.

### Batch and minibatch

A batch is a collection of training samples processed together. PPO collects a
full rollout batch, shuffles it, and divides it into smaller minibatches for
optimization.

### Bootstrapping

Updating an estimate using another current estimate, such as training
$V(s_t)$ toward $r_t+\gamma V(s_{t+1})$. It enables learning before an episode
ends but can propagate estimation error.

### Checkpoint

A saved training state such as `model_2000.pt`. It normally includes actor,
critic, observation normalizers, optimizer state, and counters, so it can be
evaluated or resumed.

### Command

A requested behavior supplied to the policy as input, such as forward speed,
turn rate, or head pose. The command expresses intention; the action expresses
how the joints should move now.

### Coordinate frame

The origin and axis directions used to describe a vector. A velocity in the
robot body frame is different from the same three numbers in the world frame.

### Critic

The training-only neural network that estimates expected future return. It
helps estimate advantages and may use privileged simulator information.

### Curriculum

A schedule that changes difficulty, command range, reset distribution, reward
weight, or another task property after the policy learns an easier stage.

### Decimation

The number of physics substeps for which one policy action is held. A 5 ms
physics step with decimation 4 produces one policy decision every 20 ms.

### Degree of freedom (DoF)

One independent direction of movement. A simple hinge joint has one rotational
DoF. “14 actuated joints” means the policy controls 14 motorized DoFs.

### Domain randomization (DR)

Sampling plausible physical parameters and disturbances during training—such
as mass, friction, voltage, sensor bias, and delay—so the policy learns a
family of robots rather than one perfect simulation.

### Distribution shift

A difference between the data distribution used to train a model and the
situations encountered during evaluation/deployment. An imitation policy can
create states absent from expert data; an offline critic can be queried on
actions absent from its log.

### Diffusion policy

A generative policy that starts from noise and iteratively denoises a
conditioned action or action sequence. This can represent several distinct
valid action modes but requires multiple inference steps.

### Entropy

A measure of uncertainty/spread in the policy's action distribution. PPO uses
an entropy bonus to preserve exploration early in learning.

### Experience replay

A buffer of past transitions sampled for off-policy updates. Replay reuses
expensive data and decorrelates adjacent samples, while creating distribution
mismatch between older behavior and the current policy.

### Environment

The complete executable task: simulated scene, robot, actions, observations,
commands, rewards, reset events, terminations, and curricula.

### Episode

One sequence from reset until timeout or another termination. The main walking
episode lasts at most 20 simulated seconds.

### Epoch

One complete optimization pass over a collected rollout batch. Microduck's
main PPO configuration uses five epochs per rollout.

### GAE

Generalized Advantage Estimation, a method that combines several
temporal-difference errors. Its $\lambda$ setting trades bias against variance.

### Foundation policy / VLA

A broadly pretrained robot policy intended for adaptation across tasks or
embodiments. A Vision-Language-Action (VLA) model conditions actions on images
and language. Most VLA pretraining is demonstration-based imitation rather
than online RL.

### Flow matching

A generative method that learns a vector field transporting samples from a
simple noise distribution toward a data distribution. Some modern VLAs use it
to generate continuous action chunks.

### Gradient and backpropagation

A gradient says how a small change in each network parameter changes the loss.
Backpropagation applies the chain rule through all layers to compute those
gradients efficiently.

### Hyperparameter

A setting chosen by the engineer rather than learned as a network weight—for
example learning rate, PPO clip value, hidden-layer widths, $\gamma$, or
$\lambda$.

### Inference

Running a frozen actor to compute an action. No PPO update, critic training, or
backpropagation occurs during inference.

### Action chunk

A sequence of several future actions predicted at once. Chunks can improve
temporal coherence; executing too much of a chunk open loop can delay feedback
correction.

### KL divergence

Kullback–Leibler divergence, a measure of how one probability distribution
differs from another. PPO monitors approximate KL to detect a policy update
that moved too far from the rollout policy.

### LoRA

Low-Rank Adaptation, a parameter-efficient fine-tuning method that adds
trainable low-rank updates to frozen or mostly frozen weight matrices. It
reduces trainable parameter count but does not automatically make base-model
inference small or realtime.

### Markov Decision Process (MDP)

The mathematical model containing states, actions, transition probabilities,
rewards, and a discount factor.

### MJCF

MuJoCo's XML model format. It describes robot bodies, joints, actuators,
contacts, sensors, meshes, and related physics/model data.

### Normalization

Transforming each observation feature using learned statistics so values with
different units/scales are numerically comparable. The actor normalizer must be
included in deployed ONNX.

### Observation

The numeric information given to a policy. It may be only part of the complete
environment state.

### ONNX

Open Neural Network Exchange, a portable neural-network graph format used to
move the trained actor from Python/RSL-RL to the runtime.

### On-policy

Learning from experience generated by the current or very recent policy. PPO
collects a rollout, updates for a few epochs, discards it, then collects fresh
experience.

### Off-policy

Learning about a target policy from experience generated by a different
behavior policy. Replay-based SAC, TD3, and DQN are off-policy; offline RL is
the special fixed-dataset setting with no new corrective interaction.

### Partial observability / POMDP

A setting where the current observation omits information needed to predict
the future. History, recurrent state, explicit estimation, or an adaptation
latent can help when hidden properties leave observable traces.

### Policy

The decision rule mapping observations (and commands) to action probabilities
or inference actions. Here the policy is an MLP neural network.

### PPO

Proximal Policy Optimization, the on-policy actor-critic algorithm used here.
Its clipped objective reduces the incentive for one rollout to cause an
excessively large policy change.

### Privileged observation

Simulator information given to the training critic but withheld from the actor
because the real robot cannot measure it equivalently.

### System identification

Estimating physical model parameters from measured input/output trajectories.
For sim-to-real work, identify and validate a nominal model before choosing
randomization around its remaining uncertainty.

### Teacher-student learning

A training scheme in which a teacher with richer information or capability
provides targets for a deployable student. The student's sensor/input contract
still determines what it can reproduce.

### Reward and reward shaping

Reward is the scalar learning objective. Reward shaping adds measurable terms
that make useful intermediate progress learnable. Poor shaping can create
exploits or a different task.

### Rollout or trajectory

A time-ordered sequence of observations, actions, rewards, and termination
flags sampled by a policy. A rollout batch contains many short sequences from
parallel environments.

### RSL-RL

The training library that provides PPO, neural-network models, rollout storage,
normalization, checkpoints, and ONNX export integration in this stack.

### Sim-to-real

Transferring a policy learned in simulation to a physical robot. Success
requires consistent interfaces and a simulation distribution that covers the
important real dynamics and uncertainties.

### State

The full variables needed to describe the environment dynamics. The simulator
state is richer than the actor observation.

### Tensor

A multidimensional numeric array. Shapes matter: one actor observation is
`(61,)`; 4,096 observations form `(4096, 61)`; actor actions form
`(4096, 14)`.

### Termination

A condition that ends and resets an episode, such as timeout, fall, terrain
bounds, or nonfinite state.

### Value function

The critic's estimate of expected discounted future reward from a state or
critic observation.

### World model / model-based RL

A world model predicts future state or a learned latent representation under
actions. Model-based RL uses such predictions for planning or policy/value
learning. Longer imagination can amplify model error.

### WCET

Worst-Case Execution Time: the longest execution time under the defined
conditions. A realtime loop must budget a defensible upper bound or percentile
and deadline policy, not only average inference speed.

### Warp and MuJoCo Warp

Warp is NVIDIA's Python/CUDA kernel framework. MuJoCo Warp implements
GPU-parallel MuJoCo-style simulation so thousands of environments can step
together.

## 20.2 Worked problem 1: observation dimensions

### Background

The actor concatenates proprioception (the robot's own measured motion/state)
and commands. A wrong total means the network/runtime contract is broken.

### Problem

Add the main terms: angular velocity 3, projected gravity 3, joint positions
14, joint velocities 14, previous actions 14, and command block 13.

## 20.3 Worked problem 2: timestep and policy rate

### Background

Physics needs small steps for contacts. Neural policy inference can run less
often using action decimation.

### Problem

Physics timestep is 0.005 s and decimation is 4. Find the policy period and
frequency. How long is a 500-step video?

## 20.4 Worked problem 3: discounted return

### Background

Return adds current and future rewards, discounting each later step by
$\gamma$. Use a short made-up reward sequence to see the calculation.

### Problem

At time 0, the next three rewards are 1, 1, and 1. Let $\gamma=0.99$ and stop
after those three terms. What is $G_0$?

## 20.5 Worked problem 4: PPO rollout size

### Background

One PPO iteration gathers the same number of time steps from every parallel
environment.

### Problem

How many transitions are collected with 64 environments and 24 steps? With
4,096 environments and 24 steps?

## 20.6 Worked problem 5: Gaussian tracking reward

### Background

A common tracking term is $\exp(-(e/\sigma)^2)$, where $e$ is error and
$\sigma$ is the error scale that still matters.

### Problem

For head tracking with $\sigma=0.5$ rad, calculate the per-joint reward at
errors 0, 0.1, and 0.5 rad.

## 20.7 Worked problem 6: PPO clipping

### Background

The probability ratio is new policy probability divided by old policy
probability. For a positive advantage, PPO clips the benefit above 1.2 when
$\epsilon=0.2$.

### Problem

Old action probability is 0.10, new probability is 0.13, and advantage is +2.
Compare unclipped and clipped contributions.

## 20.8 Worked problem 7: penalty signs

### Background

Configuration weight and function output multiply. Always reason about the
product.

### Problem

Function A returns squared error `+0.4`. Function B returns negative absolute
error `-0.4`. What weight sign makes each a penalty?

## 20.9 Worked problem 8: why uniform sampling misses idle

### Background

A continuous uniform random variable can produce values arbitrarily close to
zero but has probability zero of producing one exact point.

### Problem

Why does sampling each velocity uniformly from a range fail to train the exact
deployment command `[0, 0, 0]`?

## 20.10 Worked problem 9: command versus action

### Background

The planner states intent; the motor policy chooses immediate control.

### Problem

Classify each value as command, actor observation, or actor action:

```text
desired forward speed
measured head joint angle
new left-knee position target
desired head yaw delta
previous actor output
```

## 20.11 Worked problem 10: obstacle-avoidance architecture

### Background

The renderer shows more than the actor receives. A visible object is not a
numeric observation.

### Problem

You add boxes to flat terrain and penalize collision, but keep the 61D actor
unchanged. What can the policy learn, and what is missing?

## 20.12 Worked problem 11: actor versus critic information

### Background

Asymmetric actor-critic training lets the critic see perfect simulator facts
while the actor remains deployable.

### Problem

Why can true base linear velocity improve the critic without being supplied to
the actor? What would go wrong if the actor depended on it?

## 20.13 Worked problem 12: design one controlled experiment

### Background

RL systems have many coupled variables. A good experiment changes one causal
hypothesis and preserves a baseline.

### Problem

Turn-in-place fails in 60% of trials. Propose a controlled experiment using
the command distribution rather than changing PPO and rewards together.

## 20.14 Worked problem 13: a Q-learning backup

### Background

Q-learning trains the current action value toward immediate reward plus the
best estimated next action value.

### Problem

The old value is $Q(s,a)=1$. A transition gives reward $-1$. At the next state,
the three action values are $[2,5,4]$. Let $\gamma=0.9$ and learning rate
$\alpha=0.2$. Compute the target, TD error, and updated Q-value.

## 20.15 Worked problem 14: entropy changes the SAC objective

### Background

SAC values expected reward plus $\alpha$ times policy entropy.

### Problem

At one state, policy A has expected immediate score 5.0 and entropy 0.1.
Policy B has score 4.8 and entropy 0.8. With $\alpha=0.5$, which has the larger
one-step soft objective?

## 20.16 Worked problem 15: behavior cloning averages modes

### Background

Squared-error regression predicts the conditional mean when data contains
uncertainty or several labels for the same input.

### Problem

At an identical-looking observation, half the demonstrations steer left with
action $-1$ and half steer right with action $+1$. What deterministic scalar
action minimizes mean-squared error? Why may it be dangerous?

## 20.17 Worked problem 16: world-model error over a horizon

### Background

Open-loop model predictions feed predicted state back into later predictions.

### Problem

A simple position model has a consistent 5 mm forward error per predicted step.
Ignoring all nonlinear amplification, what bias accumulates over 25 open-loop
steps? Why is this a lower-complexity estimate rather than a guarantee?

## 20.18 Worked problem 17: mean return hides tail failure

### Background

Expected return is not a per-episode safety guarantee.

### Problem

Four hardware evaluations have returns $[100,100,100,-100]$, where the last is
a damaging fall. Compute the mean, median, and non-fall success rate. What must
the report say?

## 20.19 Worked problem 18: stale asynchronous actions

### Background

An asynchronous VLA can predict chunks while another loop executes prior
actions. The relevant latency is observation age when an action is applied.

### Problem

Image capture/preprocessing takes 20 ms, queue wait 15 ms, inference 70 ms, and
transport 10 ms. How old is the observation when the first new action arrives?
How many 50 Hz policy periods is that?

## 20.20 Open exercises

1. Derive the maximum 20-second episode length in policy steps.
2. At iteration 1,000, what is the curriculum environment-step counter?
3. Draw the path from `head_pose` command to a reward and to the ONNX input.
4. Explain why body-pose UI controls do not prove the velocity checkpoint
   learned body-pose control.
5. Design a five-case evaluation battery for exact-zero standing.
6. List three real measurements needed before transferring a policy to a new
   actuator.
7. Find one pure helper and its regression test in `tests/`; explain why the
   pure form is easier to test than a full simulator wrapper.


Return to the [book index](README.md) and repeat any lab whose terms or
calculation are still unclear.

## 20.21 Folded solutions

Try each problem before opening its solution. The explanation—not only the final
number—is the part to compare with your reasoning.

<details>
<summary>Problem 1: observation dimensions — show solution</summary>

```math
3 + 3 + 14 + 14 + 14 + 13 = 61
```

The first 48 values are proprioception/history-like action context, and the
last 13 are intention. With 4,096 environments the actor input tensor is
`(4096, 61)`.

</details>

<details>
<summary>Problem 2: timestep and policy rate — show solution</summary>

```math
\Delta t_{policy}=0.005\times4=0.020\text{ s}
```

Frequency is the inverse of period:

```math
f=\frac{1}{0.020}=50\text{ Hz}
```

Five hundred policy steps last:

```math
500\times0.020=10\text{ s}
```

This is why a 500-step recorded rollout is a 10-second behavior sample.

</details>

<details>
<summary>Problem 3: discounted return — show solution</summary>

```math
G_0=1+0.99(1)+0.99^2(1)
```

```math
G_0=1+0.99+0.9801=2.9701
```

The third reward still matters strongly because it is only two steps away.
This truncated example is for arithmetic; real GAE also uses a critic estimate
beyond a rollout boundary when appropriate.

</details>

<details>
<summary>Problem 4: PPO rollout size — show solution</summary>

```math
64\times24=1,536
```

```math
4096\times24=98,304
```

The five-iteration smoke train therefore performs real PPO updates but with a
much smaller batch and duration than a full run. It validates plumbing, not
locomotion quality.

</details>

<details>
<summary>Problem 5: Gaussian tracking reward — show solution</summary>

At zero error:

```math
e^{-(0/0.5)^2}=1
```

At 0.1 rad:

```math
e^{-(0.1/0.5)^2}=e^{-0.04}\approx0.961
```

At 0.5 rad:

```math
e^{-(0.5/0.5)^2}=e^{-1}\approx0.368
```

This shows why a wide standard deviation gives useful gradient far away but
makes small errors relatively cheap.

</details>

<details>
<summary>Problem 6: PPO clipping — show solution</summary>

```math
r=0.13/0.10=1.3
```

Unclipped:

```math
1.3\times2=2.6
```

Clipped ratio is 1.2:

```math
1.2\times2=2.4
```

The objective uses the more conservative value 2.4, removing the incentive to
increase this already-favored sample further during that update.

</details>

<details>
<summary>Problem 7: penalty signs — show solution</summary>

For A, use a negative weight; for example:

```math
-1.0\times+0.4=-0.4
```

For B, use a positive weight:

```math
+1.0\times-0.4=-0.4
```

Using a negative weight for B gives `+0.4`, rewarding the violation. Confirm
all logged penalty contributions remain nonpositive.

</details>

<details>
<summary>Problem 8: why uniform sampling misses idle — show solution</summary>

The probability of any one exact real-number triple under continuous sampling
is zero. “Very small” is also behaviorally different from an exact idle flag.
The environment therefore needs an explicit zero-command bucket with nonzero
probability.

</details>

<details>
<summary>Problem 9: command versus action — show solution</summary>

```text
desired forward speed            command (also included in observation)
measured head joint angle        proprioceptive observation
new left-knee position target    action after scale/offset mapping
desired head yaw delta           command (also included in observation)
previous actor output            observation/history context
```

Commands become part of the actor input; they are not outputs chosen by the
motor policy.

</details>

<details>
<summary>Problem 10: obstacle-avoidance architecture — show solution</summary>

It may learn post-contact reactions or a generally conservative gait. It
cannot deliberately steer around an unseen box before contact because no
pre-contact obstacle information reaches the actor. Add a local
perception/planner that changes twist commands, or create a versioned
exteroceptive policy input and matching runtime.

</details>

<details>
<summary>Problem 11: actor versus critic information — show solution</summary>

The critic is discarded after training, so privileged velocity can improve its
value/advantage estimates without becoming a runtime input. If the actor used
perfect velocity but the real robot could not reproduce it with matching
noise, delay, frame, and accuracy, deployment observations would come from a
different distribution and behavior could fail.

</details>

<details>
<summary>Problem 12: design one controlled experiment — show solution</summary>

```text
hypothesis:
  turn-in-place examples are too rare

change:
  increase TURN_IN_PLACE_FRACTION from 0.15 to 0.25 only

preserve:
  task, robot, rewards, PPO config, training budget, evaluation commands

verification:
  config test -> CPU suite -> five-iteration smoke -> equal-budget train

metrics:
  turn success, yaw error, translation drift, fall rate, forward tracking

decision:
  keep only if turn improvement is repeatable and forward behavior remains
  inside its acceptance envelope
```

Changing entropy, yaw reward, command range, and sample fraction together
would prevent a causal conclusion.

</details>

<details>
<summary>Problem 13: a Q-learning backup — show solution</summary>

The greedy next value is 5:

```math
y=-1+0.9(5)=3.5.
```

The TD error is target minus old estimate:

```math
\delta=3.5-1=2.5.
```

Update only partway using $\alpha$:

```math
Q_{new}=1+0.2(2.5)=1.5.
```

The estimate rises because this transition was better than the old prediction.

</details>

<details>
<summary>Problem 14: entropy changes the SAC objective — show solution</summary>

```math
A: 5.0+0.5(0.1)=5.05,
```

```math
B: 4.8+0.5(0.8)=5.20.
```

The soft objective prefers B despite slightly lower expected task score because
it retains more action diversity. This does not mean deployment must sample
unsafe random actions; the entropy term shapes training, and execution policy/
safety still must be specified.

</details>

<details>
<summary>Problem 15: behavior cloning averages modes — show solution</summary>

For prediction $x$:

```math
L(x)=\tfrac12(x+1)^2+\tfrac12(x-1)^2=x^2+1.
```

The derivative $2x$ is zero at $x=0$, the mean. If left and right both avoid an
obstacle, the average can drive straight into it. More context or a multimodal
policy must distinguish/represent the alternatives.

</details>

<details>
<summary>Problem 16: world-model error over a horizon — show solution</summary>

```math
25\times5\text{ mm}=125\text{ mm}=0.125\text{ m}.
```

Real error need not add linearly. A biased position changes future contact and
policy inputs, which can amplify, cancel, or redirect error. Receding-horizon
replanning replaces imagined state with observation before the full horizon.

</details>

<details>
<summary>Problem 17: mean return hides tail failure — show solution</summary>

```math
\text{mean}=\frac{100+100+100-100}{4}=50.
```

The sorted middle pair is $100,100$, so median is 100. Success is $3/4=75\%$.
The attractive median hides one severe failure; even the mean does not label
its consequence. Report every trial, success/failure category, damage/safety
intervention, and uncertainty. Four trials are too few for a reliable tail-risk
estimate.

</details>

<details>
<summary>Problem 18: stale asynchronous actions — show solution</summary>

```math
20+15+70+10=115\text{ ms}.
```

A 50 Hz period is 20 ms:

```math
115/20=5.75\text{ policy periods}.
```

The model acts from information nearly six local-control ticks old. A
manipulator may tolerate that under slow receding-horizon execution; a disturbed
balance loop likely cannot. Measure age/jitter and keep fast stabilization local.

</details>

<details>
<summary>Open exercises 1–7 — show solutions and reference checks</summary>

1. At 50 Hz, a 20-second episode contains
   $20\text{ s}\times50\text{ Hz}=1{,}000$ policy steps.
2. Curriculum steps count environment steps per rollout fragment, so iteration
   1,000 corresponds to $1{,}000\times24=24{,}000$ environment steps.
3. `CommandManager` samples `head_pose`; `generated_commands` appends its four
   values after the 48 proprioceptive values and 3D twist command, so they
   occupy actor indices 51–54. The `head_pose_tracking` reward reads the same
   command and the four physical head/neck joint positions. Export rebuilds
   this observation order, applies the saved normalizer, and embeds the actor
   in ONNX. This shared source and ordering are why command, reward, and
   deployment remain consistent.
4. Interface presence is not a learned objective. The six body-pose inputs and
   UI control can vary, but the main velocity recipe gives
   `body_pose_tracking` weight zero. A checkpoint might ignore those inputs or
   respond only through accidental correlations; testable command following
   requires nonzero task data, reward, and held-out evaluation.
5. A useful exact-zero battery includes: nominal quiet stand; randomized
   initial tilt; a bounded push followed by recovery; held-out friction and
   actuator-delay extremes; and a long-duration hold that exposes drift or
   heating. For every case report fall rate, tilt, position drift, action
   jitter, saturation, and recovery time across seeds—not just reward.
6. At minimum measure command-to-motion delay/bandwidth, torque or force versus
   command under load, and friction/deadband/backlash. Battery-voltage response,
   saturation, speed limits, thermal behavior, and sensor noise are also part
   of a defensible actuator model.
7. One example is the following pure helper and its focused test:

   ```text
   mdp.py::spin_wheel_differential_from_values
   tests/test_spin.py::test_wheel_differential_from_values_is_pure
   ```

   The helper receives tensors and constants directly, so the test can cover
   its sign, scaling, and gating without constructing MuJoCo, managers,
   sensors, or GPU state. The simulator wrapper can then be tested separately
   for correct data extraction.

</details>
