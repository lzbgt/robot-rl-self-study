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

### Better Actuator Models (BAM)

The actuator library/model used here to represent XL330 motor electrical
behavior, firmware position control, friction, voltage, load sag, and delay
more realistically than an ideal position actuator.

### Behavior cloning (BC)

Supervised learning from demonstration pairs: given the observation, predict
the demonstrator's action. BC does not require a reward or Bellman update and
is therefore imitation learning, not by itself reinforcement learning.

### Bellman backup

An update based on “immediate reward plus discounted value of what follows.”
Dynamic programming, temporal-difference (TD) learning, Q-learning, and many
critics use different forms of this recursive idea.

### Batch and minibatch

A batch is a collection of training samples processed together. Proximal
Policy Optimization (PPO) collects a full rollout batch, shuffles it, and
divides it into smaller minibatches for optimization.

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

### Generalized Advantage Estimation (GAE)

Generalized Advantage Estimation, a method that combines several
temporal-difference errors. Its $\lambda$ setting trades bias against variance.

### Foundation policy / vision-language-action (VLA)

A broadly pretrained robot policy intended for adaptation across tasks or
embodiments. A VLA model conditions actions on images and language. Most VLA
pretraining is demonstration-based imitation rather than online reinforcement
learning (RL).

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

### Kullback–Leibler (KL) divergence

Kullback–Leibler divergence, a measure of how one probability distribution
differs from another. PPO monitors approximate KL to detect a policy update
that moved too far from the rollout policy.

### Low-Rank Adaptation (LoRA)

Low-Rank Adaptation, a parameter-efficient fine-tuning method that adds
trainable low-rank updates to frozen or mostly frozen weight matrices. It
reduces trainable parameter count but does not automatically make base-model
inference small or realtime.

### Markov Decision Process (MDP)

The mathematical model containing states, actions, transition probabilities,
rewards, and a discount factor.

### MuJoCo Extensible Markup Language (XML) model format (MJCF)

It describes robot bodies, joints, actuators, contacts, sensors, meshes, and
related physics/model data.

### Normalization

Transforming each observation feature using learned statistics so values with
different units/scales are numerically comparable. The actor normalizer must be
included in the deployed export graph.

### Observation

The numeric information given to a policy. It may be only part of the complete
environment state.

### Open Neural Network Exchange (ONNX)

A portable neural-network graph format used to move the trained actor from
Python training to the runtime.

### On-policy

Learning from experience generated by the current or very recent policy. PPO
collects a rollout, updates for a few epochs, discards it, then collects fresh
experience.

### Off-policy

Learning about a target policy from experience generated by a different
behavior policy. Replay-based Soft Actor-Critic (SAC), Twin Delayed Deep
Deterministic Policy Gradient (TD3), and a Deep Q-Network (DQN) are off-policy;
offline RL is the special fixed-dataset setting with no new corrective
interaction.

### Partial observability / Partially Observable Markov Decision Process (POMDP)

A setting where the current observation omits information needed to predict
the future. History, recurrent state, explicit estimation, or an adaptation
latent can help when hidden properties leave observable traces.

### Policy

The decision rule mapping observations (and commands) to action probabilities
or inference actions. Here the policy is a multilayer perceptron (MLP) neural
network.

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

### Robotic Systems Lab reinforcement learning (RSL-RL)

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

### Worst-case execution time (WCET)

Worst-Case Execution Time: the longest execution time under the defined
conditions. A realtime loop must budget a defensible upper bound or percentile
and deadline policy, not only average inference speed.

### Warp and MuJoCo Warp

Warp is NVIDIA's Python/Compute Unified Device Architecture (CUDA) kernel
framework. MuJoCo Warp implements graphics processing unit (GPU)-parallel
MuJoCo-style simulation so thousands of environments can step together.

### Action space

The set of actions a policy is allowed to produce. It includes dimension,
bounds, and semantics—not just tensor shape. Joint targets, torques,
end-effector deltas, and base velocities are different action spaces.

### Actor-critic

An architecture with an actor choosing actions and a critic estimating value.
The critic can reduce policy-gradient variance or train an off-policy actor;
only the actor and required preprocessing usually deploy.

### Affordance

An estimate of whether a skill or interaction is feasible in the current
situation. A language instruction may make “pick up sponge” relevant while the
affordance is low because no sponge is reachable.

### Agent

The decision-making system interacting with an environment. Depending on scope,
it may mean one policy network or a larger planner, estimator, memory, skill
registry, and safety-governed runtime. State the boundary.

### Algorithm versus implementation

An algorithm is an abstract update or planning procedure. An implementation
also chooses network, optimizer, normalization, batching, numerical precision,
wrappers, and defaults. Two projects bearing the same algorithm name can behave
differently because those choices are part of the experiment.

### Calibration

Estimating a sensor or actuator mapping against known references. Camera
intrinsics map rays to pixels; camera extrinsics map between frames; encoder
zero calibration maps readings to physical joints. Calibration uncertainty and
revision belong in deployment evidence.

### Confidence interval

An interval produced by a procedure designed to cover an unknown population
quantity at a stated long-run rate under assumptions. It describes uncertainty
of an estimate, not a range containing a fixed percentage of individual robot
outcomes.

### Covariance

A matrix describing the scale and joint variation of uncertain variables.
Diagonal entries are variances; off-diagonal entries record correlation. A pose
estimate needs frame and covariance because errors in position and orientation
are often coupled.

### Cross-entropy method (CEM)

A sampling optimizer that repeatedly evaluates candidates, keeps an elite
fraction, and refits its proposal distribution. Model Predictive Control and
QT-Opt use variants to search continuous action sequences.

### Dataset, demonstration, and transition

A transition is one record such as
$(o_t,a_t,r_t,o_{t+1},d_t)$. A demonstration is a trajectory produced by an
expert or teleoperator. A dataset is a versioned collection plus schema,
provenance, splits, and license—not merely a directory of arrays.

### Dataset Aggregation (DAgger)

An interactive imitation algorithm that rolls out the learner, asks an expert
to label states the learner actually visits, aggregates those labels, and
re-trains. It attacks behavior cloning's state-distribution shift, but requires
safe expert access on learner-induced states.

### Datasets for Deep Data-Driven Reinforcement Learning (D4RL)

An influential collection and protocol for studying fixed-dataset RL. The name
does not imply that every current environment/version or score normalization is
unchanged; pin the benchmark revision.

### Digital twin

A versioned computational model connected to measured properties and validation
data from a physical system. A visually similar robot model with guessed mass,
actuators, contacts, and delay is a simulation asset, not yet a validated
digital twin.

### Discount factor

The number $\gamma\in[0,1]$ multiplying later rewards. With constant timestep
$\Delta t$, changing control rate while keeping $\gamma$ changes the effective
physical-time horizon. A continuous-time interpretation often chooses
$\gamma=\exp(-\Delta t/\tau)$ for time constant $\tau$.

### Dynamics

The rule or probability distribution governing how state changes under action.
Robot dynamics include rigid-body motion, contacts, actuators, delay, and
unmodeled disturbances—not merely geometry.

### Embodiment

The physical form and sensing/action interface of a robot: morphology, joints,
actuators, cameras, controllers, frames, and rates. Cross-embodiment learning
requires explicit adapters because the same task can use incompatible actions.

### Exteroception and proprioception

Exteroception senses the outside world, such as camera, depth, or light
detection and ranging (LiDAR).
Proprioception senses the robot, such as encoders and inertial motion. A policy
cannot avoid a pre-contact obstacle from proprioception alone unless another
module converts exteroception into its commands.

### Conditional variational autoencoder (CVAE)

A latent-variable generative model trained with reconstruction/action loss and
a divergence regularizer toward a simple latent prior. Action Chunking with
Transformers uses a CVAE-style latent to represent variation in demonstrated
action sequences.

### Action Chunking with Transformers (ACT)

An imitation architecture that predicts temporally coordinated action chunks
with a transformer and latent-variable objective. Its acronym names the method;
chunk length, temporal ensembling, action normalization, and latency are still
deployment choices.

### Conservative Q-Learning (CQL)

An offline RL method that penalizes high values on actions not sufficiently
supported by the fixed dataset while fitting observed Bellman targets. The
conservatism strength trades exploitation against pessimism.

### Implicit Q-Learning (IQL)

An offline RL method that learns value through expectile regression on dataset
actions, then extracts a policy by advantage-weighted imitation. It avoids
querying arbitrary new policy actions during its central value target.

### Generalization and out-of-distribution input

Generalization is performance on a specified held-out population. An
out-of-distribution input lies outside or in a low-density region of the
training distribution. Novel texture, novel geometry, novel dynamics, and a
new embodiment are different generalization axes.

### Hierarchical policy and option

A hierarchical policy selects goals or temporally extended skills. An option
has start conditions, an internal policy, and termination. Because it consumes
multiple primitive steps, its next-option value is discounted by the elapsed
duration.

### Inverse kinematics (IK)

Solving for joint configurations that place a link/end effector at a desired
pose. IK handles geometry, not automatically dynamics, torque, collision,
contact stability, or a time-optimal trajectory.

### Latency, jitter, and age of information

Latency is a delay between named events; jitter is variation in that delay.
Age of information is how old the source measurement is when an action uses it.
A fast average inference time can coexist with unsafe queue age and tail jitter.

### Model Predictive Control (MPC)

A controller that repeatedly predicts candidate future trajectories over a
finite horizon, optimizes a cost, executes a prefix—often one action—and
replans. Its behavior depends on model error, horizon, optimizer budget,
constraints, and terminal value/cost.

### Model-free RL

Reinforcement learning that does not explicitly learn/use a transition model
for planning. A critic still predicts expected return, so “model-free” does not
mean “learns no predictive quantity.”

### Occupancy grid

A map dividing space into cells with estimated occupied/free probability or
log-odds. Cell probabilities are not independent guarantees, and the map needs
a frame, timestamp, valid region, and uncertainty behavior.

### Online and offline reinforcement learning

Online RL can collect new experience and correct its policy distribution.
Offline RL must learn only from a fixed dataset, so unsupported actions and
coverage become central. Off-policy online RL with replay is not the same as
offline RL because it still gathers new data.

### Policy gradient

The gradient of expected return with respect to policy parameters. A common
estimator weights $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ by an advantage.
It is an expectation identity; finite batches, critics, clipping, and optimizer
choices determine practical variance and bias.

### Potential-based shaping

A reward of the form
$F(s,s')=\gamma\Phi(s')-\Phi(s)$. Under its theorem's assumptions, it preserves
optimal policies while rewarding progress in a potential $\Phi$. Arbitrary
dense bonuses do not inherit that guarantee.

### Rapid Motor Adaptation (RMA)

A teacher-student locomotion architecture in which a base policy uses a compact
environment latent and a deployment module predicts that latent from recent
proprioceptive history. It adapts only to hidden properties that affect observed
history within the trained distribution.

### Random seed

An integer initializing pseudorandom processes such as network weights, resets,
data order, and exploration. Multiple seeds sample training variability; many
episodes from one seed do not replace independent trainings.

### Return

The discounted sum of future rewards from a time step. Reward is one immediate
signal; return is what the RL objective seeks to increase in expectation.

### Reward mass

The observed weighted contribution a reward term accumulates, rather than its
configuration weight alone. Compare reward mass when copying regularizers
between tasks whose positive terms or episode lengths differ.

### Risk and Conditional Value at Risk (CVaR)

Expected return averages outcomes. Conditional Value at Risk (CVaR) summarizes
a chosen tail, such as the mean of the worst 10% returns. It can optimize tail
performance only with a correctly defined loss orientation and enough tail
samples; it is not a hard safety guarantee.

### Semi-Markov decision process

A decision process whose high-level actions can last different durations.
Option returns accumulate primitive rewards and bootstrap with $\gamma^k$ after
an option lasting $k$ steps.

### Simultaneous localization and mapping (SLAM)

Estimating robot pose while constructing or updating a map. SLAM output remains
an uncertain, time-stamped estimate; it does not by itself choose a collision-
free route or control motors.

### Transformer, attention, and token

A token is one vector/symbol presented to a sequence model. Attention computes
content-dependent weighted combinations of token values. A transformer stacks
attention and feed-forward blocks; it does not automatically understand frames,
physics, causality, or safety.

### Frequency-space Action Sequence Tokenization (FAST)

A method that transforms and compresses continuous action chunks into frequency-
space tokens for autoregressive vision-language-action models. Smooth temporal
structure can use fewer tokens than independent per-timestep bins, subject to
reconstruction and task validation.

### Truncation

Stopping data collection because of a time or external limit while the modeled
task could continue. It differs from true termination for value bootstrapping.
Record both flags rather than collapsing them into one `done` value.

### Uncertainty calibration

Agreement between predicted confidence and empirical frequency on a defined
population. If events assigned confidence 0.8 occur about 80% of the time, that
region is calibrated. Calibration can fail after distribution shift and must
change permitted behavior to matter operationally.

### Update-to-data ratio

The amount of optimizer sample consumption per newly collected transition in
off-policy training. Too little reuse wastes data; too much can overfit replay
and amplify critic error. Report the exact convention because projects count
batches and gradient steps differently.

### Vision-language model and large language model

A vision-language model (VLM) relates images and text. A large language model
(LLM) predicts/represents language at scale. Neither term implies grounded motor
skill. A VLA adds an action-learning path and still needs an embodiment adapter,
local control, timing, and safety.

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

## 20.20 Worked problem 19: camera geometry and perception age

### Background

An ideal depth camera back-projects horizontal pixel coordinate $u$ using
$X=Z(u-c_x)/f_x$. This gives a point in the camera frame; extrinsic calibration
is still required for the robot frame. While the perception system computes,
the robot continues moving, so geometric error and temporal age both consume
clearance.

### Problem

A camera has $f_x=400$ pixels and $c_x=320$ pixels. Pixel $u=360$ has depth
$Z=1.5$ m. Compute camera-frame $X$. If the rover travels at 0.4 m/s and the
measurement is 200 ms old at action time, how much unobserved forward travel
must be added to the reasoning?

## 20.21 Worked problem 20: uncertainty-aware stopping speed

### Background

Use conservative clearance $D_{safe}=\mu_D-k\sigma_D$ and stopping envelope

```math
d_{stop}=v\tau+\frac{v^2}{2a_b}+m.
```

Solving $D_{safe}=d_{stop}$ yields a maximum speed under the model. It is a
governor calculation, not a proof that Gaussian errors or constant braking hold.

### Problem

Mean clearance is 0.70 m, standard deviation 0.05 m, and $k=2$. Reaction age is
0.15 s, verified braking magnitude is 1.0 m/s², and residual margin is 0.10 m.
Compute conservative clearance and maximum speed.

## 20.22 Worked problem 21: option return and elapsed-time discount

### Background

A hierarchical option lasts several primitive steps. Its internal reward is
discounted step by step, and the next high-level value is discounted by the
entire duration. Forgetting duration makes long options appear too attractive.

### Problem

An option lasts three steps with rewards $[2,-1,3]$, discount $\gamma=0.9$,
and next-state best option value 5. Compute its internal return and full
high-level target.

## 20.23 Worked problem 22: potential shaping telescopes

### Background

Potential-based shaping adds
$F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t)$. Discounted across a trajectory,
intermediate potentials cancel. This is why progress pays while holding one
state does not create an unlimited per-step jackpot under the theorem's setup.

### Problem

For two transitions, let $\gamma=0.9$ and potentials be
$\Phi(s_0)=0.2$, $\Phi(s_1)=0.5$, and $\Phi(s_2)=0.9$. Compute each shaping
reward and their discounted sum. Verify it equals
$-\Phi(s_0)+\gamma^2\Phi(s_2)$.

## 20.24 Worked problem 23: offline extrapolation error

### Background

An offline critic sees only a fixed dataset. Function approximation can assign
an unsupported action a high value even though no transition confirms it. A
greedy actor then exploits the critic rather than the environment.

### Problem

At one state, two dataset actions have estimated values 4 and 3. An unseen
action has estimated value 9. What does naïve greedy policy improvement choose?
Explain how Conservative Q-Learning and Implicit Q-Learning respond differently
without claiming either knows the unseen action's true value.

## 20.25 Worked problem 24: action-chunk staleness

### Background

An action chunk reduces model calls but actions later in the chunk are based on
an older observation. Receding-horizon execution limits that open-loop age by
executing only a prefix.

### Problem

Inference finishes 100 ms after image capture. A policy predicts 10 actions for
a 50 Hz controller. If all 10 execute, how old is the source image when the last
action begins? If only the first 5 execute before replanning, what is the age of
the fifth? Ignore other delays and count the first action at inference finish.

## 20.26 Worked problem 25: paired evaluation

### Background

Using the same held-out scenarios for two policies can remove common scenario
difficulty. Analyze per-scenario differences rather than comparing only two
unpaired means.

### Problem

Policy A scores $[10,12,9,13]$ and policy B scores $[8,11,10,9]$ on four
matched scenarios. Compute paired differences and their mean. Why are four
scenarios still insufficient for a broad generalization claim?

## 20.27 Worked problem 26: update-to-data ratio

### Background

For off-policy learning, one convention defines update-to-data reuse as
$U=GB/N_{new}$, where $G$ is gradient batches, $B$ is batch size, and $N_{new}$
is newly collected transitions. Always state the convention because some tools
count optimizer steps rather than sampled transition slots.

### Problem

A collection cycle gathers $4096\times24$ transitions. The learner performs 96
gradient batches of size 1,024. Compute $U$. What would $U$ be for 384 batches,
and what risk grows as reuse increases?

## 20.28 Open exercises

1. Derive the maximum 20-second episode length in policy steps.
2. At iteration 1,000, what is the curriculum environment-step counter?
3. Draw the path from `head_pose` command to a reward and to the ONNX input.
4. Explain why body-pose user interface (UI) controls do not prove the velocity
   checkpoint learned body-pose control.
5. Design a five-case evaluation battery for exact-zero standing.
6. List three real measurements needed before transferring a policy to a new
   actuator.
7. Find one pure helper and its regression test in `tests/`; explain why the
   pure form is easier to test than a full simulator wrapper.


Return to the [book index](README.md) and repeat any lab whose terms or
calculation are still unclear.

## 20.29 Folded solutions

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
<summary>Problem 19: camera geometry and perception age — show solution</summary>

Back-project the horizontal coordinate:

```math
X=Z\frac{u-c_x}{f_x}
=1.5\frac{360-320}{400}
=0.15\ \mathrm{m}.
```

The point is 15 cm along the camera's declared positive horizontal axis. The
robot travels during age:

```math
d_{age}=v\tau=0.4(0.200)=0.080\ \mathrm{m}.
```

Eight centimeters must be included before braking and residual margin. The two
numbers describe different axes in this example; transform the point into the
robot frame before using it as forward clearance.

</details>

<details>
<summary>Problem 20: uncertainty-aware stopping speed — show solution</summary>

Conservative clearance is

```math
D_{safe}=0.70-2(0.05)=0.60\ \mathrm{m}.
```

Set the stopping envelope equal to 0.60 m:

```math
0.60=0.15v+\frac{v^2}{2}+0.10.
```

Equivalently, $v^2+0.30v-1.0=0$. Take the nonnegative root:

```math
v_{max}=\frac{-0.30+\sqrt{0.30^2+4}}{2}
\approx0.861\ \mathrm{m/s}.
```

Substitution gives roughly 0.60 m. This bound is only as defensible as the
measured braking, age tail, calibration, uncertainty coverage, and margin.

</details>

<details>
<summary>Problem 21: option return and elapsed-time discount — show solution</summary>

The internal option return is

```math
R_o=2+0.9(-1)+0.9^2(3)=3.53.
```

The next option is selected after three steps, so its value receives
$\gamma^3=0.729$:

```math
y=3.53+0.729(5)=7.175.
```

Using only one factor of $\gamma$ would overvalue the future after this
three-step option relative to a shorter option.

</details>

<details>
<summary>Problem 22: potential shaping telescopes — show solution</summary>

For the first transition:

```math
F_0=0.9(0.5)-0.2=0.25.
```

For the second:

```math
F_1=0.9(0.9)-0.5=0.31.
```

Discount the second shaping reward because it arrives one step later:

```math
F_0+\gamma F_1=0.25+0.9(0.31)=0.529.
```

The endpoints give the same result:

```math
-\Phi(s_0)+\gamma^2\Phi(s_2)
=-0.2+0.9^2(0.9)=0.529.
```

The intermediate $\Phi(s_1)$ cancels. Terminal handling and discount must match
the theorem; adding an arbitrary “upright every step” bonus does not telescope.

</details>

<details>
<summary>Problem 23: offline extrapolation error — show solution</summary>

Naïve greedy improvement chooses the unseen action with estimated value 9. The
choice is based on critic extrapolation, not supporting evidence.

Conservative Q-Learning adds pressure lowering values of broadly sampled or
policy-proposed actions relative to dataset actions, so the unsupported 9 should
lose its easy advantage when the regularizer works. Implicit Q-Learning fits a
state value from dataset action values using an upper expectile, then extracts
a policy by giving higher imitation weight to above-value dataset actions; its
central update does not maximize over the unseen action. Neither method observes
the unseen action's true result. If dataset coverage omits every good action,
conservatism cannot invent safe evidence.

</details>

<details>
<summary>Problem 24: action-chunk staleness — show solution</summary>

A 50 Hz period is 20 ms. With the first action beginning at 100 ms, the tenth
begins nine periods later:

```math
A_{10}=100+9(20)=280\ \mathrm{ms}.
```

The fifth begins four periods after the first:

```math
A_5=100+4(20)=180\ \mathrm{ms}.
```

Replanning after five actions reduces the oldest executed source age by 100 ms
in this simplified schedule. Real age also includes exposure, queues, transport,
and scheduling jitter; disturbance can make even 180 ms unacceptable for
balance.

</details>

<details>
<summary>Problem 25: paired evaluation — show solution</summary>

Compute A minus B for each matched scenario:

```math
d=[10-8,\ 12-11,\ 9-10,\ 13-9]=[2,1,-1,4].
```

The mean paired improvement is

```math
\bar d=\frac{2+1-1+4}{4}=1.5.
```

One scenario favors B, which a global mean could hide. Four selected scenarios
give little information about training-seed variability, rare failures, or a
larger target population. Preserve the pairs and add preregistered scenarios
and independent training seeds.

</details>

<details>
<summary>Problem 26: update-to-data ratio — show solution</summary>

New transitions are

```math
N_{new}=4096(24)=98{,}304.
```

For 96 batches:

```math
U=\frac{96(1024)}{98{,}304}=1.
```

For 384 batches, $U=4$. Four times as many sampled replay slots are optimized
per new transition. Greater reuse may improve data efficiency, but it increases
compute and can overfit replay, amplify critic/extrapolation error, or let the
actor exploit a lagging critic. It does not mean each transition is seen exactly
four times because replay sampling is random.

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
