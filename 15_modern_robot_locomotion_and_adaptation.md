# 15. Modern Robot Locomotion, Adaptation, and Sim-to-Real Research

Recent locomotion systems combine ideas rather than relying on one magical
algorithm: graphics processing unit (GPU)-parallel simulation, calibrated
actuator models, domain randomization, privileged training information,
histories or latent adaptation, curricula, and strict deployment interfaces.

This chapter is an annotated research pathway. Dates and evaluated scope are
included because “state of the art” is meaningful only relative to a protocol.

## 15.1 The recurring locomotion architecture

Many systems have this shape:

```text
terrain / navigation / operator
             |
      command or short plan
             |
  learned locomotion policy (20–200 Hz)
             |
 joint targets or torques
             |
 embedded motor control and safety (faster)
             |
        physical robot
```

The learned policy is usually a local motor skill. It does not necessarily
choose a destination, recognize an object, build a map, or enforce hard current
limits.

### 15.1.1 Locomotion is a hybrid dynamical system

During one fixed contact mode, rigid-body dynamics can be written

```math
M(q)\ddot q+C(q,\dot q)\dot q+g(q)
=S^T\tau+J_c(q)^T\lambda_c,
```

where $q$ is generalized position, $M$ inertia, $C$ velocity-dependent terms,
$g$ gravity, $\tau$ actuator torque, $S$ selects actuated joints, and contact
Jacobian $J_c$ maps contact force $\lambda_c$ into generalized force.

Touchdown and liftoff change constraints. An impact can reset velocity:

```math
x^+=\Delta(x^-),
```

with pre-impact state $x^-$ and post-impact state $x^+$. Continuous flows plus
discrete contact transitions make locomotion *hybrid*. A neural policy need not
explicitly label every mode, but contacts, history, and physics still determine
which transition is possible.

This explains several practical facts:

- a small action difference near touchdown can cause a large trajectory change;
- average one-step model error can hide wrong contact timing;
- a smooth action penalty cannot define a valid gait by itself; and
- evaluation must stratify contact failures, not only average velocity.

### 15.1.2 Classical models remain useful mental instruments

For slow static balance, the center-of-mass projection should remain inside the
support polygon. Dynamic walking intentionally violates this static condition,
so a stronger simplified model is the linear inverted pendulum. At constant
center-of-mass height $z_0$, horizontal motion relative to support point $p$
obeys

```math
\ddot x=\omega_0^2(x-p),
\qquad \omega_0=\sqrt{g/z_0}.
```

Define the capture point

```math
\xi=x+\frac{\dot x}{\omega_0}.
```

If a foot/support point can be placed at $p=\xi$ in the ideal model, divergent
motion is arrested asymptotically. This is not an exact biped controller for
Microduck—height changes, angular momentum, finite feet, torque limits, and
impact matter—but it gives a physical diagnostic: a policy falling forward
may lack reachable support ahead of its capture point.

Periodic gait stability can be studied with a Poincaré return map. Observe a
state $x_k$ at the same gait event each cycle, such as left touchdown:

```math
x_{k+1}=P(x_k).
```

A periodic gait is fixed point $x^*=P(x^*)$. Locally, perturbations evolve as

```math
\delta x_{k+1}\approx A_P\delta x_k,
\qquad A_P=\left.\frac{\partial P}{\partial x}\right|_{x^*}.
```

If every relevant eigenvalue magnitude of $A_P$ is below one, perturbations
shrink cycle to cycle in the local model. Reinforcement learning (RL) does not make this theory obsolete;
push recovery and touchdown-to-touchdown error are empirical ways to probe the
same stability question.

### 15.1.3 What RL contributes

Classical models expose structure and constraints. RL can optimize a nonlinear,
contact-rich feedback law over a broad state distribution without solving a
trajectory optimization online. In practice the strongest systems combine:

```text
measured mechanics and actuator loops
+ simulator/contact model
+ task/reward and curriculum
+ learned local feedback
+ model-based safety/navigation around it
```

The learning problem is not “replace control theory.” It is “which feedback
mapping is hard to hand-design, and how will physical structure constrain,
diagnose, and protect it?”

## 15.2 Milestone: actuator-aware sim-to-real locomotion

The 2019
[Learning agile and dynamic motor skills for legged robots](https://arxiv.org/abs/1901.08652)
paper trained policies in simulation and transferred them to ANYmal. A key
lesson is that actuator behavior and dynamics identification are first-class
parts of the learning system.

What to study:

- how actuator behavior was represented;
- what observations/actions reached the deployed network;
- which parameters were randomized;
- how real tests were evaluated; and
- which skills used separate policies or objectives.

What not to conclude: every robot can copy ANYmal's parameter values or reward
weights. Morphology, actuator, sensor, rate, and task differ.

Microduck's Better Actuator Models (BAM) XL330 model belongs to the same
engineering tradition: train through a more realistic actuator interface
instead of an unlimited ideal position source.

An ideal position source assumes $q_{t+1}$ can follow a target regardless of
load. A physical actuator is closer to

```math
\tau_t=f(q_t,\dot q_t,q_t^{target},V_t,T_t,\eta),
```

where voltage $V_t$, temperature $T_t$, and parameters $\eta$ affect available
torque, friction, saturation, and delay. The environment transition therefore
depends on actuator internal/operating state. A policy trained on impossible
torque can learn a gait no hardware controller can realize.

Actuator-aware transfer has two complementary parts:

1. identify a nominal $f$ from safe measured excitations and held-out traces;
2. randomize remaining uncertainty and operating variation around it.

Randomizing a bad functional form cannot create missing hysteresis or thermal
dynamics; fitting one trace cannot establish robustness. This nominal-plus-
residual view is a recurring theme from the 2019 work through newer system-
identification and residual-alignment methods.

## 15.3 Milestone: massively parallel simulation

The 2021
[Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
work showed that thousands of GPU-parallel environments can greatly reduce
wall-clock Proximal Policy Optimization (PPO) training time and introduced a
game-inspired terrain curriculum. It accompanied the open
[`legged_gym`](https://github.com/leggedrobotics/legged_gym) ecosystem.

The important distinction is:

```math
\text{environment steps}
\ne
\text{wall-clock seconds}.
```

Parallelism produces more samples per second; it does not make each sample more
informative. It changes optimization dynamics too: batch size, update count,
and policy lag must be understood.

For $N_e$ environments, rollout depth $H$, and policy rate $f_p$, one update
contains

```math
B=N_eH
```

transitions and

```math
t_{aggregate}=\frac{N_eH}{f_p}
```

simulated robot-seconds, but only $H/f_p$ seconds of temporal depth per world.
Increasing $N_e$ cannot reveal a 10-second consequence if every fragment is
0.48 seconds and bootstrapping/history cannot represent it.

At fixed total transition budget, changing $N_e$ can change:

- batch size and stochastic-gradient variance;
- number of policy updates;
- diversity versus temporal correlation inside a batch;
- normalization statistics;
- policy lag for off-policy replay; and
- curriculum progress if keyed to iterations rather than total transitions.

Thus a throughput benchmark and a learning comparison are separate. Plot
transitions per wall second versus environment count to find device saturation;
plot performance versus transitions and wall time to study sample and compute
efficiency.

Microduck's 4,096 environments × 24 steps is this pattern applied through
MuJoCo Warp and the Robotic Systems Lab reinforcement learning (RSL-RL)
library.

## 15.4 Open training stacks

Useful open projects occupy different layers:

| Project | Main role | Study value |
| --- | --- | --- |
| [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) | robot-focused PPO and related training components | readable runner/algorithm/storage separation |
| [`legged_gym`](https://github.com/leggedrobotics/legged_gym) | Isaac Gym locomotion environments | influential vectorized locomotion recipe; pin compatible versions |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | Isaac Sim robot-learning framework | sensors, randomization, reinforcement learning (RL), and imitation workflows |
| [`mjlab`](https://github.com/mujocolab/mjlab) | manager-based MuJoCo Warp robot learning | Microduck's environment layer |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | GPU-accelerated MuJoCo robot-learning suite | open locomotion, manipulation, vision, and sim-to-real examples |
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | general robotics physics/simulation platform | alternative parallel simulator; inspect released features, not roadmap claims |

These are not drop-in equivalents. Compare physics, renderer, task application
programming interface (API), algorithm integration, supported hardware,
license, version compatibility, and reproducibility before migrating.

The 2025
[MuJoCo Playground paper](https://arxiv.org/abs/2502.08844) documents an open
MuJoCo-based stack across locomotion and manipulation. Its breadth makes it a
valuable comparison project for the Microduck/mjlab design.

## 15.5 Domain randomization: train a distribution, not one simulator

Let physical parameters be $\xi$, such as mass, friction, delay, or motor
strength. Instead of optimizing one nominal environment, domain randomization
optimizes

```math
J(\theta)=
\mathbb{E}_{\xi\sim p(\xi),\ \tau\sim\pi_\theta,\xi}
\left[\sum_t\gamma^tr_t\right].
```

Plain language: sample a plausible robot/world, run the policy, and optimize
average performance over that population.

Early primary demonstrations include
[visual domain randomization](https://arxiv.org/abs/1703.06907) and
[dynamics randomization for control](https://arxiv.org/abs/1710.06537).

### What to randomize

Randomize uncertainty that exists and affects behavior:

- link mass, inertia, and center of mass;
- ground friction/restitution/compliance;
- actuator gain, strength, friction, damping, delay, and voltage;
- encoder/inertial measurement unit (IMU) bias, noise, mounting error, and
  dropout;
- command delay and control-period jitter;
- initial state, terrain, disturbances, and payload.

### Three failure modes

1. **Wrong center**: randomization surrounds an inaccurate nominal model.
2. **Too narrow**: the real robot lies outside the training distribution.
3. **Too broad**: policy becomes unnecessarily conservative or cannot find a
   behavior valid across contradictory physics.

Calibrate, quantify residual uncertainty, randomize, then validate held-out
corners. Do not select ranges by folklore.

### Expected robustness is not worst-case robustness

The expectation objective can sacrifice a rare parameter region to improve the
majority. Two alternatives are a lower-tail objective and a minimax objective:

```math
\max_\theta\ \mathrm{CVaR}^{lower}_\alpha(J(\theta;\xi)),
```

```math
\max_\theta\ \min_{\xi\in\Xi}J(\theta;\xi).
```

Conditional value at risk (CVaR) emphasizes the worst-performing fraction.
Minimax emphasizes the worst member of a declared set. Both can become
conservative or chase simulator artifacts, so physical plausibility of tails
and failure classification remain essential.

Parameters are often correlated. Adding a payload changes mass, center of mass,
and inertia together; battery voltage and motor torque limit co-vary; delay and
packet loss may increase under bus load. Independent uniform sampling can
create impossible combinations and miss real correlated ones. Represent known
hardware modes as joint distributions or discrete mixtures:

```math
p(\xi)=\sum_k p(k)\,p(\xi\mid\text{hardware mode }k).
```

Use held-out *combinations*, not only held-out one-dimensional endpoints. A
policy robust to low friction alone and weak motor alone may still fail when
both coincide.

Randomization curricula start near nominal physics, then widen after skill
appears. This can solve exploration, but it risks overfitting to a nominal gait
that cannot adapt. Compare against full-range training and record performance
at every range, not only the final average.

## 15.6 System identification is becoming more systematic

The open
[PACE sim-to-real project](https://github.com/leggedrobotics/pace-sim2real)
uses measured encoder data and optimization to identify actuator/joint dynamics
for legged robots. It is a useful modern study of the workflow

```text
safe excitation -> measured trajectory -> parameter fit
                -> held-out validation -> policy training/transfer
```

The reusable idea is measurement-driven identification. Its exact tooling and
models are not automatically suitable for XL330 servos or a wheeled-leg robot;
the excitation, parameters, and safety protocol must match the hardware.

Given real trajectory $y_{0:T}$ and simulated prediction
$\hat y_{0:T}(\eta;u)$ under input $u$, a basic estimator is

```math
\eta^*=\underset{\eta}{\arg\min}
\sum_t\lVert W(y_t-\hat y_t(\eta;u))\rVert_2^2
+\lambda R(\eta).
```

$W$ balances units/measurement confidence; $R$ encodes plausible parameter
ranges. Optimization does not guarantee identifiability. If changing inertia
and motor gain produces the same slow response, many $\eta$ fit equally well.
The input must be persistently exciting enough—within safety limits—to expose
the dynamics being estimated.

Validate on different amplitudes, frequencies, directions, loads, battery
states, and temperatures. Then inspect residual structure:

- zero-mean random residual suggests uncertainty/noise randomization;
- constant bias suggests calibration/nominal error;
- phase-dependent residual suggests missing delay/dynamics;
- state-dependent residual suggests an omitted nonlinear mode; and
- cross-joint residual suggests coupling not captured by independent models.

Do not squeeze all residuals into wider scalar friction bounds. Model the layer
that causes the pattern.

## 15.7 Privileged learning

Simulation exposes information unavailable on the robot: exact base velocity,
terrain height, contact force, friction, mass, and external disturbance.
**Privileged learning** uses this information during training without requiring
it at deployment.

Three patterns are common:

### Asymmetric critic

```text
actor  <- deployable observations only
critic <- actor observations + privileged state
```

The better-informed critic can estimate value with less ambiguity while the
actor remains deployable.

Let deployed observation be $o_t$ and privileged state be $p_t$. The actor is
$\pi_\theta(a\mid o)$, while critic is $V_\phi(o,p)$. The policy-gradient
estimate uses

```math
\hat g=\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid o_t)
\hat A_t(o_t,p_t).
```

Privilege enters the training baseline/advantage, not the actor input. A more
accurate critic can reduce variance, but a critic that overfits exact simulator
state may generalize poorly across randomization and destabilize advantages.
Compare actor-identical ablations across seeds and inspect value error by
domain, not merely final return.

### Teacher-student distillation

```text
privileged teacher -> target actions or latent representation
deployable student -> imitates using sensor-accessible history/vision
```

A basic student loss is

```math
L_{distill}=\mathbb E
\left[\lVert\pi_S(h_t)-\pi_T(o_t,p_t)\rVert_2^2\right],
```

where $h_t$ is deployable history. Training only on teacher trajectories can
repeat behavior cloning (BC) covariate shift; roll out the student in simulation and label its
visited states with the teacher when safe, or fine-tune with task reward.
Teacher action may also be impossible for the student's delayed/noisy sensors
to infer; no distillation loss can recover information absent from $h_t$.

### Privileged experience for later RL

A privileged policy generates useful trajectories that warm-start a visual or
off-policy learner.

The non-negotiable audit is actor information at deployment. A simulator result
with height-map truth does not prove a depth-camera student works.

Test for leakage by replacing every privileged channel with nonsense during
actor-only export and verifying actor output is unchanged. Trace dataflow, not
only network signatures: a command generator or preprocessing feature derived
from simulator truth can leak privilege outside the nominal actor tensor.

## 15.8 Online adaptation and Rapid Motor Adaptation (RMA)

Fixed domain randomization asks one policy to be robust across all sampled
physics. An adaptive system tries to infer the current environment.

[Rapid Motor Adaptation (RMA)](https://arxiv.org/abs/2107.04034) separates:

- a base policy conditioned on an environment latent; and
- an adaptation module that infers that latent from recent proprioceptive
  history.

Conceptually:

```math
z_t=g_\psi(o_{t-H:t},a_{t-H:t-1}),
\qquad a_t=\pi_\theta(o_t,z_t).
```

$z_t$ is not necessarily a human-readable friction coefficient. It is a
control-useful summary inferred from how the robot has responded.

One two-stage construction is:

1. train a base policy with privileged environment encoder
   $z_t^*=e(p_t)$ across randomized domains;
2. freeze or retain the base and train history encoder $g_\psi$ to predict the
   teacher latent:

```math
L_{adapt}=\mathbb E
\left[\lVert g_\psi(h_t)-z_t^*\rVert_2^2\right].
```

The latent is useful only insofar as the base policy's action depends on it and
history identifies it. Two robots with different friction can produce identical
history while standing still; excitation through motion reveals the difference.
This is **identifiability under the policy's own data distribution**.

History length $H$ spans physical time $H\Delta t$. Too short misses slow motor
or terrain effects; too long increases compute and can average over a recent
change. A recurrent neural network can summarize variable history, a temporal
convolution can expose a fixed receptive field, and an explicit estimator can
produce interpretable parameters. Compare latency, reset state, uncertainty,
and failure recovery—not only nominal return.

Questions for a new robot:

- Which physical changes leave an observable trace in the history?
- How long must the history be relative to the dynamics?
- How quickly does the estimate react versus amplify noise?
- What happens immediately after reset, before enough history exists?
- Was the adaptation distribution broad enough to include hardware?

Also ask what happens when conditions change abruptly. Evaluate friction or
payload switches at controlled times and plot latent/action/recovery transient.
A good final steady state with a dangerous two-second adaptation lag is not a
good locomotion controller.

For Microduck, previous action already provides one step of actuator history.
A fair adaptation experiment would add a versioned history/latent path while
holding command, reward, randomization, network budget, and evaluation constant.
First test whether explicit voltage sensing or improved BAM calibration solves
the same failure more simply.

## 15.9 Robust perceptive locomotion

Proprioception tells what the robot feels now; **exteroception** senses the
external environment before contact, using depth, light detection and ranging
(LiDAR), or vision.

The 2022
[Learning robust perceptive locomotion for quadrupedal robots in the wild](https://arxiv.org/abs/2201.08117)
work integrates exteroceptive and proprioceptive information with a recurrent
encoder designed to cope with unreliable terrain perception.

A perceptive controller is a state-estimation problem as well as a policy. A
recurrent belief can be written

```math
h_t=F_\psi(h_{t-1},o_t^{prop},o_t^{exo},m_t,\Delta t_t),
\qquad a_t=\pi_\theta(o_t^{prop},h_t,c_t),
```

where $m_t$ indicates validity/age. Proprioception can reject an exteroceptive
hallucination after contact, while exteroception anticipates terrain. Training
must include corrupted, missing, delayed, and geometrically miscalibrated input
so the fusion rule learns when not to trust it.

Coordinate transforms are part of perception. A depth point measured in camera
frame $C$ becomes body/world point through calibrated transforms such as

```math
p_B=T_{BC}p_C.
```

An incorrect camera pitch makes a flat floor look like a ramp. Domain
randomization around calibration uncertainty teaches tolerance; systematic
mounting bias still requires calibration.

Representation choices include local height maps, point clouds, range rays,
latent image features, and foothold candidates. Each trades geometry,
occlusion, compute, and deployment reproducibility. A height map convenient in
simulation may require mapping/localization that a small robot cannot run.

This addresses a different task than Microduck velocity tracking. Obstacles or
terrain must appear in the actor input (or in a planner producing safe
commands). Rendering an obstacle without sensing it does not create perceptive
locomotion.

Evaluate perception/control jointly by obstacle class, sensor condition, speed,
and stopping/recovery outcome. Pixel or depth reconstruction error alone does
not state whether the robot selected a feasible foot placement.

## 15.10 From locomotion to parkour

Agile terrain tasks need more than a flat-ground velocity reward:

- terrain/obstacle observations;
- goal/contact representation;
- curricula or privileged teachers;
- collision and constraint definitions;
- recovery from perception error; and
- evaluation by obstacle class and failure mode.

[Extreme Parkour with Legged Robots](https://arxiv.org/abs/2309.14341) studies
low-cost quadruped parkour with a front depth camera. The 2024
[SoloParkour](https://arxiv.org/abs/2409.13678) work studies constrained visual
locomotion, using privileged experience to warm-start depth-based off-policy
learning on Solo-12.

Compare what each deploys—not only the video:

- sensor and frame rate;
- observation history;
- teacher information;
- action/control rate;
- constraint definition;
- obstacle distribution; and
- number and type of real trials.

Parkour illustrates three distinct learning problems:

1. **motion feasibility**: can the body produce a jump/climb under torque and
   contact limits?
2. **perception/estimation**: where is the obstacle and how uncertain/stale is
   that estimate?
3. **selection/planning**: which skill, timing, and foothold make progress?

One end-to-end policy may contain all three, but ablations and interfaces should
still separate them. A privileged teacher can solve feasibility/selection from
perfect geometry; a student then learns deployable perception. If the student
fails, compare teacher-on-noisy-observation, perception labels, and motor
tracking before retuning every reward.

Collision penalties are not hard constraints. When success reward outweighs
them, the policy may accept impacts. Report collision/impact distributions and
use runtime feasibility/stop guards where violation cannot be tolerated.

## 15.11 Hierarchical wheeled-leg locomotion

The 2024
[Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots](https://arxiv.org/abs/2405.01792)
integrates adaptive locomotion, a learned navigation controller, and large-scale
path planning. The paper is especially relevant to Jump Rover because it
demonstrates the architectural distinction between local wheel-leg control and
navigation.

It does not imply that a new wheeled-leg platform can train before its
mechanical, actuator, sensing, and realtime interfaces are characterized.
Architecture transfers more readily than weights.

Hierarchy separates horizons and authority:

```text
global planner (seconds/metres): route through mapped world
local navigation (hundreds of ms): collision-aware short trajectory/twist
locomotion (5–20 ms): contact/actuator feedback for requested motion
drive firmware (sub-ms to few ms): current/velocity/position and protection
```

The local command must lie inside the locomotion policy's trained capability
set $\mathcal C(o)$. A command governor can project a planner request onto an
estimated safe set:

```math
c^{applied}=\underset{c\in\mathcal C(o)}{\arg\min}
\lVert c-c^{planner}\rVert_W^2.
```

Estimating $\mathcal C$ is hard; start with conservative measured bounds on
speed, turn rate, acceleration, terrain, tilt, sensor age, and stopping
distance. Log every projection so the planner is not falsely credited for
commands the local layer rejected.

For JumpRover, a cloud agent can interpret goals, retrieve maps, or propose a
task plan. The brain system-on-chip should perform local perception/planning,
and the future real-time board must retain watchdog, bus timing, actuator
limits, and safe-stop authority. Training can begin meaningfully only after the
mechanical model and board/runtime contracts provide credible actions,
observations, delays, and safety behavior.

## 15.12 From motion tracking to reusable whole-body skills

Commanded velocity rewards discover a useful gait but do not specify style or
rich whole-body coordination. Motion-reference methods import kinematic
examples and use physics/RL to make them robust and dynamically feasible.

[DeepMimic](https://arxiv.org/abs/1804.02717) established an influential
example-guided recipe. A tracking reward often combines exponential errors:

```math
r_t=w_q e^{-k_q\lVert q_t-q_t^{ref}\rVert^2}
+w_v e^{-k_v\lVert\dot q_t-\dot q_t^{ref}\rVert^2}
+w_e e^{-k_e\lVert p_t-p_t^{ref}\rVert^2}
+w_c r_{task}.
```

Reference phase/index must be observed or inferred. Retargeting human motion to
a robot is itself an optimization over different limb lengths, joints, limits,
contacts, and balance; copying angles is rarely valid.

[Adversarial Motion Priors (AMP)](https://arxiv.org/abs/2104.02180) learn a
style reward from unstructured motion clips, reducing handcrafted reference
tracking/clip selection. [Adversarial Skill Embeddings
(ASE)](https://arxiv.org/abs/2205.01906) learn a reusable latent repertoire for
downstream tasks. [MaskedMimic](https://arxiv.org/abs/2409.14393) frames diverse
partial control conditions as masked motion inpainting for physics-based
characters. These works expand control expressivity, but many results are in
simulated characters; physical robot transfer adds actuator, impact, sensing,
and safety evidence.

The 2025 preprint
[ASAP](https://arxiv.org/abs/2502.01143) studies two-stage whole-body humanoid
skill transfer with a learned residual/delta action model from real data. The
model is integrated into simulation for policy fine-tuning. This is a modern
instance of structured residual alignment: learn what the nominal simulator
misses rather than only widening every randomization.

The 2025 [BeyondMimic](https://arxiv.org/abs/2508.08241) preprint reports a
motion-tracking foundation plus guided diffusion for test-time whole-body task
control and real humanoid deployment. A 2026 preprint on
[whole-body humanoid locomotion via motion generation and tracking](https://arxiv.org/abs/2604.17335)
combines terrain-aware diffusion reference generation with an RL tracker and
closed-loop fine-tuning for onboard perceptive locomotion. These are frontier
systems reviewed in September 2026, not settled universal recipes; inspect
revisions, released code/checkpoints, real-trial counts, and failure protocols.

The conceptual progression is:

```text
single reference tracking
-> unstructured motion prior
-> reusable skill latent / partial-condition controller
-> task-guided motion generation
-> physical tracker with explicit sim-to-real alignment
```

At every stage, a generated kinematic motion is only a reference. The low-level
physics policy, actuator limits, contacts, and safety system decide whether the
robot can execute it.

## 15.13 Recent off-policy scaling is a research direction

The 2025 preprint
[Learning Sim-to-Real Humanoid Locomotion in 15 Minutes](https://arxiv.org/abs/2512.01996)
reports massively parallel FastSAC/FastTD3 recipes on humanoids. It is valuable
evidence that off-policy continuous-control methods can be engineered at GPU
parallel scale, challenging a simplistic “PPO is always the locomotion choice”
belief.

Treat it as a recent, reproducible direction with stated hardware and compute,
not a universal 15-minute promise. Reproduction must match environment count,
GPU, simulator, update-to-data ratio, task, randomization, and success criteria.

On-policy PPO discards/restricts reuse after several epochs so updates remain
close to the collection policy. Soft Actor-Critic (SAC) and Twin Delayed Deep
Deterministic Policy Gradient (TD3) learn from replay and can reuse each
transition, improving sample efficiency in some settings. Massive parallelism
creates a very fast incoming data stream, so off-policy training must balance
replay freshness, critic updates, target networks, value overestimation, and
device utilization.

Define an update-to-data ratio clearly. One useful convention is critic
minibatch samples consumed per new environment transition:

```math
\mathrm{UTD}=\frac{N_{gradient}\,B_{minibatch}}
{N_{new\ transitions}}.
```

Some papers/code use “gradient steps per collection step” instead, so the same
number can mean something different. State numerator and denominator. High UTD
extracts more optimization from data but can overfit critic errors; low UTD may
leave the learner unable to keep up with collection.

A fair PPO/off-policy comparison matches more than wall time:

| Axis | Why it matters |
| --- | --- |
| transitions | sample efficiency |
| wall time/device/power | compute efficiency |
| actor/critic capacity | representation budget |
| simulator worlds/rate | data throughput/correlation |
| reward/randomization/curriculum | task difficulty |
| evaluation seeds and physical trials | outcome uncertainty |

Algorithm family should be an experimental variable after the environment and
transfer stack are credible, not the first explanation for a broken task.

## 15.14 A research-to-project comparison card

For any locomotion paper, fill this before copying code:

```text
Paper/project and version:
Robot morphology and mass:
Actuator and low-level controller:
Policy rate / physics rate:
Observation (train actor / deploy actor / critic):
Action representation:
Command/goal representation:
Algorithm and network memory:
Simulator and environment count:
System identification:
Domain randomization ranges:
Curriculum/reset distribution:
Baselines:
Number of training seeds:
Sim evaluator:
Real trials and failure categories:
Public code/checkpoint/data:
What is directly transferable:
What must be re-measured:
```

## 15.15 Microduck research extensions

High-value controlled studies include:

1. **Robustness versus adaptation**: compare fixed history-free PPO with an
   adaptation latent under held-out actuator/friction changes.
2. **Calibration versus randomization width**: measure actuators, fit BAM, then
   ablate narrow/medium/wide residual randomization.
3. **Privileged critic ablation**: keep actor identical and compare critic
   information across seeds.
4. **PPO versus off-policy at matched cost**: match environment steps, GPU
   time, evaluator, and model capacity.
5. **Perceptive hierarchy**: compare a local obstacle planner commanding the
   unchanged walk policy against a new perceptive locomotion actor.

Each is a paper-shaped question because it changes one conceptual factor and
defines evidence before training.

Turn them into controlled designs:

| Study | Treatment | Fixed control | Primary evidence | Falsifying result |
| --- | --- | --- | --- | --- |
| robustness vs adaptation | history/latent encoder | same physics range, actor budget, seeds | held-out parameter switches and recovery time | no paired tail/recovery gain |
| calibration vs domain randomization (DR) width | measured nominal + three residual widths | same reward, algorithm, and budget | real/held-out sim tracking and worst slice | width not predictive of transfer |
| privileged critic | add exact velocity/terrain only to critic | identical actor/input/evaluator | seed-level sample efficiency and final score | advantage noise/value error not improved |
| PPO vs off-policy | algorithm/replay | matched transitions, compute record, networks/task | performance vs transitions and wall time | claimed benefit disappears when matched |
| perceptive hierarchy | planner+61D vs versioned perceptive actor | same sensor/scenes/safety | held-out obstacles, latency, collision/progress | complexity adds no robust benefit |

For adaptation, include abrupt tests: nominal → low voltage at $t=5$ s,
high → low friction on a marked patch, or payload attachment between episodes.
Plot latent, action saturation, tracking, tilt, and recovery; final averages
alone hide dangerous transients.

For JumpRover, the first research artifact should be a *measurement and
interface dataset*, not a long RL run: motor/drive response, encoder/inertial
timing, bus age, battery/load behavior, geometry/inertia, contact/wheel friction,
and safe-state transitions. Once the mechanics and real-time board exist, fit a
nominal model, reproduce the control path in simulation, and only then select
robustness/adaptation studies.

## 15.16 Exercises

1. What makes locomotion a hybrid dynamical system? Give one continuous mode
   and one discrete event for Microduck.
2. For center-of-mass height $z_0=0.10$ m, position $x=0.01$ m, velocity
   $\dot x=0.20$ m/s, and $g=9.81$ m/s², calculate $\omega_0$ and capture point
   $\xi$. State why it is only a diagnostic.
3. A Poincaré linearization has eigenvalues `0.6`, `-0.8`, and `1.05`. Is the
   fixed gait locally stable in that model? Interpret the negative eigenvalue.
4. Explain why parallel simulation improves wall-clock throughput but not
   necessarily sample efficiency. Calculate batch transitions and per-world
   temporal depth for 4,096 worlds × 24 steps at 50 hertz.
5. Explain why an ideal position actuator can produce an untransferable policy.
   Name four arguments of a more realistic torque model.
6. Compare expected domain-randomized, lower-tail CVaR, and minimax objectives.
   Which failure does each emphasize?
7. Give two physically correlated randomization variables. Why can independent
   sampling create impossible robots?
8. A parameter fit matches training steps but has phase lag on held-out sine
   sweeps. What does this residual suggest, and why is wider mass randomization
   not the first fix?
9. Give one train-only privileged variable and a way a deployment student could
   infer its effect. How would you test for leakage?
10. Contrast robustness through domain randomization with online adaptation.
    What observation-identifiability condition must adaptation satisfy?
11. A history encoder uses 50 steps at 50 hertz. What physical duration does it
    cover? Give one benefit and one risk of doubling it.
12. Design a failure taxonomy for perceptive stair climbing that separates
    perception, feasibility, control, timing, and safety.
13. Explain why a generated human-motion reference is not directly a safe
    humanoid action. Name the tracker/retargeting responsibilities.
14. Contrast reference tracking, adversarial motion priors, reusable skill
    latents, and masked/conditional motion controllers.
15. Define update-to-data ratio in your own numerator/denominator. Why can
    “UTD=4” be ambiguous across codebases?
16. Design the command-governor interface between a cloud/local planner and
    JumpRover locomotion. Which authority remains on the real-time board?
17. From one paper in this chapter, fill the research-to-project comparison
    card using only the paper and official repository. Mark omissions rather
    than guessing.
18. Propose one Microduck experiment whose negative result would still teach
    something specific; include treatment, controls, primary metric, tail case,
    and falsifier.

Continue with [vision, foundation policies, and hierarchical autonomy](16_vision_foundation_policies_and_hierarchy.md).

## 15.17 Folded solutions

<details>
<summary>Show solutions to Exercises 1–9</summary>

1. Dynamics flow continuously while a contact set is fixed, then touchdown,
   liftoff, impact, or mode switching changes constraints/reset state. For
   Microduck, single-foot support is a continuous mode and the other foot's
   touchdown is a discrete event.
2. $\omega_0=\sqrt{9.81/0.10}\approx9.90$ s⁻¹. Thus
   $\xi=0.01+0.20/9.90\approx0.0302$ m, about 3.0 cm from the reference origin.
   The model assumes constant height, point-like dynamics, simplified support,
   and ignores angular momentum/actuator/contact limits, so it diagnoses rather
   than exactly commands Microduck.
3. No. Local asymptotic stability requires all relevant magnitudes below one;
   `1.05` grows about 5% each cycle. `-0.8` shrinks in magnitude while flipping
   sign each cycle, an alternating decaying perturbation.
4. More worlds increase transitions per wall second, not information gained per
   transition. Batch size is $4096(24)=98{,}304$ transitions. Each world spans
   $24/50=0.48$ s; aggregate simulated time is 1,966.08 s.
5. It allows instantaneous/unlimited target tracking and hides voltage, load,
   friction, saturation, and delay, so the learned gait may demand impossible
   torque. A realistic map can depend on position, velocity, target/error,
   voltage, temperature, load, delay/history, and identified motor/friction
   parameters; any four earn the point.
6. Expectation optimizes average performance under $p(\xi)$ and can sacrifice
   rare cases. Lower-tail CVaR focuses on the worst chosen fraction under that
   distribution. Minimax focuses on the single worst member of a declared set,
   often most conservative and most sensitive to simulator artifacts.
7. Payload mass–center-of-mass–inertia, battery voltage–torque limit, or bus
   load–delay–loss are correlated. Independent extremes can describe a payload
   whose inertia/center is physically inconsistent or a high-voltage motor with
   an unrelated weak torque limit, wasting capacity on impossible combinations.
8. Frequency-dependent phase residual suggests missing delay, filtering, or a
   dynamic mode. Mass width changes inertial scale but does not reproduce a
   consistent phase lag; measure timing/model the causal layer first.
9. True friction can be privileged. A student may infer its effect from recent
   commands, foot motion, body response, and slip. Export the actor, replace
   privileged channels with arbitrary values, and verify outputs are unchanged;
   also trace command/preprocessing dataflow for indirect leakage.

</details>

<details>
<summary>Show solutions to Exercises 10–18</summary>

10. Domain randomization trains one robust mapping across a distribution without
    explicitly identifying its member. Adaptation infers a latent/current
    context and changes actions. Different domains must cause distinguishable
    deployable histories under the policy; standing still may not identify
    friction.
11. Duration is $50/50=1$ s. Doubling may expose slower voltage/payload effects
    and reduce ambiguity, but increases memory/compute, startup state, and risk
    of averaging over an abrupt recent change.
12. Example classes: missing/wrong/stale stair geometry (perception/timing),
    correct geometry but unreachable foothold (feasibility), feasible target
    but wrong action/tracking (control), toe/body collision or slip/saturation
    (physical execution), recovery versus fall, and local safety intervention.
    Report counts per stair/sensor/speed slice and synchronized traces.
13. Human joints/morphology do not match robot links/limits/contacts. Retargeting
    maps bodies while enforcing reachable poses and contacts; the RL tracker
    supplies feedback and dynamic feasibility under torque/impact/randomization;
    runtime safety limits authority. A kinematic reference has no guarantee of
    balance or safe torque.
14. Tracking follows an indexed clip with explicit pose/velocity rewards. A
    motion prior learns style/distribution reward from unstructured clips. A
    skill latent exposes reusable behavioral modes for downstream tasks.
    Masked/conditional controllers fill missing motion from partial goals or
    modalities. Expressivity grows, along with data, inference, and validation
    burden.
15. One definition is minibatch samples consumed by critic gradients divided by
    newly collected transitions. Another codebase may count gradient calls per
    vectorized collection call and omit minibatch size. Therefore “4” is
    ambiguous unless numerator, denominator, batching, actor/critic updates,
    and time interval are stated.
16. The planner sends timestamped, expiring goal/local twist with source and
    sequence. A local command governor clamps/projects it into measured speed,
    acceleration, terrain, age, and stopping bounds; locomotion tracks it. The
    real-time board retains fresh-I/O validation, watchdog, current/position/
    temperature limits, deadline handling, emergency stop, and safe state
    without cloud or brain cooperation.
17. A passing card quotes morphology/mass, actuator/controller, rates,
    train/deploy/critic observations, action, simulator/worlds, algorithm,
    identification/randomization, curricula, seeds/evaluation, real trials,
    failures, and released artifacts. “Not reported” is the correct answer when
    primary sources omit an item.
18. Example: test history-conditioned adaptation versus an actor of matched
    parameter count under held-out voltage drops. Hold task, reward, randomization,
    seeds, transitions, and deployment rate fixed. Primary metric is recovery
    time/tracking after the drop; tail is falls/saturation at lowest safe
    voltage. Falsifier: paired tail/recovery does not improve while nominal
    quality/latency is non-inferior. A negative result bounds the value of
    history and motivates explicit voltage sensing or better actuator fit.

</details>
