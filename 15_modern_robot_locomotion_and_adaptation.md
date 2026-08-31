# 15. Modern Robot Locomotion, Adaptation, and Sim-to-Real Research

Recent locomotion systems combine ideas rather than relying on one magical
algorithm: GPU-parallel simulation, calibrated actuator models, domain
randomization, privileged training information, histories or latent adaptation,
curricula, and strict deployment interfaces.

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

Microduck's BAM XL330 model belongs to the same engineering tradition: train
through a more realistic actuator interface instead of an unlimited ideal
position source.

## 15.3 Milestone: massively parallel simulation

The 2021
[Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
work showed that thousands of GPU-parallel environments can greatly reduce
wall-clock PPO training time and introduced a game-inspired terrain curriculum.
It accompanied the open
[`legged_gym`](https://github.com/leggedrobotics/legged_gym) ecosystem.

The important distinction is:

$$
\text{environment steps}
\ne
\text{wall-clock seconds}.
$$

Parallelism produces more samples per second; it does not make each sample more
informative. It changes optimization dynamics too: batch size, update count,
and policy lag must be understood.

Microduck's 4,096 environments × 24 steps is this pattern applied through
MuJoCo Warp and RSL-RL.

## 15.4 Open training stacks

Useful open projects occupy different layers:

| Project | Main role | Study value |
| --- | --- | --- |
| [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) | robot-focused PPO and related training components | readable runner/algorithm/storage separation |
| [`legged_gym`](https://github.com/leggedrobotics/legged_gym) | Isaac Gym locomotion environments | influential vectorized locomotion recipe; pin compatible versions |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | Isaac Sim robot-learning framework | sensors, randomization, RL/imitation workflows |
| [`mjlab`](https://github.com/mujocolab/mjlab) | manager-based MuJoCo Warp robot learning | Microduck's environment layer |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | GPU-accelerated MuJoCo robot-learning suite | open locomotion, manipulation, vision, and sim-to-real examples |
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | general robotics physics/simulation platform | alternative parallel simulator; inspect released features, not roadmap claims |

These are not drop-in equivalents. Compare physics, renderer, task API,
algorithm integration, supported hardware, license, version compatibility, and
reproducibility before migrating.

The 2025
[MuJoCo Playground paper](https://arxiv.org/abs/2502.08844) documents an open
MuJoCo-based stack across locomotion and manipulation. Its breadth makes it a
valuable comparison project for the Microduck/mjlab design.

## 15.5 Domain randomization: train a distribution, not one simulator

Let physical parameters be $\xi$, such as mass, friction, delay, or motor
strength. Instead of optimizing one nominal environment, domain randomization
optimizes

$$
J(\theta)=
\mathbb{E}_{\xi\sim p(\xi),\ \tau\sim\pi_\theta,\xi}
\left[\sum_t\gamma^tr_t\right].
$$

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
- encoder/IMU bias, noise, mounting error, and dropout;
- command delay and control-period jitter;
- initial state, terrain, disturbances, and payload.

### Three failure modes

1. **Wrong center**: randomization surrounds an inaccurate nominal model.
2. **Too narrow**: the real robot lies outside the training distribution.
3. **Too broad**: policy becomes unnecessarily conservative or cannot find a
   behavior valid across contradictory physics.

Calibrate, quantify residual uncertainty, randomize, then validate held-out
corners. Do not select ranges by folklore.

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

### Teacher-student distillation

```text
privileged teacher -> target actions or latent representation
deployable student -> imitates using sensor-accessible history/vision
```

### Privileged experience for later RL

A privileged policy generates useful trajectories that warm-start a visual or
off-policy learner.

The non-negotiable audit is actor information at deployment. A simulator result
with height-map truth does not prove a depth-camera student works.

## 15.8 Online adaptation and RMA

Fixed domain randomization asks one policy to be robust across all sampled
physics. An adaptive system tries to infer the current environment.

[Rapid Motor Adaptation (RMA)](https://arxiv.org/abs/2107.04034) separates:

- a base policy conditioned on an environment latent; and
- an adaptation module that infers that latent from recent proprioceptive
  history.

Conceptually:

$$
z_t=g_\psi(o_{t-H:t},a_{t-H:t-1}),
\qquad a_t=\pi_\theta(o_t,z_t).
$$

$z_t$ is not necessarily a human-readable friction coefficient. It is a
control-useful summary inferred from how the robot has responded.

Questions for a new robot:

- Which physical changes leave an observable trace in the history?
- How long must the history be relative to the dynamics?
- How quickly does the estimate react versus amplify noise?
- What happens immediately after reset, before enough history exists?
- Was the adaptation distribution broad enough to include hardware?

## 15.9 Robust perceptive locomotion

Proprioception tells what the robot feels now; **exteroception** senses the
external environment before contact, using depth, LiDAR, or vision.

The 2022
[Learning robust perceptive locomotion for quadrupedal robots in the wild](https://arxiv.org/abs/2201.08117)
work integrates exteroceptive and proprioceptive information with a recurrent
encoder designed to cope with unreliable terrain perception.

This addresses a different task than Microduck velocity tracking. Obstacles or
terrain must appear in the actor input (or in a planner producing safe
commands). Rendering an obstacle without sensing it does not create perceptive
locomotion.

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

## 15.12 Recent off-policy scaling is a research direction

The 2025 preprint
[Learning Sim-to-Real Humanoid Locomotion in 15 Minutes](https://arxiv.org/abs/2512.01996)
reports massively parallel FastSAC/FastTD3 recipes on humanoids. It is valuable
evidence that off-policy continuous-control methods can be engineered at GPU
parallel scale, challenging a simplistic “PPO is always the locomotion choice”
belief.

Treat it as a recent, reproducible direction with stated hardware and compute,
not a universal 15-minute promise. Reproduction must match environment count,
GPU, simulator, update-to-data ratio, task, randomization, and success criteria.

## 15.13 A research-to-project comparison card

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

## 15.14 Microduck research extensions

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

## 15.15 Exercises

1. Explain why parallel simulation improves wall-clock throughput but not
   necessarily sample efficiency.
2. Give one train-only privileged variable and a way a deployment student could
   infer its effect.
3. Contrast robustness through domain randomization with online adaptation.
4. Why must randomization ranges be based on calibration and held-out tests?
5. Design a failure taxonomy for perceptive stair climbing.
6. From one paper in this chapter, fill the research-to-project comparison card
   using only the paper and official repository.
7. Propose one Microduck experiment whose negative result would still teach
   something specific.

Continue with [vision, foundation policies, and hierarchical autonomy](16_vision_foundation_policies_and_hierarchy.md).
