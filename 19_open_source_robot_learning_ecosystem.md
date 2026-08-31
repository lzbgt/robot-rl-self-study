# 19. Open-Source Robot-Learning Ecosystem and Reproduction Labs

Open source makes a research claim inspectable, but the repository name is not
the experiment. This chapter maps major projects to their layer, shows what to
read, and gives a low-cost sequence of reproductions.

Project capabilities and versions change. Links were checked on 2026-08-31;
pin a release or commit before reproducing anything.

## 19.1 Separate the layers before choosing software

```text
dataset / demonstrations
          |
environment + simulator + robot/controller model
          |
algorithm + networks + replay/rollout storage
          |
experiment tracking + evaluator
          |
exported model + inference runtime
          |
hardware adapter + realtime safety
```

One repository may cover several layers but rarely all of them. Installing a
reinforcement learning (RL) library does not provide a correct robot model;
downloading a vision-language-action (VLA) checkpoint does not provide your
camera calibration or action adapter.

## 19.2 Small general-RL learning tools

### Gymnasium

- [Official repository](https://github.com/Farama-Foundation/Gymnasium)
- Role: standardized environment application programming interface (API) and
  classic/control benchmarks.
- Read: reset/step return values, termination versus truncation, spaces, and
  seeding.
- Use: validate an algorithm idea cheaply before robot simulation.

### CleanRL

- [Official repository](https://github.com/vwxyzjn/cleanrl)
- Role: single-file deep-RL implementations designed for readability and
  reproducibility.
- Read: one Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC) file
  end to end; map each equation to a code block.
- Caution: a concise benchmark implementation is not a robot deployment stack.

### Stable-Baselines3

- [Official repository](https://github.com/DLR-RM/stable-baselines3)
- Role: maintained PyTorch implementations and a convenient baseline API.
- Use: establish a competent algorithm baseline on Gymnasium-compatible tasks.
- Caution: defaults and wrappers are part of the experiment; inspect them.

## 19.3 Locomotion and graphics processing unit (GPU) simulation stacks

### Robotic Systems Lab reinforcement learning (RSL-RL)

- [Official repository](https://github.com/leggedrobotics/rsl_rl)
- Role: robot-focused runners, PPO and related algorithms, models, storage,
  export utilities, and teacher-student components.
- Read in order: runner → storage → PPO → actor/critic model.
- Microduck use: this is the optimizer/training loop beneath `mjlab` tasks.

### `mjlab`

- [Official repository](https://github.com/mujocolab/mjlab)
- Role: manager-based robot-learning environments on MuJoCo Warp.
- Read: environment configuration, observation/reward/event managers, and task
  registration.
- Microduck use: direct environment framework.

### MuJoCo Playground

- [Official repository](https://github.com/google-deepmind/mujoco_playground)
- [Paper](https://arxiv.org/abs/2502.08844)
- Role: GPU MuJoCo environments across locomotion, manipulation, and vision;
  multiple learning backends.
- Study: compare one task's observation/action/randomization/export path with
  Microduck's.

### Isaac Lab

- [Official repository](https://github.com/isaac-sim/IsaacLab)
- Role: robot-learning framework on Isaac Sim with sensors, actuator models,
  environment design, randomization, RL, and imitation workflows.
- Study: manager-based versus direct environments, actuator configuration, and
  the current supported version matrix.
- Caution: Isaac Sim, driver, NVIDIA Compute Unified Device Architecture
  (CUDA), and extension versions form a large compatibility surface.

### `legged_gym`

- [Official repository](https://github.com/leggedrobotics/legged_gym)
- Role: influential Isaac Gym vectorized locomotion tasks associated with
  Learning to Walk in Minutes.
- Study: reward preparation, terrain curriculum, command sampling, and task
  registry.
- Caution: it targets legacy Isaac Gym and a specific RSL-RL version. Follow
  its pinning instructions; do not combine newest packages by guesswork.

### Genesis

- [Official repository](https://github.com/Genesis-Embodied-AI/Genesis)
- Role: Pythonic robotics physics platform with parallel simulation.
- Study: supported solvers, robot import, vectorized control API, benchmark
  scripts, and released—not announced—features.
- Jump Rover use: one candidate for the future isolated four-action sandbox,
  after measured hardware/digital-twin gates pass.

## 19.4 Manipulation environments and benchmarks

### robosuite

- [Official repository](https://github.com/ARISE-Initiative/robosuite)
- [Paper](https://arxiv.org/abs/2009.12293)
- Role: modular MuJoCo robot/controller/task framework with demonstrations and
  multi-modal sensors.
- Study: how action meaning changes among joint velocity, inverse kinematics,
  and operational-space control.

### ManiSkill

- [Official repository](https://github.com/mani-skill/ManiSkill)
- Role: GPU-parallel SAPIEN manipulation environments and baselines spanning
  RL, imitation, model-based learning, and VLA evaluation.
- Study: heterogeneous parallel scenes, state versus visual observations, and
  real-to-sim/sim-to-real protocols.
- Caution: asset licenses can differ from code license.

### RLBench

- [Official repository](https://github.com/stepjam/RLBench)
- [Paper](https://arxiv.org/abs/1909.12271)
- Role: many language-described manipulation tasks in CoppeliaSim/PyRep.
- Study: task demonstrations, variation definitions, observation modes, and
  success conditions.

### LIBERO

- [Official repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [Paper](https://arxiv.org/abs/2306.03310)
- Role: lifelong/multitask manipulation benchmark with controlled spatial,
  object, goal, and task distribution shifts.
- Study: what a split is intended to test and whether a policy learns transfer
  or memorizes benchmark regularities.
- Caution: the official repository notes its main focus is imitation rather
  than sparse-reward RL.

## 19.5 Robot data and foundation-policy code

### LeRobot

- [Official repository](https://github.com/huggingface/lerobot)
- Role: dataset format, hardware integrations, policy training/evaluation, and
  pretrained-policy ecosystem.
- Study: dataset features/video timing, normalization stats, policy config,
  robot adapter, and evaluation loop.

### Open X-Embodiment

- [Official repository](https://github.com/google-deepmind/open_x_embodiment)
- [Paper](https://arxiv.org/abs/2310.08864)
- Role: standardized multi-institution robot datasets and Robotics Transformer
  X (RT-X) study.
- Study: embodiment/action transforms and dataset sampling weights.

### Octo

- [Official repository](https://github.com/octo-models/octo)
- Role: open generalist policy/checkpoints intended for downstream adaptation.
- Study: token groups, task definitions, action heads, and fine-tuning config.

### OpenVLA

- [Official repository](https://github.com/openvla/openvla)
- Role: open VLA code/checkpoints and fine-tuning path.
- Study: visual encoder fusion, action tokenization, normalization, Low-Rank
  Adaptation (LoRA), and inference serving.

### `openpi`

- [Official repository](https://github.com/Physical-Intelligence/openpi)
- Role: official open implementations/checkpoints for $\pi_0$,
  Frequency-space Action Sequence Tokenization (FAST) variants, and
  $\pi_{0.5}$ support.
- Study: `Observation`/action tensor specs, image masks, data transforms,
  normalization stats, action horizon, config, and serving.
- Caution: the maintainers explicitly frame adaptation to new robots as an
  experiment, not a guaranteed drop-in result.

### Isaac Generalist Robot 00 Technology (GR00T)

- [Official repository](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1 paper](https://arxiv.org/abs/2503.14734)
- Role: open foundation-model artifacts/workflows oriented toward humanoid and
  generalist manipulation.
- Study: data format, embodiment config, dual-system inference, and simulation
  benchmark protocol.

### SmolVLA

- [Paper](https://arxiv.org/abs/2506.01844)
- [Implementation in LeRobot](https://github.com/huggingface/lerobot)
- Role: smaller VLA and asynchronous inference designed for more accessible
  training/deployment.
- Study: observation age, action-chunk queueing, and actual central processing
  unit (CPU)/GPU latency.

## 19.6 How to inspect an unfamiliar repository

Do not start by running its most expensive command. Answer these questions:

1. What license applies to code, weights, data, and assets separately?
2. Which commit/release matches the paper?
3. What Python, framework, CUDA/driver, and simulator versions are required?
4. Is the environment API Gymnasium-old, Gymnasium-new, or custom?
5. What are observation/action shapes, units, frames, rates, and normalization?
6. Where are reward, termination, reset, and randomization defined?
7. What command recreates a released evaluation without training?
8. Are pretrained artifacts content-addressed or mutable links?
9. What external downloads are executed?
10. Does the code send telemetry or require an account?

Then create an isolated environment. Do not install a second simulator stack
into a working Microduck lock environment.

## 19.7 Reproduction ladder

Use the cheapest rung that can falsify the current assumption:

```text
read schema/config
      |
import/runtime smoke
      |
one environment + zero/random action
      |
released checkpoint evaluation
      |
short training smoke
      |
one benchmark reproduction across seeds
      |
one ablation
      |
new robot simulation
      |
processor-in-loop / hardware gates
```

Skipping from repository clone to hardware merges dependency, model, policy,
timing, and safety failures into one mystery.

## 19.8 Lab sequence A — algorithm truth

1. Run this book's bandit, value iteration, Q-learning, and PPO clipping
   programs.
2. Read one matching CleanRL implementation.
3. Use a Gymnasium classic-control task and three seeds.
4. Reproduce a documented baseline before changing it.
5. Implement one controlled ablation.

Outcome: understand algorithm mechanics independent of robot complexity.

## 19.9 Lab sequence B — vectorized robot learning

1. Run Microduck CPU tests and zero-agent playback.
2. Run a 64-environment/five-iteration PPO smoke.
3. Trace one reward and observation through `mjlab` managers.
4. Inspect RSL-RL rollout batch shape and PPO update counts.
5. Export and run CPU Open Neural Network Exchange (ONNX) rehearsal.
6. Repeat one supported MuJoCo Playground task in a separate environment.
7. Compare environment and deployment contracts, not reward numbers.

Outcome: understand a complete simulator-to-runtime motor-policy pipeline.

## 19.10 Lab sequence C — robot imitation and data

1. Inspect one LeRobot dataset in a browser/tool without training.
2. Write its schema/dataset card.
3. Train a simple behavior cloning (BC) baseline on a small split.
4. Evaluate on fixed held-out episodes.
5. Compare one-step versus chunked action prediction.
6. Inject an observation perturbation and measure recovery.
7. Only then try Action Chunking with Transformers (ACT), Diffusion Policy, or
   a VLA.

Outcome: separate data quality/action representation from model scale.

## 19.11 Lab sequence D — foundation-policy adaptation

Choose Octo, OpenVLA, `openpi`, GR00T, or SmolVLA based on supported hardware,
compute, and license.

1. Pin paper-matching repository commit and checkpoint hash.
2. Run the official evaluator on one supported task.
3. Measure model memory, average latency, p95/p99 latency, and action age.
4. Visualize observation and de-normalized action samples.
5. Fine-tune on a tiny target dataset with a fixed validation/test split.
6. Compare against target-only BC at matched target data.
7. Test one held-out object/layout and one sensor fault.
8. Document which “generalist” behavior transferred and which did not.

Outcome: evaluate pretraining as a hypothesis rather than an identity.

## 19.12 Lab sequence E — language planning with safe skills

1. Create a simulator-only registry of exact, typed skills.
2. Implement local preconditions and deterministic stop.
3. Let a large language model (LLM) rank/propose only registered skills.
4. Add an affordance/feasibility score.
5. Test absent objects, stale perception, conflicting instructions, timeout,
   and network loss.
6. Log proposal, validation decision, execution result, and fallback.
7. Do not connect generated code or free-form motor values to hardware.

Outcome: gain semantic planning without surrendering physical authority.

## 19.13 Dependency and provenance record

For every external project, preserve:

```yaml
source_url: OFFICIAL_REPOSITORY_URL
commit: full_commit_sha
paper: exact_url_and_version
license: code / weights / data / assets
environment: lockfile_or_container_digest
accelerator: GPU, driver, CUDA/framework
command: exact_reproduction_command
config: resolved_config_path_and_hash
checkpoint: source_and_sha256
dataset: version/revision_and_split
local_changes: patch_or_clean
result: raw_metrics_and_evaluator_commit
```

This turns “I ran the GitHub project” into inspectable evidence.

## 19.14 Choosing a first open-source project

| Goal | Start with | Avoid starting with |
| --- | --- | --- |
| learn RL equations/code | book examples + CleanRL/Gymnasium | billion-parameter VLA |
| learn robot PPO/sim-to-real | Microduck + RSL-RL/mjlab | unmeasured custom hardware |
| learn manipulation environments | robosuite or ManiSkill | real-arm online exploration |
| learn demonstrations | LeRobot + BC | offline RL before dataset audit |
| study lifelong transfer | LIBERO | claiming real-world generality from one split |
| study VLA adaptation | Octo/OpenVLA/SmolVLA on supported benchmark | raw motor deployment |
| study world models | DreamerV3 or Temporal Difference Learning for Model Predictive Control, second generation (TD-MPC2), official benchmark | long-horizon hardware planning first |
| study language planning | SayCan-style typed skill simulator | unrestricted generated code on robot |

The smallest project that answers your question is the fastest route to genuine
understanding.

Continue with the [glossary and worked problems](20_glossary_and_worked_problems.md),
then select one reproduction ladder whose success criterion you can state
before running it.

## 19.15 Folded lab completion rubric

<details>
<summary>Show reference evidence for Lab sequences A–E</summary>

- **A — algorithm truth:** the small implementation and reference implementation
  agree on hand-calculated updates; a documented baseline is reproduced across
  three seeds; one ablation changes a single mechanism and reports uncertainty.
- **B — vectorized robot learning:** tests, smoke training, resolved manager
  terms, rollout tensor shape, ONNX export, and CPU rehearsal are connected by
  exact paths/hashes. The five-iteration result is labeled pipeline evidence,
  not locomotion performance.
- **C — imitation and data:** a dataset card precedes training; splits prevent
  episode/scene leakage; one-step and chunked BC use matching data; the
  perturbation test reports recovery and failure, not only nominal loss.
- **D — foundation adaptation:** official checkpoint evaluation passes before
  fine-tuning; target-only BC and adaptation see the same target examples;
  model memory, tail latency/action age, in-distribution and held-out results,
  licenses, and failures are preserved.
- **E — safe language planning:** the model can select only offered exact
  identifiers (IDs);
  local schema/preconditions reject absent/stale cases; every effect has
  timeout/cancel/fallback; generated prose/code and raw motor values have zero
  execution authority.

For every sequence, commit a provenance record like Section 19.13 and a
one-command evaluator. “Repository installed” is setup evidence, not a lab
result.

</details>
