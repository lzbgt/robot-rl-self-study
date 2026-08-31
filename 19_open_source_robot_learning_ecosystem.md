# 19. Open-Source Robot-Learning Ecosystem and Reproduction Labs

Open source makes a research claim inspectable, but the repository name is not
the experiment. This chapter maps major projects to their layer, shows what to
read, and gives a low-cost sequence of reproductions.

Project capabilities and versions change. Links were checked on 2026-09-01;
pin a release or commit before reproducing anything.

By the end of this chapter, you should be able to:

- identify which layer an open repository actually supplies;
- distinguish source availability from an executable paper reproduction;
- inspect a repository from command line to metric without expensive training;
- write an observation/action/data adapter before choosing a large model;
- isolate incompatible simulator and accelerator stacks;
- record code, data, weights, assets, and licenses as separate dependencies;
- select a minimal stack for a Microduck or JumpRover question; and
- complete a reproduction lab with a one-command evaluator.

The links and active-project statements were rechecked on 2026-09-01. A dated
check is a snapshot, not a promise that `main` will still behave identically.

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

### Library, framework, benchmark, and artifact

These words are often mixed:

- a **library** provides reusable code such as losses or environment interfaces;
- a **framework** supplies conventions tying several layers together;
- a **benchmark** fixes tasks, splits, metrics, and usually a protocol;
- a **dataset** contains recorded examples under a schema and license;
- a **checkpoint** is one learned parameter state; and
- a **deployment artifact** packages the model with preprocessing,
  normalization, output conversion, schema, and runtime evidence.

A benchmark score cannot be reproduced from a library name. A checkpoint is
not a deployment artifact. Write which noun you mean.

### Four contracts cross every stack

1. **Semantic contract:** what observations/actions/commands mean, including
   frame and units.
2. **Temporal contract:** capture time, control rate, horizon, delay, and queue
   behavior.
3. **Statistical contract:** normalization, split, randomization, seed, and
   checkpoint selection.
4. **Authority contract:** which layer may command what, with timeout and
   fallback.

Two projects can expose arrays of identical shape while violating all four.
For example, a 7D action may mean absolute end-effector pose in one repository
and six deltas plus a binary gripper in another.

### A layer-selection worksheet

Before installing, fill this table:

| Need | Existing answer | Missing evidence |
| --- | --- | --- |
| physics/robot | exact model and revision | identified residuals? |
| observation | keys, shapes, frames, rate | runtime sensor parity? |
| action | units, controller, bounds | hardware conversion? |
| learning | objective and data source | baseline/tuning budget? |
| evaluation | split, trials, metrics | failure labels/tails? |
| deployment | artifact and runtime | timing/fallback proof? |

If your research question lives in one row, avoid replacing all six rows at
once.

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

### Dataset-first tools beyond one foundation model

#### robomimic

- [Official repository](https://github.com/ARISE-Initiative/robomimic)
- Role: demonstration datasets, observation encoders, behavior-cloning and
  offline-policy baselines, and reproducible manipulation evaluation.
- Study: dataset sequence sampling, normalization, train/validation masks,
  action representation, and rollout evaluator.
- Use: a smaller bridge from Chapter 14's imitation equations to code before a
  multi-billion-parameter policy.

#### DROID

- [Dataset paper](https://arxiv.org/abs/2403.12945)
- [Official repository](https://github.com/droid-dataset/droid)
- Role: large, diverse real-robot manipulation data collected across many
  scenes and institutions with a common hardware platform.
- Study: collection policy, camera/state/action timing, task language, quality
  filters, splits, and which portions are actually downloaded.
- Caution: dataset size does not guarantee coverage of a particular action,
  camera, object, or failure distribution.

#### Datasets for Deep Data-Driven Reinforcement Learning (D4RL)

- [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)
  paper](https://arxiv.org/abs/2004.07219)
- Role: historically influential fixed offline-RL datasets and normalized
  scoring protocols.
- Study: behavior-policy quality/mixtures, dataset coverage, termination
  semantics, and benchmark-version warnings.
- Caution: a convenient normalized score can conceal what physical behavior
  improved; inspect raw return and failure trajectories.

Robot-data formats differ. Reinforcement Learning Datasets (RLDS), Hierarchical
Data Format version 5 (HDF5)-based
robomimic files, and LeRobot's episode-aware Parquet/video representation have
different loading and metadata conventions. Conversion must preserve episode
boundaries, capture timestamps, camera names, action semantics, language, and
normalization provenance. A successful file conversion is not proof of a
semantically lossless conversion.

### Whole-body motion and fast control research

#### ProtoMotions

- [Official repository](https://github.com/NVlabs/ProtoMotions)
- Role: research framework and artifacts for adversarial motion priors,
  reusable skill embeddings, and MaskedMimic-style physics control.
- Study: motion retargeting, masks, reference-state initialization, reward
  terms, resolved model cards, and simulator/humanoid assumptions.

#### Aligning Simulation and Real-World Physics (ASAP) and BeyondMimic

- [Official ASAP repository](https://github.com/LeCAR-Lab/ASAP)
- [Official BeyondMimic tracking
  repository](https://github.com/HybridRobotics/whole_body_tracking)
- Role: current open pipelines for agile humanoid motion tracking, physics
  alignment, retargeting, simulation, and deployment artifacts.
- Study: released versus roadmap components, motion-data licenses, robot
  revisions, state-estimation dependencies, and real/simulation log pairing.

#### FastTD3 and Holosoma

- [Official FastTD3 repository](https://github.com/younggyoseo/FastTD3)
- [Official Holosoma repository](https://github.com/amazon-far/holosoma)
- Role: massively parallel off-policy continuous-control research and a current
  sim-to-real humanoid locomotion/motion-control stack.
- Study: replay warm-up, update-to-data ratio, large batches, distributional
  critics, curriculum, randomization, evaluator, and deployment adapter.
- Caution: “15 minutes” is a recipe result on stated tasks/hardware, not a
  portable training-time guarantee.

These projects are valuable study references even for a small robot, but their
humanoid motion schemas and actuators are not Microduck or JumpRover drop-ins.

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

### Start with identity, not installation

For a paper-matching study, record the immutable revision before following
`main`-branch instructions:

```bash
git clone OFFICIAL_URL project-name
cd project-name
git rev-parse HEAD
git status --porcelain
git submodule status --recursive
git lfs ls-files
```

Git Large File Storage (Git LFS) pointers, submodules, downloaded assets, and
model-hub revisions can all sit outside the top-level commit. Record them. A
clean `git status` says tracked source is unchanged; it says nothing about
mutable remote weights.

Map the repository before running it:

```bash
rg --files | sed -n '1,160p'
rg -n "def (main|train|evaluate)|class .*Env|register\(" .
rg -n "observation|action|reward|termination|truncation" src tests
rg -n "normalize|mean|std|checkpoint|export" src scripts
```

Use the project's documented directories instead of literally assuming `src`
or `tests`. The goal is a route map, not a giant search transcript.

### Trace one command end to end

When documentation says `train --task X`, follow:

```text
console entry point
 -> argument/config parser
 -> task registry and resolved configuration
 -> environment constructor and wrappers
 -> observation/action transforms
 -> rollout or replay storage
 -> loss and optimizer update
 -> checkpoint writer
 -> evaluation entry point and metric aggregation
```

Write file/function names at every arrow. Default values can be changed by a
configuration composition system, environment variables, command line, or
checkpoint metadata. Preserve the **resolved** configuration after all
overrides, not only the source template.

For a paper equation, trace symbol to code and diagnostic. For example:

| Paper quantity | Code question | Runtime check |
| --- | --- | --- |
| discount factor | where is it applied at truncation? | hand fixture |
| action at step t | before or after scaling/clipping? | minimum, maximum, and units |
| reward at step t | raw or weighted term? | per-term episode mass |
| normalizer | fit on which split/state? | stored stats and parity test |
| success | environment flag or postprocessor? | labelled rollout replay |

### Isolation is part of experimental design

Simulator stacks commonly constrain Python, compiler, graphics driver, CUDA,
PyTorch/JAX, and native extensions differently. Use a separate lock/container
per project. Do not solve one repository by upgrading shared packages beneath
another.

A good isolation record includes:

- operating system and architecture;
- host driver and device model;
- environment lock or container image digest;
- language/runtime version;
- installed accelerator build and detected backend;
- simulator/asset version; and
- the exact smoke command and expected short output.

A container does not include the host graphics kernel driver. “Runs in Docker”
does not erase driver/device compatibility. Test the actual accelerator backend;
a package can import successfully while silently using the CPU.

A version range such as `torch>=2` is not a lock. A lock selects exact resolved
artifacts for a platform. Preserve wheel/index source because two files with the
same package version label may target different accelerator backends.

### Treat external artifacts as a supply chain

Before executing a setup or model:

1. inspect scripts that download or execute remote content;
2. use the official author organization and a pinned revision;
3. verify published hashes where available and record your own digest;
4. inspect code, weight, dataset, and asset licenses separately;
5. run untrusted artifacts without credentials or hardware-device access;
6. use the safest restricted weight loader the framework provides; and
7. record telemetry, model-hub, and experiment-tracking network behavior.

A popular checkpoint can still be mislabeled or replaced. A checksum proves
you used the same bytes later; provenance explains why those bytes were trusted.

### Write the semantic adapter before the model adapter

For any cross-project policy, fill this schema:

```yaml
observation:
  camera.front:
    shape: [3, 224, 224]
    color_order: RGB
    frame: camera_front_optical
    capture_rate_hz: 30
    normalization: checkpoint_stats_v2
  joint_position:
    order: [joint_0, joint_1, REPLACE]
    unit: rad
    reference: calibrated_zero
action:
  meaning: joint_position_delta
  order: [joint_0, joint_1, REPLACE]
  unit: rad
  rate_hz: 50
  horizon: 10
  bounds_source: runtime_manifest
timing:
  maximum_input_age_ms: REPLACE
  queue_policy: latest_value
termination:
  success: exact_definition
  failure: exact_definition
  timeout_is_truncation: true
```

Only after every `REPLACE` is resolved should shape conversion code exist.
Unit-test zero, sign, axis, bounds, joint order, image crop/color, chunk timing,
and normalization using a tiny recorded fixture.

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

Each rung needs an observable pass condition:

| Rung | Passing evidence | Common false pass |
| --- | --- | --- |
| schema audit | units/frames/rates/splits written | shapes only |
| import smoke | exact backend and versions printed | import on CPU |
| zero/random step | finite tensors and correct terminations | viewer opens |
| released eval | documented metric within declared tolerance | attractive video |
| train smoke | update, save, reload, evaluate | loss printed once |
| seeded reproduction | raw runs and uncertainty | best seed |
| ablation | one mechanism isolated | package defaults |
| new robot sim | identified model and held-out residual | plausible motion |
| processor/hardware | timing, faults, rollback, owner | untethered demo |

### Diagnose by the earliest failing rung

- Import failure is a dependency/packaging problem, not evidence against the
  algorithm.
- Released evaluation mismatch is an artifact/protocol problem; do not fine-tune
  yet.
- A smoke run that becomes not-a-number indicates numerical/configuration
  failure; more iterations do not cure it.
- Training success but export mismatch is a deployment-contract failure.
- Simulation success but processor timing failure is a systems failure.
- Hardware failure with simulator replay mismatch points toward model/adapter;
  matching replay but physical failure points toward unmodeled plant/sensing.

Keep the last passing artifact when moving up a rung. It becomes the reference
when a later layer fails.

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

## 19.13 Bringing a new robot into an open stack

Do not begin by implementing the environment class. Begin with a frozen
interface fixture representing one real or expected control cycle:

```python
fixture = {
    "capture_ns": 8_400_000_000,
    "joint_position_rad": [0.0, 0.1, -0.2, 0.0],
    "joint_velocity_rad_s": [0.0, 0.0, 0.0, 0.0],
    "imu_gyro_rad_s": [0.01, -0.02, 0.0],
    "command": [0.1, 0.0, 0.0, 0.0],
}

def validate_action(action):
    assert len(action) == 4
    assert all(-1.0 <= value <= 1.0 for value in action)
    return action
```

Replace the dimensions and meanings with the actual robot. The fixture enables
observation assembly, normalization, policy loading, action ordering, and
timeout tests before motors or simulation are available.

### Phase 0: selection

Choose the stack whose native abstractions require the fewest semantic
conversions. Score candidates on:

- robot/model import and contact/sensor support;
- action/controller type;
- required learning algorithm and data path;
- accelerator/hardware availability;
- export/runtime path;
- license and artifact access;
- maintained tests/documentation; and
- team familiarity and debugging visibility.

Do not add the scores as if they were objective measurements. A missing
realtime export path can be a veto even if every other row is strong.

### Phase 1: interface without dynamics

Implement and test:

1. observation and command dataclasses/schema;
2. stable joint/sensor order and frame names;
3. timestamp, sequence, validity, and timeout behavior;
4. action bounds and unit conversion;
5. recorded-fixture replay; and
6. policy manifest plus rejection on mismatch.

For JumpRover, this phase is valid now even while mechanics and realtime board
are unfinished. It should include cloud-option rejection and local-stop state
machine tests, but no claim about physical balance.

### Phase 2: measured digital twin

Once hardware exists, identify mass/inertia/geometry, actuator response,
friction, backlash, delay, saturation, thermal/voltage behavior, contacts, and
sensor noise. Replay the same excitation through hardware and simulation. Split
identification and held-out validation trajectories by experiment, not by
overlapping timesteps.

Use residual plots, not only one average error. A simulator that matches gentle
motion and misses braking/impact is not validated for obstacle stopping or
jumping.

### Phase 3: deterministic baseline and learning

Build a bounded classical/scripted controller that proves sign, authority,
timing, reset, and fallback. Then add the smallest learned component:

- residual action around a stable baseline;
- velocity/pose command tracker;
- state estimator or adaptation latent; or
- planner over an existing motor policy.

Freeze the baseline before comparison. A new stack must reproduce zero/random
tests, train/save/reload smoke, held-out evaluation, and export parity before a
long run.

### Phase 4: deployment gates

Use simulation, alternate-simulator, processor-in-loop, unpowered bench,
tethered, and bounded-floor tests in order. At every gate preserve input/output
logs, target-compute latency tails, watchdog/fault results, operator/intervention
rules, artifact hash, and rollback version.

This staged route separates “the open-source algorithm works” from “our adapter
is correct” and “our robot is safe enough for this experiment.”

## 19.14 Dependency and provenance record

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

### A minimal lab report tree

```text
reproduction/
  README.md                 # question, exact commands, expected output
  PROTOCOL.md               # predeclared comparison and stopping rule
  provenance.yaml           # revisions, hashes, licenses, hardware
  configs/resolved/         # one immutable config per run
  data/raw/                 # per-seed/trial records, never hand-edited
  analysis/                 # script and tested statistic helpers
  reports/                  # generated tables/figures and interpretation
  failures/                 # videos/logs plus category labels
```

Large artifacts may live in an external content-addressed store; keep their
digests and retrieval instructions in the tree. One command should regenerate
the report from raw results without retraining.

## 19.15 Choosing a first open-source project

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

## 19.16 Exercises

1. Classify Gymnasium, LIBERO, LeRobotDataset, and an exported Microduck policy
   as library/framework, benchmark, dataset, or deployment artifact. Explain
   where categories overlap.
2. Two policies both accept 14 floating-point values. Give four reasons their
   observations or actions may still be incompatible.
3. Why must a reproduction inspect termination separately from truncation?
4. Write a code-reading route from a command-line `train` entry point to the
   logged success metric.
5. Explain why `torch>=2` is not a reproducible accelerator environment.
6. A paper releases source and one weight file but no data mixture,
   normalization, or evaluator. What is open, and what claim cannot yet be
   reproduced?
7. List six fields that a conversion from RLDS to LeRobotDataset must preserve.
8. Design an isolation plan for Microduck, Isaac Lab, and an OpenVLA project on
   one workstation.
9. The official checkpoint evaluator produces a much lower result than the
   paper. What should happen before fine-tuning?
10. Choose a minimal open stack for adding modular obstacle avoidance to
    Microduck and name the baseline that must remain frozen.
11. Which JumpRover integration work is valid before mechanics/realtime board
    acceptance, and which work must wait?
12. Draft a provenance record for one external checkpoint, including code,
    weights, data, license, environment, evaluator, and local changes.

Continue with the [glossary and worked problems](20_glossary_and_worked_problems.md),
then select one reproduction ladder whose success criterion you can state
before running it.

## 19.17 Folded lab completion rubric

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

For every sequence, commit a provenance record like Section 19.14 and a
one-command evaluator. “Repository installed” is setup evidence, not a lab
result.

</details>

## 19.18 Folded exercise solutions

<details>
<summary>Show worked answers to Section 19.16</summary>

1. Gymnasium is primarily an environment-interface library plus a collection of
   reference environments. LIBERO is a benchmark with tasks, demonstrations,
   splits, and metrics. LeRobotDataset is a dataset format/class; a concrete
   repository using it is a dataset artifact. A correctly packaged exported
   Microduck policy, normalizer, schema, adapter, manifest, and evidence form a
   deployment artifact. LeRobot as a whole is also a framework, showing why the
   exact noun and scope matter.

2. The 14 values can differ in joint order, units, absolute versus delta
   meaning, normalization, reference pose/frame, scale/bounds, control rate, or
   actuator/controller interpretation. Shape checking catches none of those
   semantic mismatches.

3. True termination ends the task process, so a value target should usually not
   bootstrap past it. A time-limit truncation stops data collection while a
   valid future may still exist. Merging them changes return/value targets and
   can explain a reproduction gap even when every network parameter matches.

4. Follow console-script registration to argument/config parser, resolved task
   registry, environment constructor and wrappers, observation/action
   transforms, rollout/replay storage, loss/optimizer, checkpoint writer,
   evaluation loader, success computation, and aggregation/logger. Record the
   exact file/function and config field at each step.

5. The range can resolve to different releases over time and platforms. It
   omits Python, CUDA build/index, driver compatibility, companion packages,
   and native-extension versions. Preserve an exact platform lock or container
   digest and verify the detected accelerator backend.

6. The code bytes and checkpoint bytes are available under their stated
   licenses. Training-data provenance, exact preprocessing/inference, and the
   reported evaluation remain incomplete. One may inspect or perhaps run the
   model, but cannot claim reproduction of the paper result until the missing
   statistical and semantic contracts are reconstructed.

7. Preserve episode boundaries and identifiers, capture timestamps/rates,
   observation keys/camera names and image modes, state/action order/units/
   frames, language/task labels, success/termination/truncation, calibration,
   normalization provenance, and split/group metadata. Six are required; a
   robust conversion records all of these and verifies a sampled episode.

8. Keep three project directories and three exact locks/containers. Share only
   explicitly versioned data through neutral files. Record the host driver as a
   common constraint, verify each PyTorch/JAX backend independently, and never
   upgrade Microduck's lock to satisfy Isaac Lab or OpenVLA. Give containers no
   motor devices or credentials during untrusted checkpoint inspection.

9. Stop at that rung. Verify paper/checkpoint/evaluator revisions, assets,
   dependencies, configuration, seeds, preprocessing/normalization, metric, and
   hardware. Preserve the mismatch and seek an official issue/revision if
   needed. Fine-tuning would erase the clean test of released reproducibility.

10. Keep the existing Microduck velocity policy and 61D contract frozen. Add a
    depth or range source, timestamped local occupancy map, geometric local
    planner, and uncertainty-aware command governor that outputs the existing
    bounded velocity command. Compare against oracle geometry first. A new
    end-to-end depth-to-joint policy is a later separate project.

11. Now: define schemas/frames/units, recorded fixtures, policy manifests,
    cloud typed-option validation, stale/replay rejection, local-stop state
    machine, simulator interfaces, and provenance/testing infrastructure. Wait
    to train or validate dynamics-sensitive balance/jump/locomotion until
    geometry, mass/inertia, actuators, sensors, timing, braking, watchdogs, and
    held-out system identification are accepted.

12. Record official source URL and commit/submodules; checkpoint identifier,
    revision, digest, and retrieval date; paper version; data mixture/version
    and license; code/weight/data/asset licenses; exact lock/container and
    accelerator; resolved configuration; evaluator commit/command; raw result;
    and a clean status or attached local patch. A digest establishes identity,
    while this context establishes provenance.

</details>
