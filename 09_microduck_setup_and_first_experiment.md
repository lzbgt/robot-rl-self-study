# 9. Setup and First Experiment

This chapter builds confidence in layers. Do not begin with a 20-hour training
run. First prove the checkout, dependency lock, task registry, central
processing unit (CPU) tests, graphics processing unit (GPU) simulation, short
Proximal Policy Optimization (PPO) loop, checkpoint load, and viewer.

The goal is not merely to make a command stop without an error. A first
experiment is a chain of falsifiable claims:

```text
source and lock are known
  -> dependencies import
  -> task registers
  -> model and selectors resolve
  -> simulation steps with finite numbers
  -> rollout and PPO update complete
  -> checkpoint reloads
  -> rendered behavior matches the evaluated artifact
```

Each stage should fail cheaply before the next stage consumes more time or
money. A long run cannot repair a broken observation permutation, and an
attractive video cannot prove that the checkpoint came from the configuration
you intended.

## 9.1 What the first experiment can establish

Separate four kinds of claim:

| Claim | Smallest useful evidence | What the evidence does not prove |
| --- | --- | --- |
| software integrity | clean sync, import, CPU tests | GPU kernels work |
| integration integrity | five-iteration GPU smoke run | a useful skill was learned |
| learning behavior | multi-seed metric and rollout battery | hardware transfer |
| deployment behavior | timed runtime rehearsal and staged robot test | unrestricted safety |

This hierarchy prevents a common reasoning error: treating success at a lower
layer as success at every higher layer. For example, a smoke run establishes
that gradients can be computed; it says almost nothing about the statistical
quality of the learned gait.

Before typing the training command, write a one-sentence hypothesis and an
acceptance test. A useful first hypothesis is:

> At the pinned commit and lockfile, the flat-walking task can complete five
> PPO updates on this GPU with 61 finite actor observations, 14 finite actions,
> a saved checkpoint, and no negative-valued term appearing as a positive
> penalty contribution.

The hypothesis is narrow enough to test. “Training works” is not.

## 9.2 Requirements and the compatibility chain

For local GPU training you need:

- Linux;
- an NVIDIA Compute Unified Device Architecture (CUDA)-capable GPU and working
  driver;
- enough disk space for Python/CUDA packages and checkpoints;
- Git; and
- [`uv`](https://docs.astral.sh/uv/) for the locked Python 3.12 environment.

The project requires Python `>=3.12,<3.13`; let `uv` honor the repository
configuration rather than installing packages into an unrelated system Python.

Training can also be submitted to Hugging Face Jobs. Playback and deployment
rehearsal still need a local environment that can load the model.

A modern robot simulator is a stack, not one program:

```text
NVIDIA driver
  -> CUDA-enabled PyTorch wheel
  -> Warp kernels and PyTorch/Warp interoperation
  -> MuJoCo and MuJoCo Warp
  -> mjlab environment managers
  -> RSL-RL learner
  -> Microduck task and model assets
```

An arrow means “depends on a compatible interface from.” The installed driver
need not have the same version number as the CUDA runtime bundled with a
PyTorch wheel, but it must be new enough to support that runtime. Similarly,
seeing a device in `nvidia-smi` proves that the operating-system driver sees the
GPU; it does not prove that the selected Python wheel contains CUDA kernels.

The repository's `pyproject.toml` expresses direct requirements and source
rules. Its `uv.lock` records the exact resolved package graph and wheel
selection. This distinction matters:

- a version *constraint* describes allowed solutions;
- a lockfile records the solution that was tested; and
- the Git commit identifies the code and robot assets that consumed it.

Together, commit plus lockfile are the minimum software identity of a run.

## 9.3 Clone and synchronize

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
uv sync
```

`uv sync` is the reproducibility test. A package that exists only because it
was manually installed into a developer's environment will be absent on a
clean machine or remote job.

On Linux running the **64-bit Arm architecture (AArch64)**, set a longer
first-download timeout if needed:

```bash
export UV_HTTP_TIMEOUT=600
uv sync
```

The project pins Torch directly and selects the CUDA 12.9 wheel index on
AArch64. Do not remove the apparently redundant direct pin; `uv` source routing
depends on it.

For a strict reproduction, ask `uv` to refuse an implicit lock update:

```bash
uv sync --frozen
```

Use ordinary `uv sync` while intentionally changing dependencies; inspect and
commit the resulting lockfile diff. Use `--frozen` when the experiment must
consume the recorded resolution exactly.

Do not “fix” a clean-environment failure with an unrecorded `pip install`.
Change the project dependency, regenerate the lock, and repeat from a clean
environment. Otherwise the next machine will rediscover the same failure.

## 9.4 Capture a baseline manifest

Run identity should be recorded before training, not reconstructed after a
surprising result:

```bash
git rev-parse HEAD
git status --short
sha256sum uv.lock
uname -a
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader
uv run python -VV
```

If `git status --short` is non-empty, save the patch with the run. A commit hash
alone does not identify uncommitted code. Never put access tokens or the full
environment variable set in the manifest; provenance and secrets have
different purposes.

Two random streams are exposed by the live task interface:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.seed 101 \
    --agent.seed 101 \
    --env.scene.num-envs 64 \
    --agent.max-iterations 5
```

The environment seed controls stochastic simulation setup; the agent seed
controls learner initialization and sampling. Fixing both makes a debugging
reproduction easier, but it does not make GPU execution mathematically
bit-for-bit deterministic. For a scientific result, vary seeds deliberately
and report the distribution rather than choosing the most attractive run.

## 9.5 Verify the GPU stack

```bash
nvidia-smi

uv run python - <<'PY'
import torch
import warp as wp

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
wp.init()
PY
```

Do not proceed to a long run if `torch.cuda.is_available()` is false. A visible
GPU in `nvidia-smi` is necessary but not sufficient; the Python wheel must also
include compatible CUDA support.

Interpret failures from the bottom of the dependency chain upward:

1. if `nvidia-smi` fails, repair driver/device access;
2. if it passes but Torch reports no CUDA runtime, inspect the installed wheel;
3. if Torch works but Warp initialization fails, inspect Warp/runtime
   compatibility;
4. if both work but task import fails, inspect Python dependencies and entry
   points; and
5. only after those pass should you debug a task configuration.

Changing reward weights cannot fix any of the first four layers.

## 9.6 Discover the live tasks

```bash
uv run list-envs
```

Look for `Mjlab-Velocity-Flat-MicroDuck`. If it is missing, task package entry
points were not installed or importing `mjlab_microduck.tasks` failed. Fix that
before debugging PPO.

The task identifier (ID) is a contract key. Similar names may select different
robots, contact models, terrain, or backlash variants. Copy it from the live
registry and record it verbatim.

## 9.7 Run the CPU regression suite

```bash
uv run --with pytest pytest tests/
```

These tests are more than unit-test decoration. They protect properties such
as:

- the 61D observation family contract;
- servo indices on roller/backlash models;
- reward sign conventions;
- not-a-number (NaN) handling;
- reset and curriculum wiring; and
- task registration.

A passing suite does not prove a good policy, but a failing invariant makes an
expensive run untrustworthy.

## 9.8 Run the five-iteration smoke train

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 \
    --agent.max_iterations 5
```

What this should prove:

1. the robot and sensors compile;
2. Warp can execute on the selected GPU;
3. actor observation shape is 61 and action shape is 14;
4. every reward and termination can run;
5. PPO can collect data, compute losses, and update weights;
6. a checkpoint and resolved parameters can be written; and
7. values remain finite for this short run.

What it does **not** prove: that the robot has learned to walk. Five iterations
are a software smoke test.

The default logger uses Weights & Biases. Authenticate with `wandb login` when
you want cloud tracking, or use W&B's offline mode for a local-only experiment:

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 \
    --agent.max_iterations 5
```

The smoke-test log should be read, not merely awaited. Check at least:

- observations, actions, rewards, advantages, losses, and gradients are finite;
- every penalty's weighted episode contribution has the intended sign;
- episode length is plausible for the termination logic;
- the requested environment and iteration counts were resolved; and
- the expected checkpoint and configuration snapshots exist.

If a self-negating penalty is accidentally assigned a negative weight, it
becomes a positive reward for violating the constraint. A total return can rise
while the robot learns exactly the wrong behavior, so inspect named terms.

## 9.9 Budget arithmetic: what did the smoke test sample?

Robotic Systems Lab reinforcement learning (RSL-RL) collects 24 policy steps
from each environment before one PPO update.
For $N$ parallel environments and $I$ iterations, the number of transitions is

```math
N_{\mathrm{transition}}=N\times 24\times I.
```

The smoke run therefore collects

```math
64\times 24\times 5=7{,}680
```

transitions. At the 50 hertz policy rate, each individual environment advances

```math
\frac{24\times5}{50}=2.4\ \mathrm{s},
```

while all environments together represent

```math
\frac{7{,}680}{50}=153.6\ \mathrm{s}
```

of aggregate simulated experience. The robot did **not** walk one continuous
153.6-second episode; 64 worlds produced short trajectories in parallel.

This arithmetic explains both the value and the limitation of the smoke test.
It touches thousands of transitions and several update paths, but each world
has seen only 2.4 seconds and only one learner seed has been sampled.

Measure two operational quantities:

```math
\mathrm{throughput}=\frac{N_{\mathrm{transition}}}{t_{\mathrm{wall}}},
\qquad
\mathrm{aggregate\ real\mbox{-}time\ factor}
=\frac{N_{\mathrm{transition}}/50}{t_{\mathrm{wall}}}.
```

Here $t_{\mathrm{wall}}$ is elapsed wall-clock seconds. Throughput helps select
an environment count; it is not a measure of policy quality.

## 9.10 Inspect the run artifacts

```bash
find logs/rsl_rl/velocity -maxdepth 3 -type f | sort | tail -40
```

Open the newest run directory and inspect:

```bash
sed -n '1,160p' logs/rsl_rl/velocity/<run>/params/agent.yaml
sed -n '1,220p' logs/rsl_rl/velocity/<run>/params/env.yaml
```

Questions to answer:

- How many environments actually ran?
- How many iterations were requested?
- Is actor normalization on?
- What is the simulator timestep and action decimation?
- Which robot spec function was resolved?

This habit prevents evaluating a checkpoint under a task you merely assumed it
used.

Also preserve the exact source identity and content hashes:

```bash
sha256sum logs/rsl_rl/velocity/<run>/model_*.pt
git diff --binary > logs/rsl_rl/velocity/<run>/working-tree.patch
```

Only create `working-tree.patch` when the tree was dirty, and inspect it for
secrets before uploading. The resolved `env.yaml` and `agent.yaml` answer what
the program actually instantiated; the source file answers why those values
were chosen. Keep both.

## 9.11 Play a local checkpoint

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
    --checkpoint-file logs/rsl_rl/velocity/<run>/model_<iteration>.pt \
    --num-envs 1 \
    --viewer native
```

Or load a run from W&B:

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity>/mjlab_microduck/<run_id> \
    --num-envs 1
```

The viewer resamples training-style commands, so the robot may change
direction. The command arrow is the requested local velocity, not an obstacle
sensor or global route.

Verify these startup facts in the terminal:

```text
checkpoint name is the one you intended
environment device is cuda:0 (or the explicitly selected device)
actor observation shape is (61,)
action shape is 14
actor first layer has 61 inputs
```

An actuator-selector warning that says 14 joints were matched while similarly
named sites also match is informational in the current stack; the resolved
action dimension must still be 14.

Playback is a controlled evaluation, not decoration. Record the command, seed,
checkpoint digest, task, and whether training noise/randomization is disabled
or retained. If two videos differ in all of those variables, their visual
difference cannot be attributed to the checkpoint alone.

## 9.12 Record a finite rollout

A video is durable evidence and exits after the requested number of policy
steps:

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
    --checkpoint-file logs/rsl_rl/velocity/<run>/model_<iteration>.pt \
    --num-envs 1 \
    --video True \
    --video-length 500
```

At 50 Hz, 500 policy steps represent 10 seconds of simulated behavior. The
video is written under the checkpoint run's `videos/play/` directory.

When comparing checkpoints, keep the task identifier (ID), commands, random seed or evaluation
battery, camera, and video length consistent.

## 9.13 Find a useful environment count

Parallel worlds improve device utilization until memory, kernel scheduling, or
the learner becomes the bottleneck. More environments do not automatically
mean faster learning in wall time. They also change the batch size per update:

```math
B=N_{\mathrm{env}}\times N_{\mathrm{step}}.
```

With 24 steps per environment, 4,096 worlds produce a batch of 98,304
transitions per iteration. If four minibatches are used, the nominal minibatch
contains 24,576 transitions before accounting for the implementation's epoch
reshuffling.

Benchmark resource scaling with the same task, seed, and short iteration count:

| Environments | Peak GPU memory | Transitions/s | Aggregate real-time factor | Result |
| ---: | ---: | ---: | ---: | --- |
| 64 | measure | measure | calculate | baseline |
| 256 | measure | measure | calculate | |
| 1,024 | measure | measure | calculate | |
| 4,096 | measure | measure | calculate | |

Select the largest stable count near the throughput plateau, leaving memory
headroom for compilation and transient allocations. An out-of-memory error is
an infrastructure limit, not evidence that the reward or algorithm is wrong.

## 9.14 Start a full run only after the smoke test

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096
```

Environment count is a throughput/memory tradeoff. Do not assume 4,096 fits
every GPU or task. Reduce it if compilation or memory fails; do not change the
task merely to hide a resource problem.

Remote alternative:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096 \
    --hf-jobs
```

See the Microduck repository's
[`scripts/hf/README.md`](https://github.com/pollen-robotics/microduck_rl/blob/main/scripts/hf/README.md)
for authentication,
job flavor, namespace, and retrieval details.

Before committing 20 hours, fill this gate:

| Question | Pass evidence |
| --- | --- |
| Is the exact code recoverable? | commit plus saved dirty patch |
| Is the environment reproducible? | lock hash and resolved configuration |
| Does the cheap suite pass? | command and test summary |
| Does the GPU smoke pass? | finite five-iteration log and artifact |
| Are reward signs credible? | named term audit |
| Is the skill measurable? | fixed evaluation battery and success definition |
| Can progress be recovered? | checkpoint upload/copy tested |

A “no” does not always forbid exploratory training, but it must be an explicit
risk rather than a surprise discovered after the budget is spent.

## 9.15 Resume rather than discard a useful run

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096 \
    --agent.run-name resume \
    --agent.load-checkpoint model_29999.pt \
    --agent.resume True
```

Resume only when the environment and observation/action contract are
compatible with the checkpoint. A checkpoint is not a generic starting point
for an arbitrary task.

Resume semantics deserve care. Loading network weights is different from
restoring optimizer state, observation-normalizer statistics, iteration count,
and curriculum phase. `--agent.resume True` requests a training continuation;
an ad-hoc weight load may instead be a warm start. Record which operation was
used because they define different experiments.

## 9.16 Diagnose by layer, not by guess

When a run fails, first classify the failure:

| Layer | Typical symptom | First evidence to inspect |
| --- | --- | --- |
| host/device | GPU absent, allocation failure | `nvidia-smi`, memory use |
| package/runtime | import, symbol, kernel error | lock, wheel versions, minimal import |
| task construction | selector or shape failure | registry, resolved config, CPU tests |
| numerical learning | NaN, exploding loss | first bad iteration and named tensors |
| objective | stable but undesirable behavior | per-term reward mass and rollouts |
| evaluation | contradictory videos/metrics | checkpoint hash and fixed battery |

This order is causal. If the task cannot resolve its joints, changing PPO's
learning rate adds a second variable without addressing the first failure.

Use a minimal reproduction that preserves the failing layer. For a numerical
failure, save the earliest offending checkpoint/configuration and reduce the
environment count. For a behavior failure, do not reduce away the command or
terrain slice on which it occurs.

### Common first-run failures

### GPU visible to the driver but not Torch

Symptom: `nvidia-smi` works, while Torch reports zero devices or `mjlab` fails
while selecting a GPU. Inspect the installed Torch build and CUDA runtime. On
AArch64, preserve the repository's direct Torch pin and CUDA index routing.

### Viewer does not open

Check `DISPLAY` on the X Window System version 11 (X11), or use a finite
video/remote viewer appropriate to the
machine. Do not interpret a display failure as a policy failure; first confirm
the simulation process and checkpoint load.

### Observation-size mismatch

The current family expects 61 actor inputs. A 51D legacy Open Neural Network
Exchange (ONNX) policy or a task that deleted command slots is incompatible.
Use the correct checkpoint/task or intentionally version the runtime contract.

### NaN during training

Record the task, iteration, offending term, and resolved configuration. Run the
CPU NaN regression tests, reduce the case to a reset/step reproduction, and
inspect contacts and sensors. Simply restarting an unchanged long run loses the
most useful evidence.

### A successful smoke policy falls immediately

Expected. Five iterations proved the pipeline, not locomotion.

### Training is finite but no skill appears

Check whether the main task reward grows, not only whether total return grows.
Then inspect command coverage, episode length, curriculum stage, observation
normalization, and a rollout. Regularizers can become less negative while the
robot still does nothing. This is an objective/data diagnosis, not a reason to
reinstall CUDA.

## 9.17 Lab report template

Record this after the first experiment:

```text
commit:
task ID:
GPU and Torch/CUDA versions:
command:
run directory:
checkpoint:
actor/action shapes:
tests result:
smoke result:
viewer or video result:
first anomaly observed:
```

This small report is the beginning of reproducible reinforcement learning (RL)
work.

Add an experiment table before a multi-run study:

| Run | Code/config difference | Env seed | Agent seed | Hypothesis | Acceptance metric |
| --- | --- | ---: | ---: | --- | --- |
| A | baseline | 101 | 101 | pipeline is finite | smoke checklist |
| B | baseline | 202 | 202 | result is not seed-only | same evaluation battery |
| C | one intended change | 101 | 101 | change improves named failure | predeclared metric |

Run C is a paired comparison with A because its seeds match. B estimates seed
sensitivity. Many more seeds are normally needed for a publishable conclusion,
but this structure already prevents changing code, randomization, and seeds at
once.

Continue with
[the anatomy of a Microduck environment](10_microduck_environment_anatomy.md).

## 9.18 Exercises

1. For 256 environments, 24 steps per environment, and 10 iterations,
   calculate transitions, per-environment simulated time, and aggregate
   simulated time at 50 hertz.
2. Explain why `nvidia-smi` succeeding while `torch.cuda.is_available()` is
   false points to a different layer than a missing task ID.
3. A 64-environment smoke run passes, but 4,096 environments run out of memory.
   What has been falsified? What remains supported?
4. Why are both a Git commit and a dirty patch needed when the worktree has
   uncommitted changes?
5. Design a four-row environment-count benchmark and state which variables
   must remain fixed.
6. A run's total return rises, the velocity-tracking term is flat, and the
   action-rate penalty approaches zero. Give the most likely interpretation
   and the next two measurements.
7. Explain why one fixed seed is useful for debugging but insufficient for a
   learning claim.
8. Distinguish a training resume from a warm start. Name at least three pieces
   of state that may differ.
9. Draft an acceptance test for “the exported checkpoint is the policy shown
   in this video.”
10. Find and correct the stale flags in this deliberately wrong command:

    ```bash
    uv run train Mjlab-Velocity-Flat-MicroDuck \
        --num-envs 64 --max-iterations 5
    ```

## 9.19 Folded solutions and example report

<details>
<summary>Show exercise solutions</summary>

1. The transition count is
   $256\times24\times10=61{,}440$. Each environment advances
   $24\times10/50=4.8$ seconds. Aggregate simulated time is
   $61{,}440/50=1{,}228.8$ seconds. The latter is summed parallel experience,
   not the duration of one trajectory.
2. `nvidia-smi` tests driver/device visibility, while Torch tests the Python
   wheel and CUDA-enabled runtime. A missing task ID occurs higher in the
   stack, usually at package entry-point registration or task import. The two
   failures therefore demand different minimal reproductions.
3. The claim “4,096 worlds fit this device/configuration” is falsified. The
   passing smoke run still supports task construction, finite stepping, PPO
   integration, and checkpoint writing at 64 worlds. It does not support skill
   quality at either count.
4. The commit identifies tracked repository content. The dirty patch records
   changes not represented by that commit. Without it, another learner cannot
   reconstruct the program that ran.
5. Use, for example, 64, 256, 1,024, and 4,096 worlds. Hold commit, lock,
   task, both seeds, rollout steps, iteration count, logging overhead, and GPU
   fixed. Measure peak memory, transitions per second, failures, and aggregate
   real-time factor. Select based on the stable throughput curve, not count
   alone.
6. The learner may be discovering inactivity: smoother actions reduce the
   regularization cost while the task is not improving. Inspect a fixed-command
   rollout and the weighted mass of every reward term. Also compare episode
   length/terminations to see whether survival or early termination explains
   the return.
7. Fixed seeds make an observed change reproducible and permit paired
   debugging. They reveal only one initialization and one stochastic path.
   Several independently seeded runs are needed to estimate variability and
   distinguish a robust effect from luck.
8. A resume normally restores weights, optimizer moments, normalization
   statistics, iteration/curriculum position, and sometimes random-number
   state. A warm start may load only weights into a newly initialized run.
   These histories produce different update dynamics even at the same first
   visible checkpoint.
9. Record the checkpoint's Secure Hash Algorithm 256-bit (SHA-256) digest,
   export command and output digest, task and resolved config, inference
   command, seed/command sequence, and video path. The runtime log must print
   or record the loaded artifact digest and expected 61-input/14-output
   contract. Replaying that manifest should reproduce the evaluated artifact.
10. The live hierarchical flags are:

    ```bash
    uv run train Mjlab-Velocity-Flat-MicroDuck \
        --env.scene.num-envs 64 \
        --agent.max-iterations 5
    ```

</details>

<details>
<summary>Show an annotated first-experiment report</summary>

This is a format example, not evidence from your machine. Replace every value
and retain the commands/logs that support it.

```text
commit:                 FULL_GIT_SHA (plus dirty patch, if any)
task ID:                Mjlab-Velocity-Flat-MicroDuck
GPU and Torch/CUDA:     exact device / torch build / CUDA runtime
command:                uv run train ... --env.scene.num-envs 64
                        --agent.max-iterations 5
run directory:          logs/<experiment>/<timestamp-or-run>
checkpoint:             exact model_*.pt and Secure Hash Algorithm 256-bit
                        (SHA-256) digest
actor/action shapes:    61 / 14, confirmed from runtime output
tests result:            N passed, M skipped; paste command and date
smoke result:            completed 5 iterations; finite observations/rewards
viewer or video result:  pipeline rendered; locomotion quality not claimed
first anomaly observed:  exact warning, metric, frame, or "none observed"
```

The important distinctions are observed versus assumed, pipeline success
versus learned-skill success, and a mutable filename versus a content hash.

</details>
