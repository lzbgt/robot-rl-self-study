# 9. Setup and First Experiment

This chapter builds confidence in layers. Do not begin with a 20-hour training
run. First prove the checkout, dependency lock, task registry, CPU tests, GPU
simulation, short PPO loop, checkpoint load, and viewer.

## 9.1 Requirements

For local GPU training you need:

- Linux;
- a CUDA-capable NVIDIA GPU and working driver;
- enough disk space for Python/CUDA packages and checkpoints;
- Git; and
- [`uv`](https://docs.astral.sh/uv/) for the locked Python 3.12 environment.

The project requires Python `>=3.12,<3.13`; let `uv` honor the repository
configuration rather than installing packages into an unrelated system Python.

Training can also be submitted to Hugging Face Jobs. Playback and deployment
rehearsal still need a local environment that can load the model.

## 9.2 Clone and synchronize

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
uv sync
```

`uv sync` is the reproducibility test. A package that exists only because it
was manually installed into a developer's environment will be absent on a
clean machine or remote job.

On Linux AArch64, set a longer first-download timeout if needed:

```bash
export UV_HTTP_TIMEOUT=600
uv sync
```

The project pins Torch directly and selects the CUDA 12.9 wheel index on
AArch64. Do not remove the apparently redundant direct pin; `uv` source routing
depends on it.

## 9.3 Verify the GPU stack

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

## 9.4 Discover the live tasks

```bash
uv run list-envs
```

Look for `Mjlab-Velocity-Flat-MicroDuck`. If it is missing, task package entry
points were not installed or importing `mjlab_microduck.tasks` failed. Fix that
before debugging PPO.

## 9.5 Run the CPU regression suite

```bash
uv run --with pytest pytest tests/
```

These tests are more than unit-test decoration. They protect properties such
as:

- the 61D observation family contract;
- servo indices on roller/backlash models;
- reward sign conventions;
- NaN handling;
- reset and curriculum wiring; and
- task registration.

A passing suite does not prove a good policy, but a failing invariant makes an
expensive run untrustworthy.

## 9.6 Run the five-iteration smoke train

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

## 9.7 Inspect the run artifacts

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

## 9.8 Play a local checkpoint

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

## 9.9 Record a finite rollout

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

When comparing checkpoints, keep task ID, commands, random seed or evaluation
battery, camera, and video length consistent.

## 9.10 Start a full run only after the smoke test

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

## 9.11 Resume rather than discard a useful run

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

## 9.12 Common first-run failures

### GPU visible to the driver but not Torch

Symptom: `nvidia-smi` works, while Torch reports zero devices or `mjlab` fails
while selecting a GPU. Inspect the installed Torch build and CUDA runtime. On
AArch64, preserve the repository's direct Torch pin and CUDA index routing.

### Viewer does not open

Check `DISPLAY` on X11 or use a finite video/remote viewer appropriate to the
machine. Do not interpret a display failure as a policy failure; first confirm
the simulation process and checkpoint load.

### Observation-size mismatch

The current family expects 61 actor inputs. A 51D legacy ONNX policy or a task
that deleted command slots is incompatible. Use the correct checkpoint/task or
intentionally version the runtime contract.

### NaN during training

Record the task, iteration, offending term, and resolved configuration. Run the
CPU NaN regression tests, reduce the case to a reset/step reproduction, and
inspect contacts and sensors. Simply restarting an unchanged long run loses the
most useful evidence.

### A successful smoke policy falls immediately

Expected. Five iterations proved the pipeline, not locomotion.

## 9.13 Lab report template

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

This small report is the beginning of reproducible RL work.

Continue with
[the anatomy of a Microduck environment](10_microduck_environment_anatomy.md).

## 9.14 Folded example report

<details>
<summary>Show an annotated first-experiment report</summary>

This is a format example, not evidence from your machine. Replace every value
and retain the commands/logs that support it.

```text
commit:                 FULL_GIT_SHA (plus dirty patch, if any)
task ID:                Mjlab-Velocity-Flat-MicroDuck
GPU and Torch/CUDA:     exact device / torch build / CUDA runtime
command:                uv run train ... --num-envs 64 --max-iterations 5
run directory:          logs/<experiment>/<timestamp-or-run>
checkpoint:             exact model_*.pt and SHA-256
actor/action shapes:    61 / 14, confirmed from runtime output
tests result:            N passed, M skipped; paste command and date
smoke result:            completed 5 iterations; finite observations/rewards
viewer or video result:  pipeline rendered; locomotion quality not claimed
first anomaly observed:  exact warning, metric, frame, or "none observed"
```

The important distinctions are observed versus assumed, pipeline success
versus learned-skill success, and a mutable filename versus a content hash.

</details>
