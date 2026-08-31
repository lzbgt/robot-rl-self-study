# 12. Export, Deployment, and Sim-to-Real

A training checkpoint is not the deployed policy. Deployment is a contract
between the exported graph, observation producer, command source, timing loop,
action mapping, actuator interface, and safety system.

## 12.1 Artifact flow

```text
model_<iteration>.pt
  actor + critic + normalizers + optimizer + training state
        |
        | scripts/export.py
        v
policy.onnx
  actor inference graph + actor observation normalizer + metadata
        |
        | scripts/infer_policy.py
        v
CPU MuJoCo rehearsal using deployment-style observations and commands
        |
        | real runtime integration and staged safety tests
        v
14 joint targets at 50 Hz on Microduck
```

The critic, reward functions, terminations, randomizers, and Proximal Policy
Optimization (PPO) optimizer are not deployed. Their job was to shape the actor
during training.

## 12.2 Export through the mandatory path

Export a local checkpoint:

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
    --checkpoint-file logs/rsl_rl/velocity/<run>/model_<iteration>.pt \
    --onnx-file walk.onnx
```

Or export from W&B:

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity>/mjlab_microduck/<run_id> \
    --onnx-file walk.onnx
```

The script:

1. loads the play configuration for the exact task identifier (ID);
2. constructs the actor and critic architecture;
3. restores the checkpoint;
4. obtains the deterministic inference policy;
5. exports the actor with its `EmpiricalNormalization` submodule; and
6. attaches metadata such as joint names, defaults, action scale, commands,
   and observation names.

Never hand-convert the checkpoint. In-simulator playback normalizes
observations and can conceal a missing-normalizer deployment bug.

## 12.3 Inspect the Open Neural Network Exchange (ONNX) interface

```bash
uv run python - <<'PY'
import numpy as np
import onnxruntime as ort

path = "walk.onnx"
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print("input:", input_info.name, input_info.shape, input_info.type)
print("output:", output_info.name, output_info.shape, output_info.type)
print("metadata:", session.get_modelmeta().custom_metadata_map)

obs = np.zeros((1, 61), dtype=np.float32)
action = session.run([output_info.name], {input_info.name: obs})[0]
print("action shape:", action.shape)
print("finite:", np.isfinite(action).all())
PY
```

For this policy family, expect one batch of 61 float inputs and 14 float
outputs. A finite result for an all-zero synthetic observation proves graph
execution, not meaningful robot behavior.

## 12.4 Rehearse the deployment loop

Current policies use the unified 13D command block, so include
`--new-cmd-obs`:

```bash
uv run scripts/infer_policy.py \
    --walking walk.onnx \
    --new-cmd-obs \
    --lin-vel-x 0.20
```

This script runs ONNX Runtime on the central processing unit (CPU) and builds
the same observation order the robot runtime must build. It also applies action
delay and maps actions as:

```python
observation = np.concatenate([
    base_angular_velocity,       # 3
    projected_gravity,           # 3
    relative_joint_position,     # 14
    joint_velocity,              # 14
    previous_action,             # 14
    command,                     # 13
]).astype(np.float32)

action = onnx_session.run(..., {input_name: observation[None, :]})[0]
target_position = default_pose + action * action_scale
```

The terminal, not the viewer window, receives keys:

```text
UP/DOWN            forward command
LEFT/RIGHT         lateral command for the walking model
A/E                turn command
SPACE              zero velocity commands
H                  enter/leave head-command mode
B                  enter/leave body-command mode
P                  apply a random push
Q                  quit
```

The body-pose user interface (UI) exists for policy families that train those
slots. The main velocity checkpoint has body-pose reward weight zero, so UI
availability is not proof of learned body control.

## 12.5 The exact 61D runtime contract

| Slice | Values | Runtime source |
| --- | ---: | --- |
| base angular velocity | 3 | calibrated inertial measurement unit (IMU), body frame, rad/s |
| projected gravity | 3 | calibrated orientation projected into body frame |
| relative joint position | 14 | servo encoders minus exact HOME offsets |
| joint velocity | 14 | encoder-derived velocity in matching order/units |
| previous action | 14 | last actor output, not measured joint target |
| twist | 3 | application/planner command |
| head pose | 4 | requested joint deltas from HOME |
| body pose | 6 | requested body deltas or zeros |

For every slice, verify:

```text
dimension
ordering
dtype
units
frame
sign
offset
latency
noise/filter behavior
reset/initial value
```

The ONNX graph normalizes the assembled raw observation. Do not pre-normalize
again in the runtime.

## 12.6 Timing is part of the policy

Microduck training assumes:

```text
physics model step        5 ms
policy period            20 ms (50 Hz)
action decimation         4
unfiltered policy output
modeled sensor/action delays
```

A real loop should use a monotonic clock, timestamp sensor snapshots, detect
missed deadlines, and choose a documented safe response. Running inference
“whenever Linux schedules it” without observing jitter changes the effective
delay distribution.

Do not silently reuse stale commands forever. High-level commands need an
owner, sequence/timestamp, and expiry behavior. The local safety controller
must be able to stop or hold the robot without a cloud round trip.

## 12.7 Sim-to-real gap

The sim-to-real gap is the difference between the transition distribution used
for training and the real robot:

```math
P_{sim}(s_{t+1}\mid s_t,a_t)
\neq
P_{real}(s_{t+1}\mid s_t,a_t)
```

Microduck reduces this gap with:

- measured computer-aided design (CAD) geometry and mass properties;
- the Better Actuator Models (BAM) voltage-controlled XL330 model;
- friction, armature, mass/inertia, and center of mass (CoM) randomization;
- battery voltage and load-sag variation;
- sensor noise, bias, IMU misalignment, and latency;
- command delay;
- backlash models with encoder-consistent observations; and
- pushes and varied initial states.

Randomization should cover plausible uncertainty, not every imaginable value.
An unrealistically wide distribution can teach an overly conservative policy
or make the task unsolvable.

## 12.8 Calibrate before randomizing

Use a parameter workflow:

```text
measure real subsystem
        v
fit nominal simulator parameter
        v
validate on held-out motion
        v
estimate repeatability and uncertainty
        v
randomize around the nominal range
```

Examples:

- identify actuator friction and torque response on a bench;
- measure battery sag under representative load;
- compare commanded and measured step responses;
- measure encoder bias and delay;
- weigh assemblies and estimate their CoM/inertia; and
- verify foot contact friction on real surfaces.

`scripts/testbench_sim2real.py` and `scripts/validate_bam_testbench.py` support
the actuator side of this process.

## 12.9 Staged physical acceptance

The exact hardware procedure belongs to the Microduck runtime/hardware project,
but a safe conceptual progression is:

1. **Static interface test:** no torque; verify joint order, signs, HOME,
   sensors, command zeros, ONNX shapes, and loop timing.
2. **Supported low-authority test:** robot secured; conservative target/current
   limits; verify action direction and emergency stop.
3. **Supported policy test:** run short windows with zero or tiny commands;
   inspect saturation, delay, temperature, and observation ranges.
4. **Tethered behavior test:** flat surface, local operator, hard timeout,
   recorded telemetry/video.
5. **Bounded untethered test:** only after earlier acceptance criteria pass.
6. **Command and disturbance battery:** expand one measured region at a time.

Never make the cloud, Wi-Fi, or a high-level planner responsible for the fast
emergency path.

## 12.10 Hot-swapping policies

The runtime can switch walking, standing, recovery, and trick policies because
they share the 61D input and 14D output contract. Hot-swapping still requires:

- the same joint/action metadata;
- correct command-slot semantics for each skill;
- clearing stale command values when a skill expects zero padding;
- defined previous-action behavior at the boundary; and
- a safe arbitration state machine.

For example, sit/stand uses the first twist slot as a posture flag, while an
all-zero twist means stand. Feeding zeros because an application forgot the
flag can look like a policy that ignores the button.

Rehearse combinations before hardware:

```bash
uv run scripts/infer_policy.py \
    --walking walk.onnx \
    --standing stand.onnx \
    --sitstand sitstand.onnx \
    --roulade roulade.onnx \
    --new-cmd-obs
```

## 12.11 Deployment acceptance record

For each released policy, retain:

```text
source commit and task ID
resolved env/agent YAML
checkpoint digest
export command and ONNX digest
ONNX input/output and metadata inspection
fixed simulation evaluation results
CPU rehearsal results
runtime version and observation/action contract version
hardware calibration version
staged physical test logs and video
known operating envelope and known failures
rollback artifact
```

“The file loads” is a format check. “The viewer walks” is a simulation check.
Neither is physical release evidence.

## 12.12 Lab: prove checkpoint-to-ONNX parity

1. Select one checkpoint and record a fixed simulation command case.
2. Export it with `scripts/export.py`.
3. Inspect its 61→14 interface and metadata.
4. Run the same command through `scripts/infer_policy.py`.
5. Compare initial observations and actions for units, signs, and scale.
6. Explain any remaining trajectory difference from simulator/runtime details.

The goal is not bit-identical long trajectories—chaotic contacts diverge—but
an evidence-backed interface match.

Continue with
[Microduck customization labs](13_microduck_customization_labs.md).

## 12.13 Folded parity-lab solution

<details>
<summary>Show the expected evidence and comparison code</summary>

The result should include model/checkpoint hashes, exact observation corpus,
normalizer provenance, action transform, runtimes, tolerances, and per-element
error—not only “both looked similar.” For the same already-normalized or same
raw-and-embedded-normalizer input (choose one contract), compare arrays:

```python
import numpy as np

# Shape: (number_of_frozen_cases, 14). These must come from the same 61D
# observations and deterministic actor mode.
checkpoint_actions = np.load("checkpoint_actions.npy")
onnx_actions = np.load("onnx_actions.npy")

assert checkpoint_actions.shape == onnx_actions.shape
assert checkpoint_actions.shape[1] == 14
error = np.abs(checkpoint_actions - onnx_actions)
print("max_abs", error.max())
print("p99_abs", np.quantile(error, 0.99))

# Select tolerances from numeric precision/runtime evidence; do not copy these
# illustrative values blindly for a quantized model.
np.testing.assert_allclose(
    onnx_actions,
    checkpoint_actions,
    rtol=1e-5,
    atol=1e-6,
)
```

First compare one-step outputs; long contact trajectories can diverge from tiny
floating-point differences even when the interface is correct. If one-step
outputs disagree, inspect observation order, raw versus normalized input,
actor mean versus stochastic sample, dtype, HOME/action scaling, and stale
previous action before blaming physics.

</details>
