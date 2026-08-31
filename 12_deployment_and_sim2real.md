# 12. Export, Deployment, and Sim-to-Real

A training checkpoint is not the deployed policy. Deployment is a contract
between the exported graph, observation producer, command source, timing loop,
action mapping, actuator interface, and safety system.

This is where a statistically good policy becomes a real-time software
component. The key question changes from “does expected return improve?” to
“does this exact byte-level artifact receive the same physical meanings, on
time, within a bounded authority envelope?”

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

Track four distinct artifacts:

| Artifact | Contains | Main use |
| --- | --- | --- |
| training checkpoint | actor, critic, optimizer, normalizers, counters | resume and exact export |
| exported actor | deterministic inference graph and actor normalization | runtime inference |
| policy manifest | schema, joint order, units, rate, source/config hashes | integration and audit |
| release bundle | actor, manifest, tests, calibration compatibility, rollback | physical deployment |

Filenames are mutable labels. Compute a Secure Hash Algorithm 256-bit
(SHA-256) digest for every immutable artifact and place the digest in the
release record. If a runtime log says only `walk.onnx`, it cannot prove which
bytes moved the robot.

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

During PPO training, actions are sampled from a distribution. Deployment
usually exports the deterministic mean action:

```math
a_t^{deploy}=\mu_\theta(\tilde o_t),
```

not a fresh Gaussian sample. Evaluation must state whether it used the mean or
stochastic actor. Comparing a stochastic checkpoint rollout with deterministic
export adds sampling noise to an interface test.

The embedded normalizer means the exported function is conceptually

```math
f_{export}(o)=f_{actor}\left(
\frac{o-\mu_{obs}}{\sqrt{v_{obs}+\epsilon}}
\right).
```

Its public input is the **raw 61D contract**. Pre-normalizing in the runtime
applies the transformation twice. Conversely, exporting only $f_{actor}$ and
feeding raw measurements omits it. Both graphs accept the same tensor shape,
which is why numerical parity tests are mandatory.

After export, bind provenance:

```bash
sha256sum logs/rsl_rl/velocity/<run>/model_<iteration>.pt
sha256sum walk.onnx
git rev-parse HEAD
```

Store the export command, exporter commit, task ID, checkpoint digest, output
digest, and resolved configuration together. Metadata inside the graph is
helpful for portability; an external signed/read-only manifest remains useful
because metadata can itself be rewritten.

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

Build a frozen parity corpus rather than testing only zero:

```text
HOME and level gravity, zero command
small positive/negative perturbation of every joint feature
representative forward, lateral, turn, head, and idle commands
states near but inside documented observation limits
recorded simulation observations from success and failure trajectories
```

For each raw 61-vector, compare deterministic checkpoint output with exported
output before converting either to joint targets. Record maximum absolute,
root mean squared, and per-action error. Then separately test the target map:

```math
q^{target}=q^{HOME}+s\odot a,
```

where $\odot$ is elementwise multiplication. This separates graph error from
HOME, scale, sign, or permutation error.

A one-step parity tolerance should come from numeric precision and runtime
provider evidence. Long contact rollouts are sensitive: tiny accepted action
differences can produce different contact sequences. Therefore trajectory
divergence does not automatically falsify export parity, while one-step output
disagreement does.

If lower-precision or quantized inference is introduced, calibrate it on this
corpus and on closed-loop rollouts. An action error $\delta a_j$ becomes target
error $s_j\delta a_j$; the actuator gain can amplify it into torque. Smaller
files are not free if they move a high-gain joint across a contact boundary.

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

The contract should be machine-readable as well as documented. One source of
truth can generate:

- simulator observation-order tests;
- runtime constants/enumerations;
- graph metadata;
- a human-readable table; and
- recorded-log column names.

Duplicating a 61-entry list by hand in several languages invites silent drift.
A schema version changes whenever meaning, order, scale, frame, or timing
changes—even if dimension remains 61.

### Calibration is a transform, not a vague offset

Let the inertial sensor frame be $I$ and the policy body frame be $B$. If
$R_{BI}$ rotates a vector from sensor coordinates into body coordinates and
$b_I$ is gyro bias, then

```math
\omega_B=R_{BI}(\omega_I-b_I).
```

Applying the inverse rotation, subtracting bias after rotation with a bias in
the wrong frame, or using degrees per second instead of radians per second all
produce plausible finite numbers with the wrong meaning.

Encoder conversion likewise needs direction $d_j\in\{-1,+1\}$, scale
$k_j$, raw zero $z_j$, and policy HOME $q_j^{HOME}$:

```math
q^{rel}_j=d_j k_j(c_j-z_j)-q_j^{HOME}.
```

Test named poses and rotations. Holding a robot level should yield projected
gravity near the training convention; rotating it gently about one documented
axis should change the expected component and sign. Save calibration identity
and checksum with each hardware log.

Velocity estimation is also part of the policy. A finite difference

```math
\dot q_t\approx\frac{q_t-q_{t-1}}{t_t-t_{t-1}}
```

must use measured timestamps, not assume every packet arrived exactly on the
nominal period. Filtering reduces noise but adds phase delay; match the filter
or cover its effect during training.

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

### 12.6.1 End-to-end age, not only inference latency

Suppose a sensor sample is captured at $t_s$, observation assembly starts at
$t_o$, inference finishes at $t_i$, communication finishes at $t_c$, and the
actuator applies the target at $t_a$. The age of information at actuation is

```math
A=t_a-t_s.
```

Optimizing only $t_i-t_o$ can leave $A$ large because a sample waited in a
queue. Measure the whole path and its distribution: median is useful for
throughput; high percentile and maximum observed value matter for control.

For policy period $T_p=20$ ms, a simple execution budget is

```math
C_{sense}+C_{encode}+C_{infer}+C_{safety}+C_{send}+C_{margin}\le T_p.
```

Each $C$ should be a measured worst-case or high-confidence bound under
representative system load, not an idle-machine average. Worst-case execution
time (WCET) is the longest time a component can take under its stated
assumptions. General-purpose Linux rarely gives a formal WCET guarantee, so the
runtime should at least record percentiles, maxima, deadline misses, and the
load conditions used.

Schedule against absolute deadlines to avoid drift:

```python
period_ns = 20_000_000
next_release = monotonic_ns()
while enabled:
    next_release += period_ns
    sample = latest_sensor_snapshot()        # timestamp travels with sample
    command = latest_valid_command_or_zero()
    observation = encode_contract(sample, command, previous_action)
    action = policy(observation)
    target = safety_limit(home + scale * action)
    send_with_sequence_and_deadline(target, next_release)
    record_timing_and_contract_values()
    sleep_until_monotonic(next_release)
```

If the loop finishes late, sleeping “20 ms from now” makes phase drift. An
absolute schedule exposes lateness and tries to return to the intended phase.
The safety policy must define whether a late cycle holds the last safe target,
uses a fresh observation immediately, ramps toward HOME, or disables torque.

### 12.6.2 Queue semantics are control semantics

For state feedback, processing every old sample in first-in, first-out order
can create an ever-growing lag during overload. A bounded latest-value mailbox
usually fits a reactive policy better: overwrite stale unread samples and
count drops. For event logs, in contrast, losing intermediate records may be
unacceptable. Choose queue behavior per data meaning.

Commands need a validity envelope:

```text
source identity, sequence, source timestamp, receive timestamp,
valid-until time, requested mode, bounded values
```

Reject out-of-order or expired commands. A cloud planner may send low-rate
intent, but only the local safety authority decides whether it is fresh enough
to enter the 50 hertz loop.

Run the deterministic queue demonstration:

```bash
python examples/realtime_data_age.py
```

It compares first-in, first-out processing with a latest-value mailbox under a
temporary consumer slowdown and prints the resulting sample ages.

### 12.6.3 Communication has a schedulable bit budget

Whether the real transport is a serial servo protocol, Controller Area Network
(CAN), CAN with Flexible Data Rate (CAN-FD), or EtherCAT, estimate utilization
before hardware integration. If message class $i$ sends $b_i$ on-wire bits at
frequency $f_i$, ideal utilization is

```math
U=\frac{\sum_i b_i f_i}{R_{link}},
```

where $R_{link}$ is usable bit rate. Include framing, identifiers, checksums,
start/stop bits, inter-frame gaps, device turnaround, retries, synchronization,
diagnostics, and firmware-update traffic. Then measure; the ideal equation does
not capture arbitration and driver latency.

As an intentionally naive serial example, suppose each of 14 devices costs 20
response bytes plus 10 command/protocol bytes at 50 hertz, with 10 on-wire bits
per byte. The raw budget is

```math
14(20+10)(10)(50)=210{,}000\ \mathrm{bit/s}.
```

That is already 21% of a 1 megabit/s link before turnaround and retries. A
group read/write protocol can change the overhead substantially. The purpose of
the estimate is to reveal feasibility and headroom early, not to replace a bus
trace.

Clock alignment matters when measurements come from different devices. A
vector assembled from “latest” joint samples and an older inertial sample is
not truly simultaneous. Preserve source timestamps, measure clock offset/drift,
and either align/interpolate within justified limits or train against the
observed skew distribution.

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

The gap is easier to act on when classified:

| Gap | Example | Primary response |
| --- | --- | --- |
| structural | missing backlash/contact/thermal mode | improve model or controller architecture |
| parametric | mass, friction, gain, voltage differ | measure, identify, randomize uncertainty |
| observation | bias, frame, filtering, missing data | calibrate and match sensor pipeline |
| temporal/computational | delay, jitter, packet loss | real-time design and matched timing randomization |
| scenario | surfaces/pushes/commands absent in training | expand task distribution and evaluation |
| objective | simulated reward tolerates unsafe exploit | redesign reward/constraint and acceptance test |

Domain randomization usually optimizes an average:

```math
\max_\pi\ \mathbb{E}_{\theta\sim p(\theta)}[J(\pi;\theta)].
```

A safety-critical design may care about a lower tail or bounded set instead:

```math
\max_\pi\ \min_{\theta\in\Theta}J(\pi;\theta).
```

Worst-case robust optimization can be overly conservative and depends on a
credible set $\Theta$; expected randomization can hide rare failures. A
practical compromise trains on a measured distribution, evaluates explicit
boundary/adversarial cases, and enforces hard runtime limits outside learned
authority.

Alternatives and complements include:

- **system identification**: fit nominal parameters from input/output data;
- **online adaptation**: infer a latent environment/motor context from recent
  history and condition the policy on it;
- **residual modeling/control**: learn only the discrepancy around a trusted
  nominal model or controller;
- **real-data fine-tuning**: update with carefully collected physical data; and
- **teacher–student distillation**: train a privileged/adaptive teacher, then
  transfer behavior to a deployable observer.

None removes interface verification. A sophisticated adaptive policy with the
wrong joint permutation is still wrong.

### Validate from open loop to closed loop

Model comparison should progress through:

1. **static calibration**: zero/HOME, gravity, masses, geometry, frame signs;
2. **one-step response**: voltage/target to immediate current, velocity, torque;
3. **multi-step open-loop trajectory**: replay identical target sequence;
4. **closed-loop subsystem**: matched controller on a supported joint/fixture;
5. **whole-robot closed loop**: bounded policies and scenarios.

One-step error exposes local dynamics without policy feedback hiding it.
Long-horizon open-loop error reveals accumulated phase/model mismatch.
Closed-loop success is the final goal but can conceal compensating errors, so
retain lower-layer evidence.

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

For measured outputs $y_t$ and simulated predictions
$\hat y_t(\theta)$, a basic fit minimizes

```math
\theta^*=\underset{\theta}{\arg\min}\
\sum_{t\in\mathcal D_{fit}}
\left\lVert W(y_t-\hat y_t(\theta))\right\rVert_2^2,
```

where $W$ scales quantities with different units or importance. Validate on a
held-out excitation $\mathcal D_{test}$; fitting and evaluating on the same
step response can overfit sensor noise or one operating point.

Parameters must be identifiable from the experiment. Slow, small motion may
reveal static friction but not voltage saturation; one direction may not reveal
asymmetry; a fixed load cannot distinguish motor gain from inertia. Design
safe excitations that separately cover position, speed, acceleration, load,
direction, voltage, and temperature regions needed by the policy.

Fit residuals are evidence for randomization. If a parameter varies between
repeated trials, randomize that repeatability range. If one nominal model has a
systematic frequency-dependent residual, merely widening a scalar parameter
may not cover the missing dynamic mode; revise the model or observation/history
instead.

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

### 12.9.1 Authority must decrease with latency

A practical robot has several control planes:

```text
cloud/remote agent: goals, semantic plan, slow perception assistance
brain system-on-chip: local perception, planning, policy inference, logging
real-time control board: timestamped I/O, limits, watchdog, safe state
motor drive: current/voltage/position protection and commutation/servo loop
mechanical system: passive stops, compliance, stable power-off behavior
```

The system-on-chip (SoC) is the main compute module. The lower a layer and the
shorter its deadline, the less it may depend on upper layers. For JumpRover,
whose brain SoC is ahead of the unfinished real-time board and mechanics, this
is a design requirement for the eventual handoff—not evidence that those
unfinished layers already satisfy it.

Neural actions are proposals inside a safety envelope. Hard checks can include:

```math
q_{min}\le q^{target}\le q_{max},
\qquad
|q^{target}_t-q^{target}_{t-1}|\le \Delta q_{max},
```

plus current, velocity, temperature, tilt, communication age, power, and mode
limits. Clipping alone can create sustained saturation; log every intervention
and enter a safe state when interventions persist.

A watchdog asks not “is the process alive?” but “has a valid, fresh, ordered
command satisfying the current mode arrived before its deadline?” Exercise it
with fault injection before free motion:

- stop the inference process;
- delay or reorder command packets;
- freeze one sensor timestamp;
- corrupt one value beyond range;
- disconnect the planner/network; and
- request a policy/schema/calibration mismatch.

Each case needs an expected bounded transition and a recorded recovery/reset
procedure. An emergency stop must be reachable independently of the policy and
tested under the same power conditions as deployment.

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

Shape compatibility is necessary, not sufficient. At switch time, policy B
receives the physical state produced by policy A. If B was trained only from a
narrow reset distribution, that state may be out of distribution. Define a
guarded transition:

```text
request skill B
 -> verify policy/schema/calibration identity
 -> command A toward a documented handoff set
 -> verify pose, speed, contacts, command freshness, and safety state
 -> initialize B's previous-action/history rule
 -> activate B with bounded authority
 -> monitor acceptance window or fall back
```

Blindly blending actions,

```math
a=(1-\beta)a^A+\beta a^B,
```

can smooth targets but produces actions neither policy trained to own and does
not guarantee contact compatibility. Use blending only when its states and
transients are trained or explicitly validated. A discrete state machine with
safe handoff sets is easier to reason about initially.

Version command semantics per skill. Reusing twist index 0 as a posture flag is
valid only when the active-policy manifest declares that interpretation and
the arbiter clears unrelated slots atomically. Stale head/body commands should
not leak across skill changes.

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

Make release compatibility executable. The runtime should refuse to arm when
policy schema, joint map, robot hardware revision, calibration version, action
rate, or safety-envelope version is incompatible with its manifest. A warning
that scrolls past while torque enables is not enforcement.

Rollback means more than retaining an old file. Keep the previous known-good
bundle, its compatible runtime/configuration, and a tested atomic activation
mechanism. Record why rollback occurred and preserve the failed bundle/logs for
diagnosis. Never overwrite the only known-good artifact during an experiment.

Telemetry needed for postmortem should be bounded and prioritized. At minimum,
retain monotonic/source timestamps, mode/policy hash, raw 61D observation,
action and applied target, command identity/age, safety interventions, actuator
status, deadline metrics, and faults. High-rate storage may use a ring buffer
that freezes recent history on a trigger so logging cannot exhaust memory.

## 12.12 Exercises and deployment lab

1. Name the information present in a training checkpoint but normally absent
   from the deployed actor. Why is the checkpoint still needed after export?
2. Derive what happens when a runtime pre-normalizes an observation and feeds
   it to a graph that already embeds the same normalizer.
3. Design a frozen 61D parity corpus with at least five semantically different
   cases. Why is all-zero alone insufficient?
4. If action scale for joint $j$ is 1 radian/unit and export error is 0.003,
   what target-position error results in radians and degrees?
5. A sensor captures at 100.000 ms, observation assembly starts at 104.000 ms,
   inference ends at 108.500 ms, send finishes at 111.000 ms, and the actuator
   applies at 113.000 ms. Compute inference latency and age of information.
6. In a 20 ms period, sensing costs 2.5 ms, encoding 1.0 ms, inference 4.5 ms,
   safety 0.5 ms, and sending 3.0 ms. Compute nominal slack. Why is this not a
   WCET proof?
7. Run `python examples/realtime_data_age.py`. Explain why the first-in,
   first-out queue becomes stale even without packet loss.
8. Repeat the chapter's serial-link calculation at 100 hertz. What utilization
   does it imply on 1 megabit/s before unmodeled overhead?
9. Write the correct body-frame angular-velocity transform from an inertial
   sensor measurement, mounting rotation, and sensor-frame bias. Name two
   plausible but wrong implementations.
10. Two joint samples are 0.020 radian apart and their measured timestamps are
    15 ms apart, although nominal period is 20 ms. Calculate velocity using
    measured and assumed time. What error does the assumption cause?
11. Classify each as structural, parametric, observation, temporal, scenario,
    or objective gap: missing backlash; wrong gyro sign; low battery voltage;
    rare 35 ms inference; unseen wet tile; reward permits head impacts.
12. Why must actuator identification use held-out excitations? Give one example
    of a parameter that a slow low-load trajectory cannot identify well.
13. Specify the expected safe response to three injected faults: expired cloud
    command, frozen sensor timestamp, and repeated joint-target saturation.
14. Design a guarded handoff from walking to stand-up recovery. Include the
    previous-action rule and fallback.
15. Explain why a cloud agent may choose a goal but must not own the emergency
    stop or 50 hertz deadline.
16. Prove checkpoint-to-ONNX parity: export one exact checkpoint, inspect the
    interface/metadata, compare deterministic one-step actions over a frozen
    corpus, rehearse deployment-style commands, and write a signed/hash-based
    evidence record.

## 12.13 Folded solutions

<details>
<summary>Show solutions to Exercises 1–8</summary>

1. The checkpoint includes critic, optimizer state, observation-normalizer
   state, counters/curriculum progress, and usually stochastic-policy training
   state in addition to actor weights. It remains the reproducible source for
   re-export, parity, audit, and training resume; the smaller actor cannot
   recreate all of that state.
2. If $N(o)=(o-\mu)/\sigma$, double normalization computes
   $N(N(o))=((o-\mu)/\sigma-\mu)/\sigma$, not $N(o)$. Only special accidental
   values of $\mu,\sigma$ hide the error. Shape and finiteness still pass.
3. Include level HOME/zero command, positive and negative one-feature probes,
   forward/lateral/turn/head commands, near-limit valid observations, and
   recorded success/failure states. Zero does not exercise joint ordering,
   command routing, realistic normalization ranges, or nonlinear activation
   regions.
4. The target error is $1(0.003)=0.003$ radian. In degrees it is
   $0.003(180/\pi)\approx0.172^\circ$. Whether that is acceptable depends on
   the joint, gains, contact state, and closed-loop evidence.
5. Inference latency is $108.5-104=4.5$ ms. Information age at application is
   $113-100=13$ ms. Reporting only 4.5 ms omits sensing wait, communication,
   and actuation timing.
6. Used time is $2.5+1+4.5+0.5+3=11.5$ ms, so nominal slack is 8.5 ms. These
   are single/average measurements unless bounded under representative worst
   load; scheduling, allocation, cache, transport retries, and contention can
   consume the apparent slack.
7. The producer creates four 200 hertz samples per 50 hertz controller period,
   but FIFO consumes only one. Its unread backlog grows, so selected capture
   times move farther behind wall time. Latest-value semantics intentionally
   drop obsolete state and retain bounded age; drop counts remain diagnostic.
8. Doubling frequency doubles the raw budget to 420,000 bit/s, or 42% of a
   1 megabit/s link. Protocol turnaround, retries, diagnostics, and scheduling
   still require measured headroom.

</details>

<details>
<summary>Show solutions to Exercises 9–15</summary>

9. Use $\omega_B=R_{BI}(\omega_I-b_I)$. Wrong variants include using
   $R_{IB}$ without transposing/inverting it, subtracting a body-frame bias as
   though it were sensor-frame, applying degrees as radians, or reversing an
   axis sign.
10. Measured-time velocity is $0.020/0.015=1.333$ rad/s. Assumed-time velocity
    is $0.020/0.020=1.0$ rad/s, underestimating by 0.333 rad/s or 25% relative
    to the measured-time value.
11. Missing backlash is structural; wrong gyro sign is observation; low
    battery voltage is parametric/operating-condition; rare 35 ms inference is
    temporal; unseen wet tile is scenario (and friction parameter once
    modeled); a reward permitting head impacts is objective. Boundaries can
    overlap, so name the causal layer the proposed fix addresses.
12. Fitting and testing the same excitation measures memorization of one
    operating trajectory, not predictive model quality. Slow low-load motion
    poorly identifies voltage/current saturation, high-speed back electromotive
    force, inertia under acceleration, or thermal effects.
13. An expired cloud command should be rejected and replaced with trained zero
    or a local safe mode. A frozen sensor timestamp should invalidate the
    observation and hold/ramp/disable according to the tested state machine,
    not reuse it silently. Repeated saturation should raise telemetry and leave
    neural authority—ramp to a bounded pose or disable as hardware permits.
14. Verify the recovery policy bundle/schema/calibration, detect a valid fallen
    state within its trained reset support, zero unrelated commands, initialize
    previous action according to its manifest (often last applied action or a
    documented zero), switch with bounded current/target rate, monitor progress
    and impacts, and fall back to torque-off/supported intervention on timeout
    or contract violation. Do not switch while the walking policy is still
    commanding motion without an explicit transition.
15. Cloud/network latency, outages, reordering, and security boundaries are
    incompatible with the fast physical deadline. The remote agent may provide
    expiring semantic intent; local real-time layers validate it and retain
    independent watchdog, limits, stop, and safe-state authority.

</details>

<details>
<summary>Show the expected parity evidence and comparison code for Exercise 16</summary>

The result should include model/checkpoint hashes, exact raw observation
corpus, normalizer provenance, action transform, runtime providers, tolerances,
and per-element error—not only “both looked similar.” Compare deterministic
actors on identical 61D raw inputs:

```python
import numpy as np

# Shape: (number_of_frozen_cases, 14). Both arrays must come from the same
# raw 61D observations; the checkpoint path must apply its normalizer once and
# the exported graph must contain/apply that same normalizer once.
checkpoint_actions = np.load("checkpoint_actions.npy")
onnx_actions = np.load("onnx_actions.npy")

assert checkpoint_actions.shape == onnx_actions.shape
assert checkpoint_actions.shape[1] == 14
error = np.abs(checkpoint_actions - onnx_actions)
print("max_abs", error.max())
print("p99_abs", np.quantile(error, 0.99))

# Derive tolerances from numeric precision/runtime evidence; do not copy these
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
actor mean versus stochastic sample, data type, HOME/action scaling, and stale
previous action before blaming physics. The final record binds checkpoint,
export, corpus, result, and rehearsal log by hashes and includes the known
operating envelope and rollback bundle.

</details>

Continue with
[Microduck customization labs](13_microduck_customization_labs.md).
