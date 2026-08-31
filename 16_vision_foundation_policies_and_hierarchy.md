# 16. Vision, Foundation Policies, and Hierarchical Autonomy

A robot that can balance is not automatically a robot that knows where to go.
This chapter connects pixels to geometry, geometry to safe local motion, and
language goals to typed robot skills. It also explains what modern
vision-language-action (VLA) policies learn, what they do not learn, and where
reinforcement learning (RL) still belongs.

By the end, you should be able to:

- derive a 3D point from a calibrated depth pixel;
- explain why timestamps and uncertainty are part of a perception output;
- compare modular, end-to-end, and hybrid visual-control architectures;
- map action regression, tokens, chunks, diffusion, and flow matching to code;
- read current generalist-policy papers without treating model size as proof;
- derive the option-value equation used in hierarchical RL; and
- specify a safe onboard/cloud architecture for JumpRover.

The central rule is simple: **semantics may run slowly, but physical stability
cannot wait**.

## 16.1 Perception turns state into belief

**Proprioception** measures the robot itself: joint encoders, an inertial
measurement unit (IMU), motor state, and battery voltage. **Exteroception**
measures the surrounding world: a red-green-blue (RGB) camera, depth camera,
light detection and ranging (LiDAR), radar, microphone, or tactile array.

An image is not a state vector. The same image can be compatible with many
worlds: a small nearby obstacle may occupy the same pixels as a large distant
one; a hidden person may still be moving; and camera motion may resemble object
motion. A useful formal model is therefore a **Partially Observable Markov
Decision Process (POMDP)**. The hidden state is $s_t$, the observation is
$o_t$, and the robot acts from a belief $b_t(s)$, a probability distribution
over possible states.

A Bayes-filter update has two steps. First predict through the dynamics:

```math
\bar b_t(s_t)=\int p(s_t\mid s_{t-1},a_{t-1})
                     b_{t-1}(s_{t-1})\,ds_{t-1}.
```

Then incorporate the new sensor measurement:

```math
b_t(s_t)=\eta\,p(o_t\mid s_t)\bar b_t(s_t),
```

where $\eta$ normalizes the distribution to integrate to one. In plain
language, prediction says where the world could have moved; measurement says
which predictions agree with the sensor. A learned recurrent policy may encode
this belief implicitly in hidden activations. A geometric system may represent
it explicitly as pose covariance, object tracks, and an occupancy map.

This distinction matters for Microduck. Its present 61-dimensional actor
observation contains proprioception and commands, not pixels or range. A box
rendered in the viewer changes neither $o_t$ nor the reward unless the
environment explicitly connects it. The policy may collide because, from its
information, the box does not exist.

## 16.2 From a pixel to a point in the world

### The pinhole camera model

A camera maps a 3D camera-frame point $(X,Y,Z)$ to pixel coordinates
$(u,v)$. For the ideal pinhole model,

```math
\begin{aligned}
u &= f_x X/Z+c_x,\\
v &= f_y Y/Z+c_y.
\end{aligned}
```

Here $f_x,f_y$ are focal lengths measured in pixels and $(c_x,c_y)$ is the
principal point. Collect them in the intrinsic matrix

```math
K=\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}.
```

If a depth sensor supplies $Z$, back-projection recovers the camera-frame
point:

```math
\begin{bmatrix}X\\Y\\Z\end{bmatrix}
=ZK^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix}.
```

Suppose $f_x=f_y=400$, $(c_x,c_y)=(320,240)$, and pixel $(360,220)$
has depth $Z=1.2$ m. Then

```math
X=1.2(360-320)/400=0.12\ \mathrm{m},
```

```math
Y=1.2(220-240)/400=-0.06\ \mathrm{m}.
```

The point is 12 cm to one camera-side axis, 6 cm to the other, and 1.2 m
forward. Axis signs depend on the declared frame convention; never guess them.

### Intrinsics, distortion, and extrinsics

**Intrinsic calibration** estimates focal lengths, principal point, and lens
distortion. Real lenses bend rays, commonly with radial and tangential terms;
undistortion must use the calibration for the actual resolution and crop.

**Extrinsic calibration** estimates the rigid transform between sensor and
robot frames. With homogeneous coordinates,

```math
{}^B p = {}^B T_C\,{}^C p,
```

where ${}^B T_C$ maps a point from camera frame $C$ into robot-base frame
$B$. The superscripts answer “expressed in which frame?” A transform in the
wrong direction can look numerically plausible while placing every obstacle
behind the robot.

Calibration is not a one-time ceremonial file. A loose bracket, changed focus,
new image crop, or mechanical impact changes the mapping. Store calibration
revision, image mode, temperature assumptions, and validation residual with
the deployment artifact.

### Error propagation

Depth and calibration errors grow into position error. For
$X=Z(u-c_x)/f_x$, a first-order variance approximation is

```math
\sigma_X^2 \approx
\left(\frac{u-c_x}{f_x}\right)^2\sigma_Z^2
+\left(\frac{Z}{f_x}\right)^2\sigma_u^2
+\left(\frac{Z(u-c_x)}{f_x^2}\right)^2\sigma_{f_x}^2.
```

This is a Jacobian covariance rule: sensitivity squared times input variance.
It explains why distant depth, edge pixels, and uncertain focal length should
not yield the same clearance margin as a clean central measurement.

## 16.3 Time is part of every measurement

A perception message needs at least:

```text
value + frame + capture timestamp + covariance/confidence + validity
```

The timestamp should describe photon capture, not when a neural network
finished. Define action-time age as

```math
A_t=t_{\mathrm{act}}-t_{\mathrm{capture}}.
```

At speed $v$, age alone creates approximately $vA_t$ of unobserved travel.
A rover moving at $0.5$ m/s with a 220 ms-old obstacle map has moved 11 cm
since the represented scene. That is substantial on a small robot.

Important sources of age and misalignment include:

- exposure and sensor readout;
- rolling-shutter rows captured at different instants;
- camera/IMU clocks with offset or drift;
- frame queues that process old images in order;
- preprocessing, inference, transport, and planning; and
- an action chunk continuing after the observation that produced it is stale.

A latest-value queue is often safer than an unbounded first-in-first-out queue:
drop an old frame when a newer one is available. Log capture time, inference
start/end, planner time, and action-application time separately. Average frames
per second (FPS) can look excellent while the tail age is unsafe.

A minimal message and age check might be:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LocalMap:
    capture_ns: int
    frame: str
    min_clearance_m: float
    clearance_std_m: float
    valid: bool

def usable(msg: LocalMap, now_ns: int) -> bool:
    age_ms = (now_ns - msg.capture_ns) / 1e6
    return msg.valid and msg.frame == "base_link" and age_ms <= 150.0
```

The number 150 ms is an example, not a universal safe threshold. Derive the
real bound from speed, braking, compute tails, and mechanical tests.

## 16.4 Three visual-control architectures

### Modular perception and planning

A modular stack may use **simultaneous localization and mapping (SLAM)** to
estimate robot pose while constructing a map:

```text
camera/depth -> detection + tracking + local map
             -> collision-aware planner -> bounded velocity command
             -> locomotion policy -> realtime controller
```

Intermediate contracts are inspectable. A developer can replay a camera log,
measure map error without moving motors, then test the planner against a known
map. Geometric collision constraints and deterministic fallbacks fit naturally.
The cost is interface engineering: frame, units, latency, and confidence errors
can propagate across otherwise-correct modules.

### End-to-end visual policy

```text
images + proprioception + language/goal -> neural policy -> robot actions
```

The representation is optimized for task success and can use visual cues a
hand-designed map discards. But data requirements, causal diagnosis, sensor
shift, and safety validation become harder. “End-to-end” never literally
removes firmware, motor limits, synchronization, or emergency-stop logic.

### Hybrid architecture

```text
learned visual encoder -> terrain/object latent or metric local map
proprioception + goal  -> learned local policy
                       -> geometric governor -> realtime controller
```

Most deployable systems are hybrids. The engineering question is not which
label is fashionable, but **where uncertainty becomes explicit and where
authority is bounded**.

A useful contract table is:

| Boundary | Required fields | Rejection example |
| --- | --- | --- |
| perception to planner | frame, time, covariance, valid region | stale map |
| planner to skill | goal, bounds, expiry, stop rule | unreachable goal |
| skill to controller | schema, units, rate, sequence | wrong joint order |
| cloud to onboard | exact skill and typed parameters | unknown skill |

## 16.5 Obstacle avoidance is a causal chain

Putting obstacles into a simulator teaches nothing by itself. Successful
avoidance requires the whole chain:

```text
vary obstacles
    -> sense before contact
    -> encode usable geometry/history
    -> make safe progress preferable to collision
    -> expose recovery decisions
    -> evaluate held-out layouts and failures
```

### A stopping-distance derivation

Let forward speed be $v$, end-to-end reaction delay be $\tau$, and verified
braking magnitude be $a_b>0$. During delay the robot travels $v\tau$. Under
constant braking, $0=v^2-2a_bd$, so braking distance is $v^2/(2a_b)$.
Add a geometric and estimation margin $m$:

```math
d_{\mathrm{stop}}=v\tau+\frac{v^2}{2a_b}+m.
```

For $v=0.4$ m/s, $\tau=0.18$ s, $a_b=0.8$ m/s², and $m=0.12$ m:

```math
d_{\mathrm{stop}}=0.4(0.18)+\frac{0.4^2}{2(0.8)}+0.12
=0.292\ \mathrm{m}.
```

This is already 29.2 cm. If the robot cannot reliably detect, command, brake,
and settle within that space, 0.4 m/s is outside the validated envelope.

If clearance estimate $D$ is approximately Gaussian with mean $\mu_D$ and
standard deviation $\sigma_D$, a one-sided conservative clearance can be

```math
D_{\mathrm{safe}}=\mu_D-k\sigma_D.
```

Choosing $k=2.33$ corresponds to roughly a 1% lower-tail probability under a
well-calibrated Gaussian assumption. Real perception errors are often
non-Gaussian and correlated, so measure empirical coverage rather than trusting
that interpretation automatically.

Solve $D_{\mathrm{safe}}\ge d_{\mathrm{stop}}$ for the maximum permitted
speed. Since the equation is quadratic,

```math
v_{\max}=-a_b\tau+
\sqrt{(a_b\tau)^2+2a_b(D_{\mathrm{safe}}-m)}.
```

This converts uncertainty into authority: greater age or variance lowers the
speed command before collision.

The runnable
[camera and stopping example](examples/camera_latency_and_stopping.py) computes
the projection, age distance, and speed envelope without third-party packages.

### Occupancy is not certainty

An occupancy grid stores a probability per cell, often in log-odds form:

```math
\ell_t=\log\frac{p_t}{1-p_t}.
```

Independent measurement updates become additions in log-odds, with a prior
correction. The independence assumption is imperfect when adjacent depth pixels
share the same failure. Do not interpret 100 correlated pixels as 100
independent confirmations.

For Microduck, the lowest-risk first project is to preserve its locomotion
actor:

```text
depth -> local occupancy/height map -> local planner
      -> [forward speed, lateral speed, yaw rate]
      -> existing 61D-input velocity policy
```

A second research project can train perceptive locomotion from depth/history
and proprioception directly to 14 joint targets. That may coordinate footsteps
with terrain, but it creates a new observation contract, dataset, runtime, and
sim-to-real validation problem. It is not a drop-in upgrade.

## 16.6 Learning a visual representation

A visual encoder maps image $I_t$ to a compact latent $z_t=f_\psi(I_t)$.
What makes $z_t$ useful depends on its training objective.

### Four common objectives

1. **Supervised prediction.** Predict depth, segmentation, object labels, or
   traversability. Labels make semantics explicit but are expensive and can
   omit unlabelled control cues.
2. **Reconstruction.** Encode enough information to reconstruct masked pixels
   or video. This uses unlabeled data, but pixel detail is not automatically
   relevant to control.
3. **Contrastive learning.** Make matching views/text close and nonmatching
   samples far apart.
4. **Task loss.** Fine-tune the encoder through behavior cloning or return. It
   focuses features on the current task but may forget reusable information.

A common contrastive loss for an image anchor $i$, positive text or view
$i^+$, negatives $j$, similarity $s$, and temperature $T$ is

```math
L_i=-\log
\frac{\exp(s(z_i,z_{i^+})/T)}
{\sum_j\exp(s(z_i,z_j)/T)}.
```

The loss teaches relative association, not metric depth or collision safety.
A representation can retrieve “chair” perfectly while confusing whether its
legs block the planned path.

### Probes test information; interventions test use

A **linear probe** freezes the encoder and trains a small linear head to predict
a property such as depth. High probe accuracy says information is decodable; it
does not prove the policy uses it. Stronger tests intervene:

- mask the obstacle region and measure action change;
- keep geometry fixed while changing texture;
- move the camera while keeping the robot state fixed;
- inject timestamp delay independently of image corruption; and
- compare success on held-out objects, rooms, and lighting.

Separate representation failure (“the latent lacks clearance”) from policy
failure (“clearance is encoded but ignored”) and calibration failure (“the
camera-to-base transform is wrong”).

## 16.7 Anatomy of a vision-language-action policy

A VLA policy consumes visual observations, a language instruction, and usually
robot state, then predicts one action or an action sequence:

```math
a_{t:t+H-1}\sim
\pi_\theta(I_{t-k:t},\,q_t,\,\text{instruction},\,e).
```

Here $q_t$ is proprioception, $H$ is action-chunk length, and $e$ encodes
the embodiment or robot-specific schema. Most VLA pretraining is supervised
imitation on demonstrations, not online RL. A neural policy is not RL merely
because its output moves a robot.

### Tokenizing observations

An image encoder turns patches or learned regions into visual tokens. A text
tokenizer converts the instruction to language tokens. Proprioception may be
projected by a multilayer perceptron (MLP) into the same hidden width. A
transformer mixes these tokens with attention. For queries $Q$, keys $K$,
and values $V$, one attention head computes

```math
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
```

This is content-dependent weighted averaging. It does not by itself impose a
coordinate frame, causality, or physical feasibility; training data and masks
must teach those relations.

### Five action parameterizations

**1. Direct regression.** Predict a continuous action and minimize, for
example, absolute error:

```math
L_{\mathrm{L1}}=\frac{1}{Hd_a}
\sum_{h=0}^{H-1}\sum_{j=1}^{d_a}
|a_{t+h,j}-\hat a_{t+h,j}|.
```

It is simple and fast. With multiple equally valid demonstrations, a single
mean prediction can average incompatible behaviors.

**2. Per-dimension action tokens.** Normalize each action component and assign
it to one of $B$ bins. An autoregressive language-model-style head predicts
categorical symbols:

```math
L_{\mathrm{token}}=-\sum_{h,j}
\log p_\theta(k_{h,j}\mid I,q,\text{instruction},k_{<h,j}).
```

Tokens reuse language-model machinery, but naïve quantization introduces bin
error and creates long output sequences at high control rates.

**3. Action chunks.** Predict $H$ future actions at once. If policy inference
is 10 Hz and hardware control is 50 Hz, a five-step chunk can fill the interval.
Chunks represent temporal coordination and reduce calls, but a long open-loop
chunk reacts slowly to disturbance. Receding-horizon execution predicts $H$
steps and executes only the first $E<H$ before replanning.

**4. Diffusion or flow action generation.** Start with noise and iteratively
transform it into a continuous action chunk. This can represent multiple modes
without averaging them. It costs multiple numerical steps and introduces
sampling variation.

**5. Compressed trajectory tokens.** Frequency-space Action Sequence
Tokenization (FAST) transforms action chunks into a frequency representation,
quantizes coefficients, and compresses them. Smooth robot trajectories often
concentrate energy in low frequencies, so a chunk can require far fewer tokens
than independent timestep bins. The
[FAST paper](https://arxiv.org/abs/2501.09747) reports that this improves
high-frequency dexterous-action modeling in its evaluated settings.

### Flow matching from equation to pseudocode

Let $x_1$ be a demonstrated action chunk and $x_0$ Gaussian noise. A simple
linear probability path at interpolation time $\tau\in[0,1]$ is

```math
x_\tau=(1-\tau)x_0+\tau x_1.
```

Its target velocity is $u=x_1-x_0$. Train a conditional vector field
$v_\theta$ from vision-language context $c$:

```math
L_{\mathrm{flow}}=
\mathbb E_{x_0,x_1,\tau}
\left[\|v_\theta(x_\tau,\tau,c)-(x_1-x_0)\|_2^2\right].
```

At inference, sample $x(0)\sim\mathcal N(0,I)$ and solve

```math
\frac{dx}{d\tau}=v_\theta(x,\tau,c)
```

numerically to $\tau=1$. An Euler implementation is conceptually:

```python
x = gaussian_noise(action_chunk_shape)
for step in range(num_flow_steps):
    tau = step / num_flow_steps
    velocity = model(x, tau, images, instruction, robot_state)
    x = x + velocity / num_flow_steps
action_chunk = denormalize(x)
```

Real implementations choose a path, sampler, masks, padding, and numerical
schedule carefully. The pseudocode maps the mathematics to computation; it is
not a production controller.

### Normalization and embodiment adapters

Robot datasets disagree about action coordinates and scale. A value of 0.1 can
mean 0.1 radians, 0.1 m/s, a normalized gripper command, or an end-effector
delta. A robust adapter declares:

- joint, task-space, or base-velocity action meaning;
- absolute versus delta coordinates;
- reference frame and axis order;
- units, control rate, and chunk horizon;
- bounds and normalization statistics;
- absent dimensions and masks; and
- conversion to the local safety-controlled command.

Cross-embodiment data is useful only after this schema work. File-format
compatibility is not semantic compatibility.

## 16.8 A research map from 2022 through 2026

This area changes quickly. The table is a map of design ideas, not a
leaderboard. Results across papers use different robots, data, tasks, and
success definitions. Chapter 18 contains detailed paper seminars.

The names below include Robotics Transformer 1 (RT-1), Robotics Transformer 2
(RT-2), and Generalist Robot 00 Technology (GR00T). They are project names, not
certifications of general capability.

| Work | Main design lesson | Evidence boundary |
| --- | --- | --- |
| [RT-1](https://arxiv.org/abs/2212.06817) | scale a token policy over diverse real tasks | one fleet/task family |
| [RT-2](https://arxiv.org/abs/2307.15818) | co-train web semantics and robot action tokens | closed models/setup |
| [Open X-Embodiment](https://arxiv.org/abs/2310.08864) | normalize and mix many robot datasets | manipulation-heavy mix |
| [Octo](https://arxiv.org/abs/2405.12213) | open generalist transformer for adaptation | fine-tuning still needed |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | open 7B visual-language backbone to action tokens | large runtime footprint |
| [π₀](https://arxiv.org/abs/2410.24164) | flow-matched continuous action chunks | adapter/data overlap matter |
| [GR00T N1](https://arxiv.org/abs/2503.14734) | reasoning backbone plus continuous action expert | mainly manipulation evidence |
| [π₀.₅](https://arxiv.org/abs/2504.16054) | heterogeneous co-training for open-world tasks | reported platform/protocol |
| [SmolVLA](https://arxiv.org/abs/2506.01844) | smaller open model and asynchronous inference | staleness must be measured |
| [FAST](https://arxiv.org/abs/2501.09747) | compress smooth chunks into action tokens | tokenizer is one component |
| [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | continuous chunks and parallel optimized fine-tuning | benchmark-specific comparison |

**Robotics Transformer 1 (RT-1)** and **Robotics Transformer 2 (RT-2)** made
large-scale language-conditioned robot action tokenization influential. Open
X-Embodiment and Robotics Transformer X (RT-X) then made cross-robot data
normalization a central research problem.

[Octo's official code](https://github.com/octo-models/octo),
[OpenVLA's official code](https://github.com/openvla/openvla), and
[Physical Intelligence's `openpi`](https://github.com/Physical-Intelligence/openpi)
make different parts of the stack inspectable. OpenVLA also studies Low-Rank
Adaptation (LoRA), which updates a low-rank correction
$W'=W+BA$ instead of every element of $W$. It reduces trainable parameters;
it does not guarantee low inference latency.

Generalist Robot 00 Technology (GR00T) N1 separates a semantic reasoning
system from a continuous action system. SmolVLA, distributed through the
[LeRobot project](https://github.com/huggingface/lerobot), emphasizes a smaller
open model and asynchronous inference. Both reinforce the same systems lesson:
model architecture does not remove action schemas, observation age, or local
safety.

OpenVLA-OFT reports large gains in its LIBERO and real-robot protocols by
combining continuous actions, chunks, parallel decoding, and an absolute-error
loss. This is more educational than quoting its final success number: several
apparently small action-head decisions can dominate adaptation quality and
throughput.

As of this book's September 2026 review, 2026 papers are still recent
preprints. For example,
[Green-VLA](https://arxiv.org/abs/2602.00919) proposes staged visual-language,
multi-embodiment, embodiment-specific, and RL-alignment phases plus
out-of-distribution detection. Treat it as a current research signal whose
claims require reproduction, not as settled superiority. Publication date and
model name are never substitutes for matched trials.

### How to choose a starting point

Ask in this order:

1. Does the released action schema match the robot and task?
2. Does the pretraining data contain related viewpoints, objects, and motions?
3. Are weights, adapters, normalization, and evaluator all available?
4. Can onboard hardware meet memory and tail-latency requirements?
5. Is there enough target data to fine-tune and validate held-out conditions?
6. Does the license permit the intended use and redistribution?

A small behavior cloning (BC) policy with correct data can beat an enormous
foundation policy with the wrong embodiment. Run that baseline.

## 16.9 Hierarchical reinforcement learning and options

In hierarchical RL, a higher policy chooses a goal, latent, or temporally
extended skill; a lower policy executes it. An **option** consists of:

- an initiation set $\mathcal I_o$: states where it may start;
- an internal policy $\pi_o(a\mid s)$; and
- a termination probability $\beta_o(s)$.

Because an option can last $k$ primitive steps, the high level forms a
semi-Markov decision process. If rewards during the option are
$r_t,\ldots,r_{t+k-1}$, its option return is

```math
R_o=\sum_{i=0}^{k-1}\gamma^i r_{t+i}.
```

The option-value Bellman equation is

```math
Q(s,o)=\mathbb E\left[
R_o+\gamma^k\max_{o'}Q(s',o')
\mid s,o\right].
```

The factor is $\gamma^k$, not merely $\gamma$, because physical time passed
inside the option. Long skills accumulate more reward and more risk before the
high level can reconsider.

A robot option should be a typed contract:

```yaml
option: dock
preconditions:
  dock_track_age_ms_max: 150
  localization_std_m_max: 0.08
parameters:
  dock_id: dock.main
  approach_speed_mps: 0.08
success: charging_contact_confirmed
abort: [track_lost, bumper_hit, timeout, tilt_limit]
timeout_s: 30
fallback: stop_balanced
```

The initiation set becomes preconditions, termination becomes success/abort,
and the internal policy becomes a versioned skill. This connects RL theory to
a practical application programming interface (API).

## 16.10 Language planning needs physical affordance

A language model (LM) can score whether a skill sounds useful. It cannot infer
current physical feasibility from words alone. SayCan formalizes the
combination for candidate skill $o$:

```math
\mathrm{score}(o)\propto
p_{\mathrm{LM}}(o\mid \text{instruction},\text{history})
\;p_{\mathrm{afford}}(\mathrm{success}\mid b,o).
```

The first factor asks “does this step help the request?” The second asks “can
the robot do it from the current belief?” If the requested mug is meaningful
but unreachable, semantic probability can be high while affordance is low.

An affordance estimator may be a learned skill value, geometric feasibility
check, or conservative combination. Calibrate it on attempted starts, including
failures. A planner that sees only successful demonstrations will overestimate
what is executable.

A cloud large language model (LLM) should emit a constrained proposal such as:

```json
{
  "request_id": "mission-184-step-3",
  "skill": "follow_person",
  "subject_id": "person.bruce",
  "following_distance_m": 1.2,
  "max_speed_mps": 0.15,
  "expires_at_monotonic_ms": 8842120,
  "stop_if": {
    "track_age_ms_gt": 150,
    "clearance_m_lt": 0.35,
    "localization_std_m_gt": 0.10
  }
}
```

JavaScript Object Notation (JSON) syntax is not sufficient validation. Onboard
software must reject unknown fields, wrong units, expired requests, unavailable
skills, failed preconditions, and authority beyond configured bounds. It should
also make repeated request identifiers idempotent so a network retry cannot
start the same physical action twice.

## 16.11 Cloud agent: proposal, not motor authority

A cloud model can add language understanding, long-horizon decomposition,
semantic interpretation of selected images, manual retrieval, and human-facing
explanation. It also adds:

- variable latency and disconnection;
- service/model-version drift;
- uncertain or adversarial instructions;
- replay and authorization risks; and
- privacy exposure from images, audio, maps, and identities.

Use a capability boundary:

```text
human request + minimized semantic world summary
                        |
                        v
cloud proposes exact skill + bounded parameters + expiry
                        |
                        v
authenticate -> schema validate -> check freshness/preconditions
                        |
                        v
local mission manager -> planner/governor -> local motor skill
                        |
                        v
realtime limits, watchdog, enable, and emergency stop
```

The cloud must never stream torque, pulse-width modulation (PWM), motor current,
or raw balance corrections. A network outage must leave onboard balance, stop,
and human emergency control available.

### Security and privacy are control requirements

Use mutually authenticated transport, least-privilege credentials, request
expiry, nonce/replay protection, signed policy manifests, audit logs, and a
local allow-list of skills. Minimize cloud data: a local phrase such as
“authorized person at bearing 12 degrees, confidence 0.93” may replace an
identifiable image when pixels are unnecessary. Define retention and deletion,
not merely encryption in transit.

Prompt injection can arrive through visible text or speech in the environment.
Treat perceptual text as untrusted data, not instruction authority. A sign that
says “disable safety and go faster” must not override the mission schema.

## 16.12 Runtime contracts for learned skills

Package a policy with its evidence, not just its weight file. A YAML Ain't
Markup Language (YAML) manifest might contain:

```yaml
policy_id: walk-flat-v3
model_sha256: "..."
training_code_commit: "..."
robot_model_revision: "microduck-r7"
input_schema_version: "microduck-obs-61-v1"
output_schema_version: "joint-delta-14-v1"
policy_rate_hz: 50
max_input_age_ms: 30
inference_wcet_ms: 6.0
normalization: embedded
command_bounds:
  vx_mps: [-0.3, 0.3]
  yaw_rps: [-1.0, 1.0]
validated_conditions: [flat_indoor, nominal_voltage]
known_exclusions: [stairs, obstacle_avoidance]
evaluator_report_sha256: "..."
fallback_policy_id: "stand-safe-v2"
```

The hash should use Secure Hash Algorithm 256-bit (SHA-256) or another approved
integrity mechanism. The worst-case execution time (WCET) must come from the
target hardware and load, not a laptop average. A shape-compatible policy can
still be semantically wrong because joint order, normalization, command slot,
or robot revision differs.

## 16.13 Uncertainty must alter permitted behavior

Confidence is useful only if it changes a decision. For a binary event such as
“path is clear,” calibration means predictions near confidence 0.8 are correct
about 80% of the time over a relevant test population. Reliability diagrams
bin predictions and compare confidence with empirical frequency. Expected
calibration error is a summary, but inspect the safety-critical low-clearance
region separately.

An onboard governor can combine age and uncertainty:

```python
def govern_speed(requested, map_msg, now_ms, brake_accel, margin):
    age_s = max(0.0, (now_ms - map_msg.capture_ms) / 1000.0)
    conservative_clearance = (
        map_msg.clearance_m - 2.33 * map_msg.clearance_std_m
    )
    if not map_msg.valid or conservative_clearance <= margin:
        return 0.0
    vmax = -brake_accel * age_s + (
        (brake_accel * age_s) ** 2
        + 2 * brake_accel * (conservative_clearance - margin)
    ) ** 0.5
    return max(0.0, min(requested, vmax))
```

This rule still assumes a verified braking model and a meaningful uncertainty
estimate. Test sensor blindness, frozen timestamps, overconfident false
clearance, frame mismatch, delayed inference, and controller rejection. A
fallback that was never fault-injected is only a design intention.

## 16.14 Worked JumpRover architecture

JumpRover is a wheeled-leg platform whose brain system-on-chip is partly
verified while mechanics, realtime board, and physical plant are not yet
accepted. That status changes the correct RL plan: architecture and interfaces
can progress, but dynamics training should not pretend unknown hardware is
final.

### Proposed multi-rate layers

| Layer | Example rate | Owns | Must not own |
| --- | ---: | --- | --- |
| motor electronics | 10–40 kHz | current/commutation, hard trips | semantics |
| realtime board | 0.5–2 kHz | sensors, limits, enable, watchdog | cloud calls |
| stabilizer/skill | 50–200 Hz | balance, contact, bounded motion | mission text |
| local planner | 10–30 Hz | collision-aware commands | motor current |
| perception | 5–30 Hz | tracks, map, covariance, age | actuator enable |
| mission manager | 1–10 Hz | options, recovery, lifecycle | raw torque |
| cloud agent | asynchronous | semantic proposal/explanation | hard deadlines |

Rates are hypotheses until measured. A brushless direct-current motor (BLDC)
stage may use field-oriented control (FOC) on the realtime board. The board's
microcontroller unit (MCU) must retain enable, limits, heartbeat, and safe-stop
authority even if the SG2002 brain crashes.

### Interface sequence

```text
camera/IMU/encoders
       |
       +--> timestamped state estimator --> local belief
       |                                  |
       +--> visual perception ------------+--> local planner
                                               |
cloud proposal --> mission validator --> option command + bounds
                                               |
                                  policy/command governor
                                               |
                                  realtime board + watchdog
                                               |
                                         power stage/motors
```

Every arrow needs units, frame, rate, timestamp, sequence number, validity,
timeout, and behavior on loss. The diagram is not an implementation until
those contracts and tests exist.

### Hardware gates before dynamics learning

Do not begin expensive locomotion training against guessed mechanics. Require:

1. accepted mass, inertia, center of mass, wheel radius, linkage geometry, and
   joint limits from the physical build;
2. measured command-to-torque or command-to-force behavior across speed,
   voltage, and temperature;
3. measured latency, jitter, saturation, dead zone, friction, backlash, and
   braking distance;
4. synchronized encoder and IMU logs under controlled excitation;
5. proven watchdog, disable, overcurrent, overtemperature, tilt, and
   communication-loss behavior; and
6. a deterministic non-learning baseline that can safely exercise the plant.

These measurements identify a simulator distribution. Domain randomization
then covers credible residual uncertainty; it should not conceal absent
mechanical knowledge.

### Three staged learning projects

**Stage A: semantic autonomy without learned dynamics.** Use local mapping,
geometric planning, and conservative hand-engineered drive/stabilization.
Connect cloud planning only through typed options. This validates the entire
authority and freshness path.

**Stage B: bounded local RL skill.** Train a proprioceptive stabilizer or
command tracker after plant identification. Keep the planner and collision
governor outside. Verify simulation, processor-in-loop, hardware-in-loop, then
tethered physical trials.

**Stage C: perceptive mobility research.** Add height maps, visual history, or
latent terrain observations. Compare against Stage B under identical held-out
terrain and faults. The new policy must beat the modular baseline in more than
average reward: success, collision, intervention, tail latency, energy, and
recovery all matter.

### Acceptance evidence for handoff

A team receiving the RL stack should get:

- versioned mechanical/electrical parameters and calibration;
- simulator and system-identification residuals;
- logged observation/action schema with replay tooling;
- policy manifest, training commit, seed/config, and dataset provenance;
- checkpoint-selection protocol and held-out evaluation report;
- timing traces on target compute;
- fault-injection and fallback results;
- physical operating envelope and known exclusions; and
- rollback image plus named release owner.

This is how Microduck's learning transfers: not by copying its 14-joint network,
but by copying disciplined contracts, measurement, reward audits, staged
training, artifact provenance, and sim-to-real tests.

## 16.15 Worked design: person following

Consider “follow Bruce to the workshop.” Decompose the problem before choosing
a model.

1. The cloud or onboard semantic model resolves the request to an authorized
   `follow_person` option and subject identifier (ID).
2. Onboard perception detects and tracks the subject, returning bearing,
   range, identity confidence, covariance, capture time, and track age.
3. A local planner combines that track with obstacles and produces a bounded
   velocity command.
4. The local locomotion policy tracks the command; a governor clips it using
   stopping distance and tilt limits.
5. The realtime board enforces communication timeout and motor limits.
6. Track loss, identity switch, low clearance, excessive age, or human stop
   causes local stop; reacquisition does not occur silently outside consent
   rules.

Evaluate the chain with a factorial test matrix:

- known and unseen clothing/backgrounds;
- crossing people and partial/full occlusion;
- bright, dim, backlit, and motion-blurred video;
- static, approaching, departing, and abruptly stopping subject;
- narrow passages and moving obstacles;
- injected perception delay, dropped frames, and network loss; and
- authorized versus unauthorized people.

Report false-follow rate, missed-follow rate, identity switches, range error,
map age, minimum clearance, stop latency, intervention rate, and mission
success. A single success percentage cannot reveal whether the robot followed
the wrong person dangerously.

## 16.16 Exercises

1. A camera has $f_x=500$, $c_x=320$. A pixel at $u=370$ has depth 2 m.
   Compute $X$ in the camera frame.
2. Explain the difference between intrinsic and extrinsic calibration, and
   give one failure caused by each.
3. A rover moves at 0.6 m/s and its map is 250 ms old. How far has it traveled
   since image capture, before adding braking distance?
4. Derive the stopping-distance equation from constant-acceleration motion.
5. With mean clearance 0.8 m, standard deviation 0.1 m, and $k=2$, what is
   conservative clearance? Why might the claimed tail probability be wrong?
6. Explain why a visible simulated obstacle is not necessarily an actor
   observation.
7. Compare modular and end-to-end obstacle avoidance for Microduck.
8. What does a linear probe establish, and what does it fail to establish?
9. Why can direct squared-error action prediction be poor for two equally
   valid ways around an obstacle?
10. If an action chunk has $H=20$ at 50 Hz, how much future time does it
    cover? Give one benefit and one risk.
11. Explain flow-matching training and inference without using the phrase
    “it denoises.”
12. Why can FAST use fewer tokens than independent timestep quantization for
    smooth robot motion?
13. Distinguish VLA imitation pretraining from online RL fine-tuning.
14. An option lasts four primitive steps with rewards $(1,1,0,2)$ and
    $\gamma=0.9$. Compute its internal discounted return and the discount on
    the next-option value.
15. Construct a case in which a language planner gives a skill high semantic
    probability but the affordance probability should be near zero.
16. List six checks an onboard validator must perform on a cloud skill request.
17. Design a policy manifest for one Microduck behavior, including one
    exclusion that prevents a semantic mismatch.
18. Propose the first JumpRover learning experiment that is valid before the
    final mechanics exist, and one experiment that must wait.

Continue with
[research literacy and capstone projects](17_research_literacy_and_capstones.md).

## 16.17 Folded solutions

<details>
<summary>Show worked answers to Section 16.16</summary>

1. Back-project along the horizontal axis:

   ```math
   X=Z(u-c_x)/f_x=2(370-320)/500=0.20\ \mathrm{m}.
   ```

   The point is 20 cm along the camera's positive horizontal axis under the
   declared convention.

2. Intrinsics describe projection inside the camera: focal lengths, principal
   point, and distortion. Wrong intrinsics can bend a straight wall or give
   incorrect lateral position. Extrinsics describe the rigid transform between
   camera and robot/world frames. A reversed or stale extrinsic can put a
   correctly detected obstacle on the wrong side of the rover.

3. Age distance is

   ```math
   d=v\tau=0.6(0.250)=0.15\ \mathrm{m}.
   ```

   The rover traveled 15 cm before considering planner delay or braking.

4. During reaction delay, constant speed contributes $v\tau$. During
   braking, use $v_f^2=v_i^2+2ad$ with $v_f=0$ and $a=-a_b$, yielding
   $d=v^2/(2a_b)$. Add a residual margin $m$, giving
   $d_{\mathrm{stop}}=v\tau+v^2/(2a_b)+m$.

5. The conservative clearance is $0.8-2(0.1)=0.6$ m. A Gaussian tail
   interpretation may be wrong because errors can be skewed, heavy-tailed,
   correlated, shifted by domain change, or miscalibrated. Check empirical
   coverage on relevant held-out scenes.

6. Rendering changes the pixels a human sees. An actor responds only to values
   included in its observation tensor or commands derived from sensors.
   Microduck's current 61D actor has no obstacle range, pixels, or map.

7. The modular route converts perception into safe velocity commands for the
   existing actor, preserves the observation contract, and enables component
   replay tests. End-to-end depth-to-joints may coordinate foot placement more
   tightly but needs a new policy, sensor simulation/randomization, runtime
   schema, much more data, and separate safety validation. Begin with the
   modular baseline so the research comparison is meaningful.

8. A linear probe shows that a property is linearly decodable from a frozen
   representation on the probe distribution. It does not show that the control
   policy uses that property, that the correlation is causal, or that it
   survives deployment shift. Use masking/intervention and held-out tests.

9. If demonstrations turn left and right with equal probability, squared error
   favors their conditional mean. The mean may command straight ahead—the only
   colliding action. A multimodal head, explicit route variable, or planner can
   preserve the alternatives.

10. The chunk covers $20/50=0.4$ s. It reduces inference frequency and learns
    coordinated trajectories. It may continue executing stale intent for up to
    that horizon unless receding-horizon replanning or interruption is used.

11. Training samples a demonstrated chunk, a noise chunk, and an interpolation
    time. The model learns the velocity that transports the interpolated point
    along a path toward the data distribution, conditioned on observations.
    Inference samples noise and numerically integrates this learned velocity
    field to produce a continuous action chunk.

12. Smooth trajectories have correlated timesteps and most energy in a few
    low-frequency coefficients. A frequency transform exposes that structure,
    allowing compression to represent a whole chunk with fewer symbols than
    quantizing every component at every time independently.

13. VLA imitation fits demonstrated actions conditioned on observations and
    language using supervised likelihood/regression/generative losses. Online
    RL collects reward-bearing interaction under the current policy and changes
    behavior to improve expected return. A later RL-alignment phase does not
    retroactively make all pretraining RL.

14. The internal return is

    ```math
    R_o=1+0.9(1)+0.9^2(0)+0.9^3(2)=3.358.
    ```

    The next-option value is multiplied by $\gamma^4=0.6561$.

15. For “clean the spill,” `pick_up_sponge` is linguistically useful, but its
    affordance should approach zero if the sponge is behind a locked door or
    absent from the map. Semantic relevance does not create reachability.

16. At minimum validate authentication/authorization, exact schema and units,
    request ID/replay status, expiry/freshness, skill allow-list and version,
    parameter bounds, current preconditions/affordance, and an available
    fallback. Six are requested; production should perform all of these.

17. A valid Microduck manifest declares model hash, training commit, robot
    revision, 61D input schema, 14D output order/scale, embedded observation
    normalization, 50 Hz rate, measured inference bound, command limits,
    evaluator report, and fallback. `known_exclusions: [obstacle_avoidance]`
    prevents a planner from assuming that a flat-ground velocity tracker sees
    obstacles.

18. Before final mechanics, validate the typed cloud-to-onboard option path in
    a software simulator with stale/replayed/invalid messages and prove local
    stop under disconnection. Training a dynamics-sensitive jump or balance
    policy must wait for accepted geometry, mass/inertia, actuators, realtime
    timing, braking, synchronized logs, and safety interlocks.

</details>
