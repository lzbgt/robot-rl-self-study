# 16. Vision, Foundation Policies, and Hierarchical Autonomy

A robot that walks is not automatically a robot that knows where to go. This
chapter separates perception, state estimation, local control, planning, and
cloud reasoning, then shows where modern vision-language-action models fit.

## 16.1 Perception changes the observation problem

**Proprioception** measures the robot itself: joint encoders, IMU, motor state.
**Exteroception** measures the surrounding world: camera, depth, LiDAR, radar,
or tactile arrays.

Images are high-dimensional and ambiguous. One RGB frame does not directly
state depth, velocity, object identity, traversability, or occlusion. A visual
control system commonly needs:

- calibration (intrinsics/extrinsics);
- timestamps and synchronization;
- an encoder or geometric perception pipeline;
- temporal history for motion and occlusion;
- an explicit confidence/validity representation; and
- behavior when the sensor is delayed, dark, blocked, or missing.

## 16.2 Three ways to connect vision to control

### Modular perception and planning

```text
camera -> detector/depth/SLAM -> map or local obstacles
       -> planner -> safe velocity command -> locomotion policy
```

Advantages: inspectable intermediate outputs, reusable mapping/planning,
easier geometric safety constraints. Risks: interface errors and perception
mistakes propagate.

### End-to-end visual policy

```text
camera + proprioception + goal -> neural policy -> motor/task-space action
```

Advantages: features optimize for the task; may exploit cues omitted by a hand
designed representation. Risks: high data demand, harder diagnosis,
distribution shift, and latency.

### Hybrid

```text
camera -> pretrained/learned encoder -> compact latent or terrain map
       + proprioception -> local policy
       + geometric safety layer
```

Most real systems are hybrids somewhere, even if a paper calls one block
“end-to-end.” Motor drivers, limits, synchronization, and emergency handling
remain outside the learned model.

## 16.3 Obstacles must affect information and objective

Putting boxes in a simulator does not teach obstacle avoidance. At minimum:

1. the agent must receive pre-contact obstacle information or a planner must
   convert it into commands;
2. training must vary obstacle layouts enough to prevent memorization;
3. reward/constraints must distinguish safe progress from collision;
4. resets must expose relevant decisions; and
5. evaluation must hold out shapes, positions, textures, and sensor failures.

For Microduck, two valid projects are:

### Project A: preserve the locomotion actor

```text
depth/camera -> local map -> collision-aware planner -> [vx, vy, yaw rate]
                                                -> existing velocity actor
```

This is the lower-risk route when the current actor already tracks commands.

### Project B: train perceptive locomotion

```text
depth/history + proprioception + goal -> new actor -> 14 joint targets
```

This can coordinate footsteps with terrain, but it changes observation
contract, training cost, runtime compute, failure modes, and sim-to-real burden.

## 16.4 Representation learning

A visual encoder maps an image $I_t$ to a smaller feature vector:

```math
z_t=f_\psi(I_t).
```

The encoder can be:

- trained end-to-end from task reward;
- pretrained with supervised labels;
- pretrained self-supervised on images/video;
- frozen during policy learning; or
- jointly fine-tuned.

A visually rich latent is not necessarily control-relevant. Conversely, a
task-trained latent can discard object attributes needed by future tasks.
Probe latents with held-out conditions and downstream success rather than
judging a visualization alone.

## 16.5 Generalist and foundation robot policies

A **generalist robot policy** is trained across many tasks, datasets, or robot
embodiments instead of from scratch for one skill. A **foundation model** is a
broadly pretrained model intended to adapt to many downstream tasks. These
terms describe scope and reuse, not guaranteed general intelligence.

A Vision-Language-Action (VLA) policy typically maps images and language,
sometimes proprioception, to robot actions or action tokens:

```math
a_{t:t+H}=\pi(I_{t-k:t},\text{language},q_t).
```

Most VLA pretraining is imitation/supervised sequence modeling on robot
demonstrations, not classical online RL. RL may later fine-tune preferences or
task reward, but the terms should not be conflated.

## 16.6 Octo: an open generalist policy

The 2024 [Octo paper](https://arxiv.org/abs/2405.12213) presents an open-source
transformer-based policy pretrained on 800,000 trajectories from the Open
X-Embodiment dataset and studies adaptation across robot setups.

Study questions:

- How are different observation and action spaces tokenized/normalized?
- What part is pretrained versus newly initialized for a robot?
- Which task/robot combinations are evaluated zero-shot versus fine-tuned?
- What inference hardware and rate are used?
- Does the target task exist in the data distribution?

The transferable idea is diverse pretraining plus explicit embodiment adapters.
The pretrained weights do not know a new robot's safety limits automatically.

## 16.7 OpenVLA

The 2024 [OpenVLA paper](https://arxiv.org/abs/2406.09246) presents an open
7-billion-parameter VLA trained on 970,000 real-robot demonstrations, combining
pretrained visual/language components with action prediction. It also studies
parameter-efficient fine-tuning and quantized serving.

Those scales are dramatically different from Microduck's small 50 Hz MLP.
OpenVLA may help interpret language and visual manipulation tasks; placing a
multi-billion-parameter model directly in a hard balance loop would require a
completely different latency, reliability, and safety case.

## 16.8 $\pi_0$ and flow-matching action generation

The 2024 [$\pi_0$ paper](https://arxiv.org/abs/2410.24164) combines a pretrained
vision-language model with **flow matching**, a generative approach for
continuous action sequences. It evaluates diverse manipulation behaviors
across several robot platforms.

The research direction is important: semantic visual-language knowledge and
continuous dexterous action generation can be trained together. The correct
engineering questions remain concrete:

- What data overlaps the target task and embodiment?
- What is the control horizon and replan rate?
- How are actions normalized and converted to hardware commands?
- What happens under instruction ambiguity or sensor shift?
- Which low-level and safety controllers remain outside the model?

## 16.9 Foundation policy versus local motor skill

These layers solve different timescales:

| Layer | Example output | Typical concern |
| --- | --- | --- |
| language/semantic agent | exact skill request and parameters | grounding, hallucination, network delay |
| task planner | ordered skill graph | feasibility and recovery |
| navigation/perception | local path or velocity | obstacles, localization, uncertainty |
| learned motor skill | joint/velocity targets | balance, contact, actuator mismatch |
| realtime controller | current/PWM/enable | deadlines and hard limits |

A large model can choose “follow the person at 0.3 m/s.” It should not invent a
raw 20 kHz PWM stream. A small local policy can stabilize a commanded motion
without understanding the sentence “follow Bruce to the workshop.”

## 16.10 Hierarchical reinforcement learning

In **hierarchical RL**, a higher policy chooses a skill, goal, latent, or
subtask; a lower policy executes it.

An option is often described by:

- initiation set: states where it can start;
- internal policy: actions while active; and
- termination condition: when it ends.

For a robot:

```text
option: dock
preconditions: dock detected, localization confident, battery state valid
parameters: dock ID, approach side
termination: charging contact confirmed or timeout/fault
```

The option boundary is also an API and safety boundary. Exact typed parameters
are safer than free-form text passed into control code.

## 16.11 Cloud agent: proposal, not authority

A cloud model can add:

- language understanding;
- long-horizon task decomposition;
- semantic interpretation of selected images/maps;
- retrieval of manuals or prior mission context; and
- human-facing explanation.

It also adds variable latency, disconnection, uncertain outputs, privacy/data
flow, service/version drift, and adversarial or ambiguous input.

A robust flow is:

```text
human request + summarized world state
                  |
                  v
cloud agent proposes exact option IDs/parameters
                  |
                  v
local schema validation + precondition check
                  |
                  v
local planner / behavior tree / safety filter
                  |
                  v
local motor skill -> realtime controller
```

The cloud does not send torque, PWM, motor current, balance correction, or
unbounded wheel/servo targets. Loss of cloud must leave local stop and balance
available.

## 16.12 Runtime contracts for learned skills

Package a skill with a manifest, not just a model file:

```yaml
policy_id: walk-flat-v3
model_sha256: "..."
training_code_commit: "..."
robot_model_revision: "..."
input_schema_version: microduck-obs-61-v1
output_schema_version: joint-delta-14-v1
policy_rate_hz: 50
max_inference_ms: 6.0
normalization: embedded
command_bounds:
  vx_mps: [-0.3, 0.3]
  yaw_rps: [-1.0, 1.0]
validated_conditions: [flat_indoor, nominal_voltage]
known_exclusions: [stairs, obstacle_avoidance]
fallback_policy_id: stand-safe-v2
```

The manifest prevents a semantically wrong but shape-compatible model from
being hot-swapped.

## 16.13 Perception uncertainty should change behavior

Do not reduce every perception output to a confident point estimate. Useful
signals include confidence, age, covariance, valid-region mask, and source.

Example local rule:

```python
if obstacle_map.age_ms > 200 or obstacle_map.confidence < 0.7:
    requested_speed = min(requested_speed, 0.05)
if obstacle_map.invalid:
    request_stop()
```

Thresholds need system-level validation. The principle is that uncertainty
affects permitted behavior rather than being logged and ignored.

## 16.14 Worked Jump Rover architecture example

For a wheeled-leg rover with an SG2002 brain and a realtime motion MCU:

1. keep motor current/FOC, enable, limits, watchdog, and emergency behavior on
   realtime hardware;
2. run a bounded balance/drive policy only where worst-case timing is proven;
3. run perception, world state, local planning, and skill orchestration on the
   onboard brain;
4. let the cloud propose semantic goals or exact skills asynchronously; and
5. ensure network loss triggers no unsafe motor-state transition.

Chapter 17 turns this into a capstone. The Jump Rover repository's project
handoff must additionally gate training on measured mechanics and accepted
realtime-controller evidence.

## 16.15 Exercises

1. Explain why a visible simulated obstacle is not an actor observation.
2. Compare modular and end-to-end obstacle avoidance for a small biped.
3. Why is a VLA usually closer to imitation learning than online RL?
4. Design an exact JSON-like option request for “follow person,” including
   speed and stop conditions but no raw motor command.
5. What should happen if cloud response arrives 3 seconds late?
6. List the latency budgets that must be measured before placing a visual model
   in a control loop.
7. Write a policy manifest for one Microduck behavior.
8. Give three held-out tests for a person-following perception system.

Continue with [research literacy and capstone projects](17_research_literacy_and_capstones.md).

## 16.16 Folded solutions

<details>
<summary>Show reference answers to Section 16.15</summary>

1. Rendering affects what a human sees. The actor can respond only to tensors
   actually included in its observation. Microduck's 61D actor has no pixels,
   depth, range, or obstacle map.
2. A modular design lets local perception/planning convert geometry into twist
   commands for the existing fast policy; components are easier to test and
   the 61D contract survives. End-to-end depth-to-joint control may coordinate
   tighter maneuvers but needs far more data, a new runtime schema, sensor
   randomization, latency proof, and independent safety.
3. A VLA is usually trained to predict actions from logged vision/language/
   action demonstrations with a supervised sequence loss. A neural policy does
   not become RL unless reward-bearing interaction or an RL objective actually
   updates it.
4. One bounded future option could be:

   ```json
   {
     "skill": "follow_person",
     "track_id": "person.bruce",
     "following_distance_m": 1.2,
     "max_speed_mps": 0.15,
     "deadline_ms": 1500,
     "stop_if": {
       "track_age_ms_gt": 200,
       "map_age_ms_gt": 200,
       "minimum_clearance_m_lt": 0.35
     }
   }
   ```

   A local signed schema, identity/consent rule, planner, and stop path must
   validate it. There is no PWM, torque, wheel speed, or servo target.
5. Reject a response that arrives after its bound or revalidate it against the
   current world and issue a fresh request. Never execute stale semantic intent
   merely because the JSON was valid when generated.
6. Measure capture/exposure, preprocessing, queueing, inference mean/tail/WCET,
   postprocessing, transport, planner, actuation delivery, total observation
   age, jitter, missed deadlines, and fallback transition.
7. A minimal manifest needs policy/model hashes, training commit, robot/model
   revision, input/output schemas, normalization, action scaling, policy rate,
   maximum input age, inference WCET, validated operating envelope, exclusions,
   calibration, evaluator report, and rollback artifact.
8. Hold out a person/appearance, lighting/background, and motion/occlusion
   pattern. Also test crossing identities and complete track loss; report false
   follow, missed follow, ID switch, age, localization error, and stop latency.

</details>
