# 8. Software and Control Architecture

An RL experiment is not just a neural network. It is a chain of models,
interfaces, clocks, and artifacts. A policy can be mathematically correct and
still fail if one link changes between training and deployment.

## 8.1 The project stack

```text
Onshape/CAD and measured robot facts
        |
        v
MuJoCo XML model (MJCF) + collision models + HOME pose
        |
        v
MuJoCo physics through MuJoCo Warp on the GPU
        |
        +---- BAM XL330 actuator model
        +---- sensors, contacts, delays, noise, domain randomization
        |
        v
mjlab manager-based RL environment
        |
        v
RSL-RL PPO actor/critic and rollout storage
        |
        +---- checkpoints and metrics
        v
ONNX export with observation normalizer
        |
        v
CPU runtime at 50 Hz on the real robot
```

The major dependencies have distinct jobs:

| Component | Responsibility |
| --- | --- |
| MuJoCo | rigid-body dynamics, joints, contacts, sensors |
| MuJoCo Warp / Warp | CUDA/GPU-parallel simulation of many environments |
| `mjlab` | environment managers, task registry, vectorized wrapper, viewer, export support |
| RSL-RL | PPO, neural networks, rollout buffers, checkpoints |
| BAM (Better Actuator Models) | voltage-aware Dynamixel XL330 actuator and friction behavior |
| this repository | Microduck robot models, tasks, rewards, randomization, tests, export and rehearsal |
| Microduck runtime repository | real sensors, command sources, ONNX inference, motor communication |

MJCF is MuJoCo's XML-based robot/model description format. CUDA is NVIDIA's
GPU-computing platform. ONNX (Open Neural Network Exchange) is a portable
neural-network file format. **Inference** means running a frozen trained
network to obtain an output; unlike training, it does not update weights.

## 8.2 Repository map as a learning map

```text
src/mjlab_microduck/
├── robot/
│   ├── microduck_constants.py       robot choice, HOME pose, actuator setup
│   └── microduck/                   MJCF, scenes, meshes, backlash generator
├── actuator/
│   └── friction_dr_bam.py           actuator friction DR and backlash feedback
├── tasks/
│   ├── __init__.py                  task IDs and runner registration
│   ├── mdp.py                       custom rewards, events, observations, curricula
│   ├── symmetry.py                  optional 61D mirror transform
│   ├── backlash.py                  base-task to backlash-task wrapper
│   └── microduck_*_env_cfg.py       task-family configuration factories
└── train_cli.py                     local/Hugging Face training entry point

scripts/
├── export.py                        checkpoint -> normalized ONNX
├── infer_policy.py                  CPU MuJoCo deployment rehearsal
└── testbench_sim2real.py            measured actuator comparison tools

tests/                               CPU regression and configuration tests
logs/rsl_rl/<experiment>/<run>/      checkpoints, parameters, metrics, videos
```

Read in dependency order. Start with the task registration, then its
configuration factory, then follow only the custom functions it references
into `mdp.py`. Reading all of the large MDP module from top to bottom is not a
good beginner exercise.

## 8.3 Task registration

Every user-facing task ID binds four things:

```python
# Abridged from tasks/__init__.py
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

- `env_cfg` is the training environment.
- `play_env_cfg` makes one policy easier to inspect, often with fewer or more
  visible disturbances.
- `rl_cfg` defines the networks and PPO hyperparameters.
- `runner_cls` connects the environment to RSL-RL.

`uv run list-envs` reads this live registry. Documentation can become stale;
the registry is executable truth.

## 8.4 Manager-based environment anatomy

`mjlab` divides environment behavior into managers:

| Manager/config section | Question it answers |
| --- | --- |
| `scene` | What robot, terrain, sensors, and spacing exist? |
| `actions` | How do policy outputs become simulator controls? |
| `observations` | What numeric information reaches actor and critic? |
| `commands` | What behavior is requested this episode? |
| `events` | What changes at startup, reset, or intervals? |
| `rewards` | What outcomes produce learning signal? |
| `terminations` | When is an episode over? |
| `curriculum` | How does difficulty or a parameter change with training steps? |
| `metrics` | What diagnostic values are logged without changing learning? |

The velocity factory starts from `mjlab`'s closest locomotion template and
modifies it:

```python
def make_microduck_velocity_env_cfg(play=False, rough=False):
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (
        feet_ground_cfg,
        self_collision_cfg,
        foot_height_scan_cfg,
    )
    # Then specialize actions, observations, commands, rewards, DR, and terrain.
    return cfg
```

Building on a maintained template retains important sensor and safety wiring.
A standalone task must recreate that wiring deliberately.

## 8.5 One 20 ms policy step

The physics timestep is 5 ms and action **decimation** is 4, meaning one policy
decision is reused for four smaller physics steps. One policy action is
therefore held across four physics steps:

```text
t = 0 ms     assemble observation; run actor; set action target
t = 5 ms     physics + actuator substep
t = 10 ms    physics + actuator substep
t = 15 ms    physics + actuator substep
t = 20 ms    physics + actuator substep; compute next observation/reward
```

This gives a 50 Hz policy rate while resolving contacts and actuator dynamics
at 200 Hz. Changing either clock changes the control problem and must be
matched in deployment.

## 8.6 Observation contract

The actor layout is intentionally fixed across the policy family:

```text
[0:3]    base angular velocity
[3:6]    projected gravity
[6:20]   14 relative joint positions
[20:34]  14 joint velocities
[34:48]  14 previous actions
[48:51]  twist: vx, vy, yaw rate
[51:55]  head pose: four joint deltas
[55:61]  body pose: x, y, z, roll, pitch, yaw deltas
```

Unused command slots remain present and are zero-padded or sampled over tiny
ranges. Deleting a term would change the input size and prevent runtime policy
hot-swapping.

The actor excludes true base linear velocity because the real robot does not
have a direct perfect measurement. The critic includes it because privileged
information can improve value estimates during simulation training.

Observation corruption models hardware imperfections. The broader practice of
sampling physical parameters and disturbances is called **domain
randomization**: training across a range of simulated domains so the real robot
is less likely to fall outside the learned experience.

- small additive sensor noise;
- encoder bias;
- IMU mounting variation;
- delayed angular velocity, gravity, joint velocity, and actuation;
- backlash-specific encoder views.

If the actor sees a biased or backlash-adjusted quantity, a tracking reward on
that quantity must use the same view. Otherwise the policy is punished for
correcting the measurement it receives.

## 8.7 Action contract

The actor emits 14 floating-point values. The joint-position action uses
`scale=1.0` and the model's default joint pose as its offset. Conceptually:

```math
q^{target} = q^{HOME} + a
```

The exact action manager also applies its configured ordering and any limits.
The target then enters the BAM actuator model; it is not an ideal instantaneous
joint angle.

The 14 servo order is:

```text
0..4    left hip yaw, hip roll, hip pitch, knee, ankle
5..8    neck pitch, head pitch, head yaw, head roll
9..13   right hip yaw, hip roll, hip pitch, knee, ankle
```

Roller and backlash MJCF models contain interleaved passive joints. Custom MDP
code must use `_servo_joint_ids` and `_servo_joint_pos` rather than assuming
that MuJoCo joint index equals servo index. All unactuated joints begin with
`passive_`, and actuator/observation selectors exclude that prefix.

## 8.8 The actuator is part of the environment

The real XL330 is not an ideal position source. The BAM configuration includes
firmware position-loop behavior, motor electrical effects, torque limits,
friction, battery voltage, load-dependent voltage sag, and command delay.

The main configuration randomizes values such as:

```python
FrictionDRBamActuatorCfg(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=200.0,
    vin_range=(6.5, 8.2),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    delay_min_lag=3,
    delay_max_lag=6,
)
```

This is a real project example of model-based sim-to-real engineering: the
policy learns to control a family of plausible actuators rather than a perfect
one. Under BAM, MuJoCo's ordinary `dof_frictionloss` is not the friction
authority; friction randomization must scale the actuator's own field.

## 8.9 Three robot models, one policy interface

| Model | Why it exists |
| --- | --- |
| `robot_walk.xml` | light walking model; trunk/head ground contacts are stripped |
| `robot_allcollisions.xml` | lying, stand-up, tricks, and body contacts |
| `robot_allcollisions_rollers.xml` | full body plus passive wheel joints |

Backlash versions add an unactuated hinge in series with every servo while
keeping actor and action dimensions unchanged. A base and backlash A/B test
must use otherwise matching robot models or the comparison is confounded.

## 8.10 Training artifacts and provenance

A run directory normally contains:

```text
params/env.yaml       resolved environment used for the run
params/agent.yaml     resolved PPO/network configuration
model_<n>.pt          actor, critic, normalizers, optimizer, counters
events.out...         local scalar logs
*.onnx                exported inference policy, when exported
videos/play/          finite recorded rollouts, when requested
```

The resolved YAML is valuable. Source code may change after a run; the saved
parameters show what the checkpoint actually trained against. Keep it with any
policy you evaluate or deploy.

## 8.11 Lab: trace a task without training

1. List the registry:

   ```bash
   uv run list-envs
   ```

2. Find the registration of `Mjlab-Velocity-Flat-MicroDuck` in
   `src/mjlab_microduck/tasks/__init__.py`.
3. Open `make_microduck_velocity_env_cfg` and identify one scene sensor, one
   command, one reward, one termination, and one curriculum.
4. Find the actor and critic hidden layers in `MicroduckRlCfg`.
5. Explain which of those objects exists during real ONNX inference.

Continue with
[Microduck setup and the first experiment](09_microduck_setup_and_first_experiment.md).

## 8.12 Folded lab solution

<details>
<summary>Show a reference trace for Section 8.11</summary>

1. `uv run list-envs` proves registration through the installed project rather
   than a filename guess. Preserve the exact task ID it prints.
2. `tasks/__init__.py` registers `Mjlab-Velocity-Flat-MicroDuck` with
   `make_microduck_velocity_env_cfg()`, a play configuration, `MicroduckRlCfg`,
   and `MicroduckOnPolicyRunner`.
3. Valid examples include the `foot_height_scan` terrain ray sensor; `twist`
   command; `track_linear_velocity` reward; `nan_state` termination; and
   `action_rate_weight` curriculum. The important result is the path from
   factory to a named manager term, not memorizing these examples.
4. Actor and critic both use hidden dimensions `(512, 256, 128)` with ELU
   activation and observation normalization. Their inputs differ because the
   critic may receive privileged observations.
5. Real ONNX inference deploys the actor and its actor-observation normalizer.
   The critic, reward/termination/event/curriculum managers, rollout storage,
   PPO losses, optimizer, and simulator randomization exist only to generate or
   improve the checkpoint.

A repeatable source trace is:

```bash
rg -n 'Mjlab-Velocity-Flat-MicroDuck' src/mjlab_microduck/tasks/__init__.py
rg -n 'MicroduckRlCfg|hidden_dims|obs_normalization' \
  src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py
```

</details>
