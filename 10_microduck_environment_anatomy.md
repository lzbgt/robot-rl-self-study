# 10. Anatomy of a Microduck Environment

An environment is an executable task specification. It defines what the robot
can observe and do, how the world changes, what counts as progress, and which
experience PPO collects.

This chapter uses `Mjlab-Velocity-Flat-MicroDuck` as the main worked example.

## 10.1 Start from the factory

The task registry points to:

```python
make_microduck_velocity_env_cfg(play=False, rough=False)
```

The factory begins with `mjlab`'s velocity template and then specializes it for
Microduck. This is an important design pattern:

```python
cfg = make_velocity_env_cfg()
cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
cfg.scene.sensors = (
    feet_ground_cfg,
    self_collision_cfg,
    foot_height_scan_cfg,
)
```

The resulting object is still just configuration. Managers copy and resolve it
when an environment instance is constructed.

## 10.2 Scene: the physical question

The scene selects:

- the robot MJCF and its initial state;
- terrain;
- collision rules;
- contact and ray sensors;
- environment count and spacing; and
- viewer camera.

Flat velocity uses a plane. Rough velocity uses a generator with robot-scale
terrain; step heights are capped around 1.5 cm because carrying human-scale
terrain assumptions to a 25 cm robot would create a different task.

The walking model has deliberately simplified fall contacts. Tasks that begin
or finish on the ground use the all-collisions model. Model choice is part of
the task definition, not a rendering preference.

### Try it: compare resolved scenes

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck --agent zero --num-envs 1
uv run play Mjlab-Velocity-Rough-MicroDuck --agent zero --num-envs 1
```

A zero agent is useful for inspecting physics and initial conditions without
attributing motion to a policy.

## 10.3 Actions: the controllable question

The velocity task has one action term with dimension 14:

```python
joint_pos_action = cfg.actions["joint_pos"]
joint_pos_action.scale = 1.0
```

The action manager interprets each network output as an offset from the default
joint pose. BAM then turns the position target into voltage/torque behavior.

Action design choices include:

- position, velocity, torque, or residual targets;
- scale and offset;
- clipping;
- ordering; and
- delay/filter behavior.

An action space should match what the runtime can reproduce. Training with an
action low-pass filter and deploying without it creates a different plant;
deploying a filter that training never saw does the same. This policy family is
intentionally unfiltered.

## 10.4 Observations: the information question

An observation term is a function plus optional noise, scale, clipping, delay,
and history. Actor terms are concatenated in order.

The main actor gets 48 proprioceptive values plus a 13D command block. The
critic receives additional simulator-only signals. Two principles matter:

1. **Availability:** actor inputs must be reproducible on hardware.
2. **Meaning:** training and runtime must use the same units, frame, sign,
   ordering, offset, delay, and filtering.

For example, “angular velocity” is incomplete documentation. You must know:

```text
world or body frame?
rad/s or deg/s?
which axis signs?
raw, calibrated, or filtered?
current sample or delayed sample?
```

The current actor intentionally removes perfect base linear velocity and the
general terrain height scan. A per-foot ray sensor supports foot-height
features and rewards, but there is no forward obstacle representation in the
actor.

## 10.5 Commands: the intention question

Commands are targets sampled by the training environment and supplied by an
operator or planner at deployment.

The velocity task's twist command uses these ranges:

```text
forward velocity vx       -0.4 .. +0.4 m/s
lateral velocity vy       -0.3 .. +0.3 m/s
yaw rate                  -1.0 .. +1.0 rad/s
resampling interval        3.0 .. 8.0 s
```

It also creates explicit buckets:

- standing/zero-command environments, because continuous uniform sampling
  almost never produces exactly zero;
- forward-command environments; and
- turn-in-place environments with zero linear velocity and a meaningful yaw
  command.

The turn-in-place code is a useful real example of fixing the data
distribution rather than tuning PPO:

```python
# Abridged from VelocityCommandCommandOnly._resample_command
turn_ids = env_ids[random_values < rel_turn_in_place_envs]
self.vel_command_b[turn_ids, 0:2] = 0.0
sign = where(random_values < 0.5, -1.0, 1.0)
mag = uniform(0.4 * max_yaw_rate, max_yaw_rate)
self.vel_command_b[turn_ids, 2] = sign * mag
```

Independent uniform sampling made useful spin-in-place examples too rare. An
explicit 15% bucket gives PPO enough on-policy data.

The head command is four joint deltas from HOME. Its range widens by
curriculum. The six-dimensional body-pose slot is retained, but its tracking
reward has weight zero in the main velocity policy; do not expect this
checkpoint to obey body-pose commands reliably.

## 10.6 Rewards: the objective question

The main velocity reward stack includes:

| Term | Purpose | Typical configured weight |
| --- | --- | ---: |
| linear velocity tracking | follow requested planar velocity | +2.0 |
| angular velocity tracking | follow requested turn rate | +2.0 |
| upright | keep trunk orientation useful | +2.0 |
| pose | retain a workable leg posture | +1.0 |
| air time | encourage stepping when commanded | +3.0 |
| head-pose tracking | follow four head targets | +2.0 |
| body angular velocity | discourage unnecessary rotation | -0.05 |
| angular momentum | reduce excess whole-body motion | -0.02 |
| action rate | reduce command jitter | starts at -0.1, ramps stronger |
| foot slip | reduce sliding without blocking turning | -0.1 |
| self-collision | avoid invalid body contact | -1.0 |

Weights cannot be compared in isolation. A term's scale, frequency, gate, and
the total positive reward mass determine its influence.

### Real code example: Gaussian head tracking

The custom reward compares the commanded head delta with measured joint delta:

```python
# Abridged from mdp.head_pose_tracking
cmd = env.command_manager.get_command("head_pose")       # (N, 4)
measured = joint_pos[:, neck_ids] + backlash_position
actual = measured - default_joint_pos[:, neck_ids]
error = actual - cmd
per_joint = torch.exp(-((error / std) ** 2))
return per_joint.mean(dim=-1)                            # (N,)
```

At `abs(error) == std`, one joint contributes $e^{-1}\approx0.37$. A very wide
standard deviation weakens the gradient near small errors; a very narrow one
can make far states nearly zero-reward and can tax unavoidable walking
oscillation.

The implementation reads through simulated backlash so the tracking reward and
encoder observation describe the same physical angle.

### Reward sign audit

There are two penalty styles in this project:

```python
# Style A: function returns nonnegative cost
cost = error.square()
# Configure a NEGATIVE weight.

# Style B: helper returns a nonpositive penalty
penalty = -error.abs()
# Configure a POSITIVE weight.
```

A negative weight on Style B makes violations profitable. The reliable runtime
check is:

```text
every Episode_Reward/<penalty> contribution must be <= 0
```

### Reward gates

A gate defines when a reward is meaningful. Foot air time should activate when
walking is commanded, not while standing. A recovery reward should pay for
progress toward upright rather than pay continuously for remaining fallen.

Hard state-based gates are often more effective than small penalties because
they define what counts as the maneuver. For a roll, support contact and
orientation axes can distinguish a real forward roll from a shoulder flop.

## 10.7 Events and domain randomization: the variation question

Event terms run in modes:

| Mode | Example |
| --- | --- |
| startup | foot friction, encoder bias fields, mass/inertia setup |
| reset | root pose, joints, CoM, actuator friction scale, armature |
| interval | velocity pushes every few seconds |

Domain randomization (DR) trains one policy over a distribution of plausible
robots:

```math
\theta_{physics} \sim p(\theta)
```

Microduck randomizes selected quantities such as friction, mass/inertia, CoM,
battery voltage/sag, armature, encoder bias, IMU alignment, delays, and pushes.

DR is not a substitute for calibration. Zero-centered IMU orientation
variation teaches tolerance to uncertain mounting magnitude; it cannot remove
a fixed systematic mounting bias. The runtime must calibrate that bias.

Custom randomization must restore defaults before applying a sampled change.
If a +5% mass change is added to the already-randomized mass every reset, the
distribution drifts outside the intended range during long training.

## 10.8 Terminations: the data-recycling question

The velocity task can terminate for timeout, falling, terrain bounds, or a
nonfinite state. A NaN guard checks joints, root state, and named sensor data:

```python
bad = ~torch.isfinite(data.joint_pos).all(dim=1)
bad |= ~torch.isfinite(data.joint_vel).all(dim=1)
bad |= ~torch.isfinite(data.root_link_pos_w).all(dim=1)
bad |= ~torch.isfinite(data.root_link_quat_w).all(dim=1)
bad |= ~torch.isfinite(data.root_link_lin_vel_w).all(dim=1)
bad |= ~torch.isfinite(data.root_link_ang_vel_w).all(dim=1)
```

Sensor-derived critic terms are separately sanitized so one invalid contact or
ray value does not poison the value network before the environment resets.

Do not reuse walking terminations blindly. A recovery task needs time on the
ground. An episodic trick may need a short horizon and a landing criterion.

## 10.9 Curricula: the pacing question

A curriculum changes the problem as training progresses. In this repository,
steps mean environment steps:

```text
curriculum step = PPO iteration * num_steps_per_env
                = PPO iteration * 24
```

The action-rate curriculum uses discrete stages:

```python
weight_stages = [
    {"step": 0,         "weight": -0.1},
    {"step": 500 * 24,  "weight": -0.2},
    {"step": 750 * 24,  "weight": -0.4},
    {"step": 1000 * 24, "weight": -0.6},
    {"step": 1250 * 24, "weight": -0.8},
    {"step": 1500 * 24, "weight": -1.0},
]
```

The helper mutates the live manager copy:

```python
term_cfg = env.reward_manager.get_term_cfg(reward_name)
term_cfg.weight = selected_stage["weight"]
```

Writing to `env.cfg.rewards` after manager initialization is a silent no-op
because managers copied their configurations.

A stage should follow demonstrated competence. If the main skill metric drops
exactly at every boundary, the schedule is outpacing the policy.

## 10.10 What the velocity policy can and cannot do

| Capability | In the current task? | Reason |
| --- | --- | --- |
| commanded forward/lateral walk | yes | command input and active tracking reward |
| commanded turn or stand | yes | explicit command buckets and tracking |
| commanded head pose | yes | 4D input and active reward |
| commanded body pose | not reliably | slot exists, reward weight is zero |
| global waypoint navigation | no | no waypoint/map state or navigation objective |
| obstacle avoidance | no | no forward obstacle observation or avoidance objective |
| visual decision making | no | camera pixels/features are not actor inputs |

Putting a box in the rendered scene does not create perception. An actor can
only condition behavior on information in its observation or inferable from
contact after collision.

Two sound extension architectures are:

1. a perception/planning system detects obstacles and sends safe local twist
   commands to the unchanged 61D locomotion policy; or
2. a deliberately versioned policy family receives exteroceptive features and
   is trained on obstacle-rich scenes with a new runtime contract.

The first preserves a small, fast, hot-swappable motor policy. The second may
learn tighter perception-action coupling but requires new training data,
network design, inference budgets, tests, and deployment plumbing.

## 10.11 Lab: predict behavior from configuration

Without running training, answer these from
`microduck_velocity_env_cfg.py`:

1. What command causes standing?
2. Why is turn-in-place sampled in its own bucket?
3. Which head joints are commandable?
4. Why does `body_pose` remain in the observation when its reward is zero?
5. Which randomization changes the actuator rather than MuJoCo joint friction?
6. At which PPO iteration does the action-rate weight first become `-0.6`?

Then run `uv run play ... --agent zero` and identify which scene facts can be
verified without a trained policy.

Continue with
[training, evaluation, and debugging](11_training_evaluation_and_debugging.md).

## 10.12 Folded lab answers

<details>
<summary>Show answers to the Section 10.11 configuration trace</summary>

1. Standing is an **exact zero twist command** selected by the command term's
   `rel_standing_envs` bucket. Near-zero uniform samples are not the same
   training case.
2. Independent continuous sampling makes simultaneous near-zero translation
   and meaningful yaw too rare. `rel_turn_in_place_envs` reserves a deliberate
   fraction of experience for that behavior.
3. The 4D head command controls `neck_pitch`, `head_pitch`, `head_yaw`, and
   `head_roll`, in that order.
4. The body slot preserves the shared 61D hot-swap interface and keeps those
   input weights available to policy-family tasks that train them. In the main
   velocity recipe the ranges remain small but `body_pose_tracking` has weight
   zero, so the interface does not prove body-pose skill.
5. `randomize_joint_friction` changes the BAM actuator's per-environment
   `friction_scale`. Randomizing MuJoCo `dof_frictionloss` would be a silent
   no-op because BAM owns this friction model.
6. The `-0.6` stage begins at `1000 * NUM_STEPS_PER_ENV`; with 24 rollout steps
   that is environment step 24,000, or PPO iteration 1,000.

With `--agent zero`, you can verify model loading, gravity, collision/contact
geometry, passive-joint behavior, reset poses, actuator plumbing, and whether a
zero action is numerically finite. You cannot verify learned command tracking,
gait quality, recovery, or sim-to-real robustness because no trained actor is
making decisions.

</details>
