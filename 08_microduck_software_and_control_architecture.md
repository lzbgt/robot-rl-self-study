# 8. Software and Control Architecture

A reinforcement learning (RL) experiment is not just a neural network. It is a
chain of models, interfaces, clocks, and artifacts. A policy can be
mathematically correct and still fail if one link changes between training and
deployment.

This chapter is a worked software-architecture case study. The goal is not to
memorize filenames; it is to learn how a mathematical reinforcement-learning
problem becomes versioned interfaces whose units, order, clocks, and owners can
be tested.

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
| MuJoCo Warp / Warp | NVIDIA Compute Unified Device Architecture (CUDA) simulation across many graphics processing units (GPUs) |
| `mjlab` | environment managers, task registry, vectorized wrapper, viewer, export support |
| Robotic Systems Lab reinforcement learning (RSL-RL) | Proximal Policy Optimization (PPO), neural networks, rollout buffers, checkpoints |
| Better Actuator Models (BAM) | voltage-aware Dynamixel XL330 actuator and friction behavior |
| this repository | Microduck robot models, tasks, rewards, randomization, tests, export and rehearsal |
| Microduck runtime repository | real sensors, command sources, Open Neural Network Exchange (ONNX) inference, motor communication |

MuJoCo's Extensible Markup Language (XML) model format (MJCF) is its
robot/model description format. CUDA is NVIDIA's GPU-computing platform. ONNX
is a portable neural-network file format.
**Inference** means running a frozen trained network to obtain an output;
unlike training, it does not update weights.

### Four graphs coexist

Beginners often see only the neural-network graph. A deployed learned
controller actually has four coupled graphs:

1. **physical graph**: bodies, joints, contacts, actuators, sensors;
2. **dataflow graph**: timestamped measurements to observation to action;
3. **learning graph**: rollout, returns, actor/critic losses, optimizer;
4. **artifact graph**: source revision, resolved configuration, checkpoint,
   normalizer, export, runtime build.

An edge mismatch in any graph can cause failure. For example, swapping two
joint observations preserves tensor shape and neural arithmetic but changes
the dataflow/physical mapping. Loading an actor without its normalizer preserves
weights but breaks the artifact graph.

This architecture descends from two modern scaling choices: vectorized physics
places many environment copies on a graphics processor, and a manager-based
task expresses observations/rewards/events as independently configured terms.
Alternative stacks may use one monolithic `step()` function, a differentiable
simulator, an off-policy replay service, or distributed actor processes. The
invariant questions remain: who owns state, when is it sampled, how is an
action transformed, and which artifact proves the answer?

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
into `mdp.py`. A Markov Decision Process (MDP) organizes state, action,
transition, reward, and discount; reading all of the large `mdp.py` module from
top to bottom is not a good beginner exercise.

The live upstream links make the theory-to-code mapping inspectable:

- [task registry](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/__init__.py)
  maps a task name to concrete factories and runner;
- [velocity configuration](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)
  specializes scene, commands, observations, reward, events, and PPO settings;
- [custom MDP functions](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/mdp.py)
  implement the project-specific mathematics; and
- [robot constants](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/robot/microduck_constants.py)
  bind models, HOME pose, collisions, and BAM actuators.

Pin the commit used by a run. A link to `main` is a navigation aid, not
reproducibility evidence.

## 8.3 Task registration

Every user-facing task **identifier (ID)** binds four things:

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

Registration happens at Python import time. The factory call creates a concrete
configuration object; managers later copy/resolve its term configurations. This
has two consequences:

- changing source after a run does not retroactively change its saved resolved
  configuration; and
- mutating `env.cfg` after manager initialization may not change the manager's
  own copied term. Runtime curricula must use manager accessors such as
  `get_term_cfg(...)`.

The object lifecycle is approximately:

```text
import task package
  -> execute registrations
  -> select task ID
  -> construct/copy environment config
  -> compile MuJoCo scene and resolve entity selectors
  -> construct managers from resolved term configs
  -> construct RSL-RL runner and networks
  -> collect rollout/update/save
```

A regular expression that matches the wrong joint can therefore fail during
resolution before a learning step. Central processing unit (CPU) configuration
tests deliberately force that early, cheap failure.

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

Manager composition is analogous to factoring an equation. If reward terms
$f_i$ and weights $w_i$ are registered independently, the manager evaluates

```math
r_t=\sum_i w_i f_i(s_t,a_t,s_{t+1};\eta_i),
```

where $\eta_i$ denotes term parameters such as standard deviation, command
name, sensor selector, or gate. The resolved configuration must record both
$w_i$ and $\eta_i$; a scalar total reward cannot reconstruct either.

The same principle applies to observations. Concatenation is an interface
function

```math
o_t=\mathrm{concat}(g_1(s_t),g_2(s_t),\ldots,g_k(s_t)).
```

Changing term order changes the mathematical function even when total
dimension remains 61.

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

The rate calculation is

```math
\Delta t_{policy}=d\,\Delta t_{physics}
=4(0.005)=0.020\ \mathrm{s},
```

```math
f_{policy}=\frac{1}{\Delta t_{policy}}=50\ \mathrm{Hz}.
```

Action holding is part of the transition kernel. A policy trained at 50 Hz and
executed at 100 Hz does not merely react faster: it applies twice as many
targets over the same physical time and receives differently spaced state
changes.

### Simulation time, wall time, and data age

One default rollout contains

```math
4096\ \text{environments}\times24\ \text{steps}
=98{,}304\ \text{transitions}.
```

Each environment advances only $24/50=0.48$ simulated seconds, while the
aggregate batch represents about $4096(0.48)=1966$ simulated robot-seconds.
Parallelism increases throughput; it does not make any one trajectory see a
longer future.

Deployment adds an **age of information** budget. If an inertial measurement
was sampled at $t_s$, assembled into an observation at $t_o$, inference ends at
$t_i$, and a servo consumes the command at $t_a$, then its effective age at
actuation is $t_a-t_s$. Queueing every stale sensor packet can make this age
grow even when no packet is lost. A low-level control path normally wants a
bounded latest-value queue, timestamps, and explicit stale-data fallback.

Training delay/noise must mimic this causal order. Adding independent noise to
an already current vector is not the same as sampling heterogeneous sensor
timestamps, holding an old action, or delaying a bus command.

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
- inertial measurement unit (IMU) mounting variation;
- delayed angular velocity, gravity, joint velocity, and actuation;
- backlash-specific encoder views.

If the actor sees a biased or backlash-adjusted quantity, a tracking reward on
that quantity must use the same view. Otherwise the policy is punished for
correcting the measurement it receives.

The actual network input is normalized featurewise:

```math
\tilde o_{t,i}=\frac{o_{t,i}-\mu_i}
{\sqrt{\sigma_i^2+\epsilon}}.
```

Thus the actor contract is not only “61 floats.” It is the ordered tuple

```text
(meaning, unit, frame, timestamp/age, corruption, mean, variance)
```

for every feature. A useful deployment test freezes several physically named
states and compares the complete normalized 61-vector between training-side
and runtime encoders before comparing actions.

Previous action occupies 14 dimensions because commanded target history helps
the actor infer actuator/delay state. It is not a complete memory: hidden
temperature, payload, friction, and older contact history may still make the
observation partially Markov. Alternatives include stacking more history, a
recurrent actor, or a dedicated adaptation estimator; each increases training
and deployment state that must be synchronized.

## 8.7 Action contract

The actor emits 14 floating-point values. The joint-position action uses
`scale=1.0` and the model's default joint pose as its offset. Conceptually:

```math
q^{target} = q^{HOME} + a
```

The exact action manager also applies its configured ordering and any limits.
The target then enters the BAM actuator model; it is not an ideal instantaneous
joint angle.

The units expose why `scale=1.0` is not “no transform.” A normalized actor
number of $0.1$ becomes a $0.1$ radian target offset, about $5.73^\circ$:

```math
0.1\ \mathrm{rad}\times\frac{180^\circ}{\pi}
\approx5.73^\circ.
```

Action order is a matrix permutation. If runtime servo order differs, the
physical target is

```math
q^{runtime}=q^{HOME}+Pa,
```

where permutation matrix $P$ should be identity for the documented contract.
A shape check cannot detect $P\neq I$; a one-hot action probe can.

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

### Network size and rollout memory are calculable

For the actor mean path `61 → 512 → 256 → 128 → 14`, including a bias per
output neuron, the learned scalar count is

```math
(61\times512+512)
+(512\times256+256)
+(256\times128+128)
+(128\times14+14)
=197{,}774.
```

A learned 14-value log-standard-deviation vector brings the training actor to
about 197,788 scalars, depending on the exact RSL-RL distribution
implementation. At 32-bit floating point, the mean network weights occupy
about 0.75 mebibytes (MiB) before file-format metadata and normalizer state. This explains
why a small feed-forward actor can be plausible on a robot central processing
unit (CPU), but measured worst-case latency still decides.

Rollout storage is larger because it scales with environments and time. Actor
observations alone require approximately

```math
4096\times24\times61\times4
=23{,}986{,}176\ \text{bytes}
\approx22.9\ \text{MiB}.
```

Critic observations, actions, rewards, values, log-probabilities, advantages,
masks, optimizer activations, simulator state, and gradients add substantially
more. This is why “model size” and “training memory” are different budgets.

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

Formally, let physical parameters be $\xi$—voltage, delay, friction, mass,
center of mass, encoder bias, and so on. Domain-randomized training optimizes

```math
J(\theta)=
\mathbb{E}_{\xi\sim p_{train}(\xi),\,\tau\sim\pi_\theta,P_\xi}
[G(\tau)].
```

The distribution $p_{train}$ is part of the task. If real parameters lie
outside its support, the expectation provides no coverage argument. If it is
unphysically broad, one actor may become unnecessarily conservative or unable
to distinguish contradictory dynamics.

At a 5 ms actuator update, a lag of three to six update slots corresponds to a
nominal 15–30 ms command delay **if** the active actuator implementation defines
lag in physics-update slots. Verify that dependency version rather than infer
units from the field name. A configuration range without its update clock is
not a physical specification.

Randomization must restore nominal values before sampling each episode. An
additive center-of-mass update accidentally applied to the already randomized
value performs a random walk over resets; eventually its distribution bears no
relationship to the stated range.

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

The resolved YAML Ain't Markup Language (YAML) file is valuable. Source code
may change after a run; the saved parameters show what the checkpoint actually
trained against. Keep it with any policy you evaluate or deploy.

A minimal policy manifest should bind:

```yaml
task_id: Mjlab-Velocity-Flat-MicroDuck
source_commit: <git commit>
lockfile_sha256: <digest>
env_config_sha256: <digest>
agent_config_sha256: <digest>
checkpoint_sha256: <digest>
onnx_sha256: <digest>
observation_schema: microduck-policy-family-61d-v1
policy_period_s: 0.020
evaluator_commit: <git commit>
acceptance_report: <path or immutable URL>
```

Secure Hash Algorithm 256-bit (SHA-256) digests detect accidental artifact
substitution; they do not prove the artifact is good. The acceptance report
supplies behavioral evidence, and a trusted release/signing process supplies
authenticity.

The provenance chain should be traversable in both directions:

```text
hardware policy -> ONNX digest -> checkpoint -> resolved config/source
training run     -> checkpoint -> export command -> ONNX -> acceptance report
```

Without this chain, “the trained model” is ambiguous after several 20-hour
runs.

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
6. Calculate policy period/rate and aggregate simulated seconds for a 24-step,
   4,096-environment rollout.
7. Compute the actor mean-network parameter count independently. Which extra
   training parameter and export state are not in that sum?
8. Design a one-hot action-order test that detects a servo permutation while
   the robot is safely unpowered or simulated.
9. For one observation feature, write its complete contract: meaning, unit,
   frame, source timestamp, corruption, and normalization.
10. Explain why modifying `env.cfg` after manager construction can be a silent
    no-op and identify the correct curriculum access path.
11. Draw the provenance path from a deployed ONNX file back to source and
    forward to its acceptance evidence.
12. Compare a manager-based and monolithic environment. Give one advantage and
    one failure mode of each without declaring one universally superior.

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
4. Actor and critic both use hidden dimensions `(512, 256, 128)` with an
   exponential linear unit (ELU) activation and observation normalization.
   Their inputs differ because the critic may receive privileged observations.
5. Real ONNX inference deploys the actor and its actor-observation normalizer.
   The critic, reward/termination/event/curriculum managers, rollout storage,
   PPO losses, optimizer, and simulator randomization exist only to generate or
   improve the checkpoint.
6. $4(5\text{ ms})=20$ ms and $1/0.020=50$ Hz. Each environment advances
   $24/50=0.48$ s; aggregate experience is $4096(0.48)=1966.08$ simulated
   robot-seconds, or about 32.77 robot-minutes.
7. Sum

   ```math
   (61\times512+512)+(512\times256+256)
   +(256\times128+128)+(128\times14+14)=197{,}774.
   ```

   A typical diagonal Gaussian
   adds 14 learned log-standard-deviation values during training. The deployed
   graph also needs actor normalizer statistics/logic, though those are not
   dense-layer weights.
8. In simulation or with torque disabled, set action vector to zero, then set
   exactly component $j$ to a small safe value. Inspect the post-transform
   physical target array and assert only documented servo $j$ changes by the
   expected radians. Repeat for all 14. Do not infer order by moving powered
   hardware first.
9. Example: projected-gravity x component; dimensionless unit vector component;
   body frame; derived from the inertial estimate sampled at timestamp $t_s$;
   delayed/noised by the configured term; normalized by saved feature mean
   $\mu_i$, variance $\sigma_i^2$, and epsilon. The same template should be
   filled with actual evidence for every feature.
10. Managers deepcopy/resolve their term configurations at initialization, so
    later writes to the outer environment config need not touch the live term.
    A curriculum should read/update the manager's live term through an accessor
    such as `env.event_manager.get_term_cfg(...)`, then set it through the
    supported manager path.
11. ONNX digest → export record → checkpoint digest → saved agent/environment
    YAML digests → lockfile/source commit → task/model sources. In the other
    direction, ONNX digest → frozen evaluator/config → per-condition metrics,
    rollouts, reviewer decision, and hardware stage authorization.
12. Managers make terms independently configurable/loggable/testable and reuse
    templates, but copied/resolved configuration can surprise runtime mutation
    and hidden term interactions remain possible. A monolithic `step()` makes
    causal order locally visible and may be easier for a tiny environment, but
    reward/observation reuse, per-term logging, and systematic configuration can
    become tangled. Evidence and project scale decide.

A repeatable source trace is:

```bash
rg -n 'Mjlab-Velocity-Flat-MicroDuck' src/mjlab_microduck/tasks/__init__.py
rg -n 'MicroduckRlCfg|hidden_dims|obs_normalization' \
  src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py
```

</details>
