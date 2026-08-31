# 10. Anatomy of a Microduck Environment

An environment is an executable task specification. It defines what the robot
can observe and do, how the world changes, what counts as progress, and which
experience Proximal Policy Optimization (PPO) collects.

This chapter uses `Mjlab-Velocity-Flat-MicroDuck` as the main worked example.

In mathematical language, an environment implements a Markov Decision Process
(MDP): state space, action space, transition law, reward, initial-state
distribution, discount, and termination rules. In engineering language, it is
also an interface contract among simulator, learner, exporter, and runtime.
Those views line up as follows:

| MDP object | Microduck implementation | Question to audit |
| --- | --- | --- |
| hidden state $s_t$ | MuJoCo bodies, joints, actuator and sensor state | What affects the future? |
| observation $o_t$ | ordered actor/critic observation terms | What may each network know? |
| action $a_t$ | 14 joint-position offsets | What can the policy command? |
| transition $p$ | physics, Better Actuator Models (BAM), delays, events, randomization | What dynamics generate the next state? |
| reward $r_t$ | weighted reward manager terms | What behavior is optimized? |
| initial distribution $\rho_0$ | reset events and command sampling | Where does experience begin? |
| terminal flags | timeout, fall, bounds, nonfinite state | When does return stop or bootstrap? |

The simulator state is richer than the actor observation, so the deployed
problem is more precisely a Partially Observable Markov Decision Process
(POMDP). Previous action, angular velocity, projected gravity, and joint state
form a compact *information state*: not a perfect reconstruction of physics,
but enough history and sensing for a useful reactive controller.

The environment is therefore a hypothesis about sufficiency. If obstacle
distance is absent, no optimizer can make the actor condition its first step
on an unseen obstacle. If motor temperature changes future torque but is
neither observed nor inferable from history, the actor must learn a robust
compromise rather than temperature-specific control.

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

The live sources for this walkthrough are the
[`velocity` configuration](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py),
the [custom MDP functions](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/mdp.py),
and the [task registry](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/__init__.py).
Follow references between them: the configuration names *which* function and
parameters are active, while `mdp.py` defines the tensor calculation.

Starting from the closest maintained template is software reuse with a
scientific benefit. Closely related tasks inherit the same observation noise,
delay, termination, and actuator assumptions. Copy-pasted standalone
environments tend to drift invisibly, turning an intended reward comparison
into a physics comparison.

## 10.2 Scene: the physical question

The scene selects:

- the robot's MuJoCo Extensible Markup Language (XML) model format (MJCF) file
  and its initial state;
- terrain;
- collision rules;
- contact and ray sensors;
- environment count and spacing; and
- viewer camera.

Flat velocity uses a plane. Rough velocity uses a generator with robot-scale
terrain; step heights are capped around 1.5 cm because carrying human-scale
terrain assumptions to a 25 cm robot would create a different task.

“Robot-scale” has a physical meaning. For motions dominated by gravity, the
dimensionless Froude number compares inertial speed with gravitational speed:

```math
\mathrm{Fr}=\frac{v^2}{gL},
```

where $v$ is forward speed, $g$ is gravitational acceleration, and $L$ is a
representative leg length. Two geometrically similar robots with comparable
$\mathrm{Fr}$ have speed scaling approximately as $v\mathrel{\propto}\sqrt L$,
not directly with height. Terrain should likewise be expressed relative to
foot size, clearance, and leg length. A 1.5 cm step is 6% of a 25 cm robot's
height; the same relative obstacle for a 1.7 m human would be about 10 cm.

Dynamic similarity is only a starting point: motor torque density, foot
geometry, joint range, controller frequency, and contact compliance do not all
scale geometrically. The practical lesson is to measure the actual robot and
state normalized terrain ratios rather than borrowing a benchmark's absolute
meters.

The walking model has deliberately simplified fall contacts. Tasks that begin
or finish on the ground use the all-collisions model. Model choice is part of
the task definition, not a rendering preference.

Collision geometry and visual geometry serve different jobs. Visual meshes
make videos recognizable; collision primitives define the contact problem the
physics engine must solve. A foot that looks flat but has a rounded or
misaligned collision shape changes the support polygon and reward-relevant
contacts. Inspect both.

Contact capacity and solver iterations are computational parameters with
behavioral effects. The rough task raises the contact limit and solver work
because a fallen robot touching several terrain boxes can overflow a walking
configuration sized for two feet on a plane. Dropped or poorly converged
contacts can appear as explosive motion or nonfinite state; that is a physics
model failure, not exploration.

### Try it: compare resolved scenes

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck --agent zero --num-envs 1
uv run play Mjlab-Velocity-Rough-MicroDuck --agent zero --num-envs 1
```

A zero agent is useful for inspecting physics and initial conditions without
attributing motion to a policy.

For a target pose, extend the test: hold its corresponding joint targets for
several seconds from perturbed resets and record trunk height *and tilt*.
Settling at the expected height while lying sideways is not equilibrium
validation. A pose-based task whose goal is not mechanically supportable asks
learning to fight gravity forever.

## 10.3 Actions: the controllable question

The velocity task has one action term with dimension 14:

```python
joint_pos_action = cfg.actions["joint_pos"]
joint_pos_action.scale = 1.0
```

The action manager interprets each network output as an offset from the default
joint pose. Better Actuator Models (BAM) then turns the position target into
voltage/torque behavior.

The conceptual map for servo $j$ is

```math
q^{target}_{t,j}=q^{HOME}_j+s_j a_{t,j},
```

where $s_j=1$ radian per policy unit in this task. The target is not the next
joint angle. The actuator's feedback law, voltage limit, motor constants,
friction, load, and physics determine how much motion occurs during the next
20 milliseconds. This separation is fundamental:

```text
policy action -> target -> actuator dynamics -> torque -> acceleration -> state
```

For a simplified proportional–derivative servo,

```math
\tau^{cmd}_{t,j}
=K_{p,j}(q^{target}_{t,j}-q_{t,j})-K_{d,j}\dot q_{t,j},
```

followed by motor-voltage/torque saturation. The real BAM path is richer, but
the equation explains why a large target offset may produce a saturated motor
rather than an instantaneous large pose change.

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

The alternatives embody different assumptions:

| Policy output | Benefit | Main burden |
| --- | --- | --- |
| joint position target | compatible with smart servos; bounded posture intent | inner-loop dynamics are hidden from policy |
| joint velocity target | direct motion intent | needs matched velocity loop and limits |
| torque/current | maximum dynamic authority | model fidelity and hardware safety become harder |
| residual over classical target | preserves a known controller | baseline controller constrains discoverable behavior |

Action ordering is as important as dimension. A 14-value vector can have the
right shape and command the wrong legs. A deployment conformance test applies
one small nonzero component at a time in simulation and on an unloaded or
otherwise safe actuator fixture, then verifies the named joint, sign, and
units. This tests the permutation that shape checks cannot see.

The policy is stochastic during PPO training. If its Gaussian sample is later
clipped to a legal interval, all samples beyond the bound produce the same
command. Extensive saturation therefore wastes probability mass and hides the
size of the requested change. Log both pre-limit action statistics and applied
targets; a policy that lives on limits is often exploiting an overly wide
interface or compensating for insufficient actuator authority.

## 10.4 Observations: the information question

An observation term is a function plus optional noise, scale, clipping, delay,
and history. Actor terms are concatenated in order.

The main actor gets 48 proprioceptive values plus a 13D command block. The
critic receives additional simulator-only signals. Two principles matter:

1. **Availability:** actor inputs must be reproducible on hardware.
2. **Meaning:** training and runtime must use the same units, frame, sign,
   ordering, offset, delay, and filtering.

Formally, the actor receives a possibly corrupted measurement function

```math
o_t=h(s_t,\epsilon_t,\tau_t),
```

where $\epsilon_t$ represents sensor noise/bias and $\tau_t$ represents sample
age or delay. The policy computes $a_t\sim\pi_\theta(\cdot\mid o_t)$, not from
the simulator state $s_t$ directly. Treating $o_t=s_t$ when designing the
runtime is a common source of sim-to-real failure.

The ordered actor vector is:

```text
3 angular velocity + 3 projected gravity
+ 14 relative joint positions + 14 joint velocities
+ 14 previous actions
+ 3 twist command + 4 head command + 6 body command
= 61 values
```

“Projected gravity” means the world gravity direction expressed in the robot
body frame. It gives roll/pitch orientation without exposing a globally
meaningful yaw. Relative joint position means measured joint angle minus the
documented HOME angle. Previous action helps the memoryless network infer part
of the actuator/delay state.

The construction can be viewed as typed concatenation:

```python
parts = [
    angular_velocity_body_rad_s(),       # shape (N, 3)
    projected_gravity_body(),            # shape (N, 3)
    relative_servo_positions_rad(),       # shape (N, 14)
    servo_velocities_rad_s(),             # shape (N, 14)
    previous_policy_action(),             # shape (N, 14)
    command_twist_head_body(),            # shape (N, 13)
]
observation = concatenate(parts, axis=-1)
assert observation.shape == (num_envs, 61)
assert isfinite(observation).all()
```

Comments supply human meaning; automated tests must lock shape, ordering,
values at named probe states, and the featurewise normalizer. Otherwise two
programs can agree on 61 while disagreeing on every joint.

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

Using simulator-only variables for the critic is called an asymmetric
actor–critic. The critic can use true velocity or contact facts to estimate
return more accurately during training, while the actor remains deployable.
At inference the critic is discarded. Privilege must not leak into actor
features, command generation, or a runtime preprocessing branch.

Observation normalization changes the function represented by saved weights:

```math
\tilde o_i=\frac{o_i-\mu_i}{\sqrt{v_i+\epsilon}}.
```

The actor consumes $\tilde o$, so exporting only network matrices but not
$(\mu,v,\epsilon)$ exports a different policy. The mandatory Microduck export
path bakes this transform into the Open Neural Network Exchange (ONNX) graph.

For each feature, write a contract row containing semantic name, unit, frame,
sign, order, source sensor, calibration, expected range, timestamp, delay,
noise model, and fallback. That row is more useful than a bare neural-network
index during hardware integration.

## 10.5 Commands: the intention question

Commands are targets sampled by the training environment and supplied by an
operator or planner at deployment.

This makes a commanded policy a conditional controller:

```math
a_t\sim\pi_\theta(a\mid o^{robot}_t,c_t),
```

where $c_t$ is intention and $o^{robot}_t$ is measured robot state. A planner
chooses *what* local motion is desired; the locomotion policy chooses *how* to
realize it under contacts and disturbances. Confusing a command with a sensor
would make the architecture circular: requested forward speed does not reveal
actual forward speed or an obstacle.

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

Why are buckets mathematically necessary? A continuously uniform random
variable takes any exact value with probability zero. Thus

```math
\Pr(v_x=0, v_y=0, \omega_z=0)=0
```

under independent continuous sampling, even though exact zero is the most
important deployed idle command. Near-zero is also rare. With tolerance
$\epsilon=0.02$, the chance that independently sampled translation satisfies
$|v_x|<\epsilon$ and $|v_y|<\epsilon$ is

```math
\frac{0.04}{0.8}\frac{0.04}{0.6}\approx0.0033,
```

only about 0.33%, before requiring a useful yaw rate. Data imbalance, not an
optimizer defect, explains why spontaneous turn-in-place training can fail.

The implemented distribution is a mixture:

```math
p(c)=p_{stand}p_{stand}(c)
+p_{turn}p_{turn}(c)
+p_{general}p_{general}(c),
```

with probabilities summing to one. This converts capability priorities into
experience frequency. It also means evaluation should report each bucket
separately; an overall mean can hide complete failure on a small but important
slice.

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

A nonzero command input with zero reward can keep network connections from
remaining exactly unused, but it does not create a reason to obey that input.
Capability requires the complete causal chain:

```text
command varies -> actor observes it -> outcomes depend on action
-> reward distinguishes correct response -> training samples the region
-> evaluation tests it
```

At deployment, command timing joins the contract. The local twist should have a
timestamp, validity duration, rate/acceleration limits, and stale-command
fallback. A cloud planner that pauses cannot be allowed to leave an old forward
command active indefinitely; the real-time layer should decay or replace it
with the trained exact-zero command.

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

For configured feature functions $f_i$ and weights $w_i$, the environment
returns

```math
r_t=\sum_{i=1}^{K}w_i f_i(s_t,a_t,s_{t+1},c_t).
```

The learner optimizes expected discounted sums of $r_t$; it never reads the
English purpose column. Every claim such as “discourage slip” is true only if
the function, sign, units, contact selector, and gate implement it.

Define the empirical reward mass of term $i$ over an evaluation set of $T$
steps as

```math
M_i=\frac{1}{T}\sum_{t=1}^{T}w_i f_{i,t}.
```

$M_i$ is expressed in reward per step and includes actual behavior. A weight of
`-1.0` on a function usually near zero can be weaker than `-0.05` on a large
function active every step. Compare $M_i$, its distribution across command
slices, and behavior videos before copying a weight to another task.

Reward terms also interact. If positive tracking terms sum to roughly eight
units on a good step, a `-0.1` smoothness term has a different relative price
than it would in a task whose positive stack sums to two. This is why “standard
regularizer weights” are not portable constants.

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

For one scalar error $e$, the reward and its derivative are

```math
g(e)=\exp\left(-\frac{e^2}{\sigma^2}\right),
\qquad
\frac{dg}{de}=-\frac{2e}{\sigma^2}
\exp\left(-\frac{e^2}{\sigma^2}\right).
```

The gradient is zero at perfect tracking, grows for small errors, reaches its
largest magnitude at $|e|=\sigma/\sqrt2$, then decays toward zero far from the
target. Therefore $\sigma$ describes the error region where learning signal is
concentrated. It is not merely a cosmetic tolerance.

Suppose $\sigma=0.5$ radian. At $e=0.1$, $g\approx0.961$; at $e=0.5$,
$g\approx0.368$; at $e=1.0$, $g\approx0.018$. Tightening to $0.1$ makes the
same 0.5-radian error score $e^{-25}$, effectively removing its gradient from
early training. A curriculum can widen reachable commands only while the
current policy remains inside a gradient-bearing region.

The implementation reads through simulated backlash so the tracking reward and
encoder observation describe the same physical angle.

### Separate avoidable bias from necessary motion

Microduck's head is a large fraction of the robot's mass, so walking naturally
causes oscillatory head error. A tight instantaneous penalty priced that
necessary motion so heavily that standing still became a better solution. The
remedy was not “more PPO”; it was to penalize only the slow bias component.

An exponential moving average (EMA) of error obeys

```math
m_t=(1-\alpha)m_{t-1}+\alpha e_t,
\qquad
\alpha=\frac{\Delta t}{\tau+\Delta t}.
```

With time constant $\tau=1$ second and policy interval $\Delta t=0.02$ second,
$\alpha\approx0.0196$. Alternating zero-mean gait oscillation largely cancels,
while a persistent downward droop remains in $m_t$ and can be penalized by
$-|m_t|$. This is a theory-to-code example of asking which part of an error is
actually controllable.

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

Write a gate explicitly as $z_t\in\{0,1\}$:

```math
r^{air}_t=z^{walking}_t f_{air}(s_t),
\qquad
z^{walking}_t=\mathbf{1}[\lVert c^{twist}_t\rVert>\epsilon].
```

Without the gate, stepping in place could earn air-time reward during a stand
command. With a poorly chosen gate that pays only while fallen, remaining
fallen may become profitable. For recovery progress, a potential difference is
safer:

```math
r^{progress}_t=\gamma\Phi(s_{t+1})-\Phi(s_t).
```

Holding one bad state then pays approximately zero instead of a jackpot each
step. Chapter 1 showed why discounted potential shaping preserves the optimal
policy under its assumptions; here the engineering job is to choose a bounded,
measurable potential such as uprightness.

### Reward-design unit tests

Reward code deserves tests at named states, not only whole training runs. For
each term, construct at least:

1. a perfect state and a clearly worse state;
2. gate-open and gate-closed states;
3. left/right mirrored states when symmetry is intended;
4. values at physical limits; and
5. a nonfinite input for terms on risky sensor paths.

Then assert direction and sign rather than overfitting to one floating-point
constant:

```python
assert tracking(perfect) > tracking(offset)
assert air_time(standing_command) == 0.0
assert weighted_penalty(violation) <= 0.0
assert isfinite(reward(extreme_but_valid_state))
```

## 10.7 Events and domain randomization: the variation question

Event terms run in modes:

| Mode | Example |
| --- | --- |
| startup | foot friction, encoder bias fields, mass/inertia setup |
| reset | root pose, joints, center of mass (CoM), actuator friction scale, armature |
| interval | velocity pushes every few seconds |

Domain randomization (DR) trains one policy over a distribution of plausible
robots:

```math
\theta_{physics} \sim p(\theta)
```

More completely, training maximizes an average over physical parameters,
sensor corruption, commands, and initial state:

```math
J(\pi)=
\mathbb{E}_{\theta,\epsilon,c,s_0}
\left[\sum_{t=0}^{H-1}\gamma^t
r(s_t,a_t,s_{t+1},c;\theta)\right].
```

This objective does not guarantee success for every sampled robot. A broad
distribution can lower average performance or produce a conservative policy;
a narrow or biased distribution may omit the real robot. Choose ranges from
measurement, datasheets, system identification, calibration residuals, and
known operating conditions. Label guessed ranges as hypotheses to be updated.

Distribution shape also encodes belief. A uniform range says every interior
value is equally plausible and hard bounds are real. A truncated normal says
central values are more plausible. A discrete mixture can represent known
hardware revisions. Correlations matter: payload mass and center-of-mass shift
should not always be sampled independently if one physical payload causes
both.

Microduck randomizes selected quantities such as friction, mass/inertia, CoM,
battery voltage/sag, armature, encoder bias, inertial measurement unit (IMU)
alignment, delays, and pushes.

DR is not a substitute for calibration. Zero-centered IMU orientation
variation teaches tolerance to uncertain mounting magnitude; it cannot remove
a fixed systematic mounting bias. The runtime must calibrate that bias.

Custom randomization must restore defaults before applying a sampled change.
If a +5% mass change is added to the already-randomized mass every reset, the
distribution drifts outside the intended range during long training.

The accumulation bug is visible algebraically. If the intended rule is

```math
m_k=m_0(1+u_k),\qquad u_k\sim\mathcal U[-0.05,0.05],
```

then every reset remains within 5% of $m_0$. The wrong recursive rule

```math
m_k=m_{k-1}(1+u_k)
```

forms a multiplicative random walk whose spread grows with reset count. A test
should perform hundreds of resets and assert all realized values remain inside
the configured support, then verify that the distribution is not collapsed to
one value.

Randomization order should mimic the physical causal path. Encoder mounting
bias changes a measurement, not the true joint. Battery sag changes available
actuator voltage, not the desired command. Command latency holds an old target;
it is not interchangeable with independent noise added to the current target.
Placing uncertainty at the wrong layer may yield a robust policy for a machine
that cannot exist.

## 10.8 Terminations: the data-recycling question

The velocity task can terminate for timeout, falling, terrain bounds, or a
nonfinite state. A **not-a-number (NaN) guard** checks joints, root state, and
named sensor data:

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

Keep two meanings distinct:

- **termination**: the task's future has ended, such as an unrecoverable fall;
- **truncation**: collection stopped at a time or spatial boundary even though
  a valid continuation conceptually exists.

For a truncated transition, the return target normally bootstraps from the
critic:

```math
y_t=r_t+\gamma V(o_{t+1}).
```

For a truly terminal transition, future value is zero:

```math
y_t=r_t.
```

Treating every timeout as terminal teaches an artificial value cliff at the
horizon. Treating catastrophic nonfinite physics as a normal bootstrap target
can inject meaningless values. Inspect how the environment and learner pass
these flags rather than assuming the words are interchangeable.

Termination also changes the data distribution. If falling resets
immediately, the learner sees many reset states and little self-recovery. That
is appropriate for a walking-only controller intended to switch to a separate
recovery policy, but wrong for a single policy expected to stand up.

## 10.9 Curricula: the pacing question

A curriculum changes the problem as training progresses. In this repository,
steps mean environment steps:

```text
curriculum step = PPO iteration * num_steps_per_env
                = PPO iteration * 24
```

This counter is *per environment rollout depth*, not the total number of
parallel transitions. Increasing from 64 to 4,096 environments increases data
per iteration, but stage 1,000 still begins after 1,000 PPO iterations unless
the configuration says otherwise. Wall-clock arrival may change because
throughput changes.

A scheduled curriculum creates a sequence of data distributions
$p_0,p_1,\ldots$ and objectives $r_0,r_1,\ldots$. Learning is therefore
nonstationary: the policy is chasing a moving problem. The purpose is to keep
useful experience and gradients available, not to make a plot look smoothly
difficult.

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

Three common curriculum axes solve different shortages:

| Curriculum | Changes | Use when |
| --- | --- | --- |
| command range/mix | requested behaviors | rare or difficult commands lack data |
| randomization range | physical uncertainty | nominal skill exists before robustness |
| reward/regularization weight | optimization priorities | smoothness would suppress early exploration |

A fourth method, reverse curriculum, changes reset states. If a maneuver learns
the beginning but never encounters successful endings, initialize some worlds
near the end, then move the reset frontier backward. This supplies on-policy
experience in the missing region without dictating a full waypoint trajectory.

Iteration schedules are simple and reproducible, but competence-triggered
schedules can be safer conceptually: widen only after a held-out success metric
exceeds a threshold for several evaluations. They require hysteresis and logged
state so noise cannot flip difficulty repeatedly. Whichever method is used,
record the active stage beside every metric.

### Trace one stage all the way to behavior

At iteration 1,000, `step=1000*24=24,000`. The action-rate weight becomes
`-0.6`, the standing fraction becomes `0.15`, and the head command range also
widens. Because several changes coincide, a metric discontinuity cannot be
attributed to one of them without an ablation. Staggered boundaries improve
causal diagnosis; synchronized boundaries simplify an intended phase change.
This is an experimental-design tradeoff, not a formatting detail.

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

The changing arrows seen in a default viewer can look like a random walk, but
that description is misleading. The environment randomly resamples desired
local velocities every 3–8 seconds; PPO trains a *conditional velocity
tracker*. Randomness supplies a training distribution. The learned intention
interface is twist plus four head targets. It does not include a destination,
path, obstacle map, or active body-pose objective in this checkpoint.

A capability claim needs five linked facts:

| Link | Velocity tracking example | Obstacle avoidance requirement |
| --- | --- | --- |
| representation | twist command in 61D actor input | range/depth/local-map feature or safe planner command |
| variation | sampled velocity commands | varied obstacle geometry and relative motion |
| consequence | actions change measured velocity | actions change clearance/collision outcome |
| objective | velocity tracking rewards | clearance/progress/collision/safety objective |
| evaluation | fixed command buckets | unseen layouts, near misses, collision and progress metrics |

If any link is absent, optimization cannot establish the capability.

Two sound extension architectures are:

1. a perception/planning system detects obstacles and sends safe local twist
   commands to the unchanged 61D locomotion policy; or
2. a deliberately versioned policy family receives exteroceptive features and
   is trained on obstacle-rich scenes with a new runtime contract.

The first preserves a small, fast, hot-swappable motor policy. The second may
learn tighter perception-action coupling but requires new training data,
network design, inference budgets, tests, and deployment plumbing.

For Microduck and JumpRover, the first architecture is the safer initial
handoff: perception estimates local obstacles and robot pose, a planner emits a
bounded timestamped twist, and the 50 hertz local policy tracks it. Cloud
reasoning can propose goals or plans, but a real-time local process must check
freshness and limits. A network outage must degrade to stop or a locally safe
behavior rather than freezing the last plan.

The second architecture becomes attractive when local geometric features must
couple tightly to foot placement, for example stepping over a narrow obstacle.
Then version the observation schema rather than silently inserting features
into the 61D vector, and train/evaluate the complete sensor-to-action latency.

## 10.11 Exercises: read configuration as a scientific claim

Use the live
[`microduck_velocity_env_cfg.py`](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)
and `mdp.py` rather than answering only from this chapter.

1. What command causes standing? Why is a near-zero uniform sample not an
   equivalent training case?
2. Why is turn-in-place sampled in its own bucket? Calculate the probability
   that uniform translation lies within $\pm0.02$ m/s on both axes.
3. Which head joints are commandable, and in what order?
4. Why does `body_pose` remain in the observation when its reward is zero?
   What capability can and cannot be claimed?
5. Which randomization changes the actuator rather than MuJoCo joint friction?
   Why would randomizing the wrong field silently fail?
6. At which PPO iteration does the action-rate weight first become `-0.6`?
   Name two other scheduled changes close enough to confound interpretation.
7. For $g(e)=\exp(-(e/0.5)^2)$, compute $g(0)$, $g(0.5)$, and $g(1.0)$.
   Explain why tightening the standard deviation can hurt early learning.
8. A helper returns `-abs(error)`. Which weight sign makes it a penalty? What
   runtime log sign should result?
9. Write pseudocode for a 500-reset test that catches accumulating mass
   randomization.
10. A 20-second horizon ends while the robot is upright and walking. Is that
    naturally a termination or truncation, and should the value target
    bootstrap?
11. Design a one-hot action-order conformance test. What error can it catch
    that a `(14,)` shape assertion cannot?
12. A reward term has weight `-0.05` and unweighted mean 8. A second has weight
    `-1.0` and unweighted mean 0.1. Compute their mean reward masses and compare
    their actual influence.
13. With $\Delta t=0.02$ second and EMA time constant $\tau=1$ second,
    calculate $\alpha$. What type of tracking error passes through this slow
    state?
14. List the minimum environment and runtime changes required for a policy to
    avoid obstacles directly rather than through planner-provided safe twists.
15. Run a zero-agent scene inspection. Separate facts it can verify from
    learned capabilities it cannot verify.
16. Select one reward term and trace: configuration entry → function → tensor
    inputs → gate → returned sign → configured weight → logged weighted mass.
    Write the trace as a seven-line audit record.

## 10.12 Folded solutions

<details>
<summary>Show solutions to Exercises 1–8</summary>

1. Standing is an **exact zero twist command** selected by the command term's
   `rel_standing_envs` bucket. A near-zero command still asks for motion and can
   activate walking gates; continuous sampling also gives exact zero
   probability zero.
2. Independent sampling makes zero translation plus meaningful yaw rare. The
   translation probability is
   $(0.04/0.8)(0.04/0.6)=1/300\approx0.0033$, or 0.33%. The explicit
   `rel_turn_in_place_envs` bucket provides a controlled fraction instead.
3. The 4D head command controls `neck_pitch`, `head_pitch`, `head_yaw`, and
   `head_roll`, in that order. The dimensions are deltas from HOME.
4. The body slot preserves the shared 61D hot-swap interface and exposes the
   same schema to task variants. The small changing inputs can avoid a wholly
   dead connection, but a zero tracking weight supplies no incentive to obey
   them. Schema compatibility may be claimed; reliable body-pose control may
   not.
5. `randomize_joint_friction` changes BAM's per-environment
   `friction_scale`. BAM computes actuator friction and overwrites/does not use
   MuJoCo's ordinary `dof_frictionloss` as a conventional actuator would, so a
   random number in that unused field changes no transition the policy sees.
6. The `-0.6` stage starts at `1000*24=24,000` environment steps, which the
   project maps to PPO iteration 1,000. Standing fraction becomes 0.15 there,
   and the head-command range widens there as well; either can contribute to a
   coincident metric change.
7. $g(0)=1$, $g(0.5)=e^{-1}\approx0.368$, and
   $g(1)=e^{-4}\approx0.0183$. A narrower standard deviation moves early,
   inaccurate states into the almost-zero tail, leaving little gradient. It
   can also overprice physically necessary oscillation.
8. The function is already nonpositive, so a **positive** weight preserves it
   as a penalty. A negative weight would double-negate it into positive reward.
   The weighted `Episode_Reward/<term>` contribution should be nonpositive.

</details>

<details>
<summary>Show solutions to Exercises 9–16</summary>

9. Save the compile-time/default mass $m_0$, reset 500 times, collect each
   realized mass, and assert every value lies in
   $[0.95m_0,1.05m_0]$. Also assert the sample standard deviation is nonzero.
   A compact sketch is:

   ```python
   samples = []
   for _ in range(500):
       env.reset()
       samples.append(read_mass())
   assert all(0.95 * m0 <= m <= 1.05 * m0 for m in samples)
   assert stdev(samples) > 0
   ```

   A restore-then-randomize implementation passes; a multiplicative random
   walk eventually escapes the interval with high probability.
10. A time limit interrupts a physically valid continuation, so it is naturally
    a truncation. The target should normally include
    $\gamma V(o_{t+1})$. A task-specific finite-horizon objective could define
    time as terminal, but that must be intentional and time remaining may then
    belong in the observation.
11. Start at HOME and apply a small positive offset to action index $j$, all
    other entries zero. Verify only the documented servo moves in the positive
    convention, then repeat for all 14 indices. This catches permutation and
    sign errors that preserve vector shape.
12. The masses are $-0.05(8)=-0.4$ and $-1(0.1)=-0.1$ reward per step. The
    first term has four times the observed mean influence despite its 20-times
    smaller absolute weight.
13. $\alpha=0.02/(1+0.02)\approx0.0196$. The EMA retains slow or persistent
    bias and attenuates fast, roughly zero-mean oscillation. It is therefore a
    stateful filter, not a new measurement.
14. Add a physically plausible obstacle sensor/model, a versioned actor
    feature and normalizer contract, obstacle-rich scene/reset distributions,
    rewards or constraints for clearance/collision and task progress, a model
    architecture with adequate temporal/spatial context, matching runtime
    preprocessing and latency, and evaluation on held-out layouts. A box in
    the viewer alone satisfies none of those causal links.
15. A zero actor can verify loading, gravity, collision shapes, passive-joint
    motion, reset states, action plumbing at zero, and numerical finiteness. It
    cannot verify tracking, gait quality, recovery, obstacle avoidance,
    robustness, or hardware transfer because no learned decisions are used.
16. One acceptable record for action rate is:

    ```text
    config: rewards[action_rate_l2]
    function: inherited action-rate L2 cost
    inputs: current and previous 14D applied policy actions
    gate: active each policy step
    return: nonnegative squared change
    weight: negative, scheduled -0.1 toward -1.0
    log: weighted episode contribution must remain <= 0
    ```

    A different term is correct if every link is tied to the live code and
    its sign is reasoned from the function's actual return.

</details>

Continue with
[training, evaluation, and debugging](11_training_evaluation_and_debugging.md).
