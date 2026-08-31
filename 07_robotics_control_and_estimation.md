# 7. Robot Dynamics, Control, and State Estimation

A reinforcement learning (RL) policy becomes a robot controller only through
mechanics, electronics, sensors, timing, and lower-level control loops. This
chapter supplies the robotics background needed to understand those
interfaces.

## 7.1 Configuration and degrees of freedom

A robot's **configuration** describes its pose. Joint coordinates are commonly
collected in a vector $q$; joint velocities are $\dot q$ and accelerations are
$\ddot q$.

A **degree of freedom** (DoF) is an independent coordinate needed to describe
motion. A single revolute hinge has one DoF: its angle. A free rigid body in 3D
has six: three position coordinates and three orientation coordinates.

Microduck has 14 actuated joint coordinates, but its simulated state also
includes the free base pose and velocity. “14-action policy” therefore does not
mean the full physical state has dimension 14.

## 7.2 Position, velocity, acceleration, force, and torque

- position tells where something is;
- velocity is the rate of position change;
- acceleration is the rate of velocity change;
- force changes linear momentum; and
- torque is the rotational counterpart of force.

For a revolute joint,

```math
\tau = I\alpha
```

is the simplest analogy: torque $\tau$ produces angular acceleration $\alpha$
according to rotational inertia $I$. A whole robot is coupled, so moving one
joint can accelerate many links.

## 7.3 The robot dynamics equation

A standard compact rigid-body equation is

```math
M(q)\ddot q+C(q,\dot q)\dot q+g(q)+f(\dot q)
=\tau+J(q)^TF_{ext}.
```

Read it term by term:

- $M(q)\ddot q$: inertia resisting acceleration;
- $C(q,\dot q)\dot q$: velocity-dependent Coriolis/centrifugal effects;
- $g(q)$: gravity torque;
- $f(\dot q)$: friction and other losses;
- $\tau$: actuator torque;
- $J(q)^TF_{ext}$: external/contact force mapped into joint torques.

You do not need to symbolically solve this equation to train Proximal Policy
Optimization (PPO). You do need to understand that mass, center of mass,
inertia, friction, contact, voltage, and delay change how the same command moves
the robot.

## 7.4 Coordinate frames

A vector is incomplete without a frame. “Velocity $[1,0,0]$” could mean world
east or robot-forward.

Common frames are:

- **world frame**: fixed to the environment;
- **base/body frame**: moves with the trunk;
- **sensor frame**: aligned with an inertial measurement unit (IMU) or camera
  mounting; and
- **joint frame**: defined by the robot model's joint axis.

A rotation matrix $R_{WB}$ can map a body-frame vector into world coordinates:

```math
v_W=R_{WB}v_B.
```

A reversed transform, wrong axis sign, or degrees/radians mismatch can look
like a failed policy even if the network is correct.

## 7.5 Orientation representations

Common orientation representations include:

- roll-pitch-yaw/Euler angles: intuitive but singular at some orientations;
- rotation matrices: 9 numbers with orthogonality constraints;
- quaternions: 4 numbers with unit-length constraint; and
- projected gravity: gravity direction expressed in the body frame.

Projected gravity is useful for locomotion because it tells the policy which
way is “down” without exposing a fragile Euler-angle parameterization. A
quaternion and its negation represent the same physical rotation, which must be
considered in errors and interpolation.

## 7.6 Contacts make robot dynamics hybrid

A walking robot alternates between continuous flight/swing motion and discrete
contact events. Impact can abruptly change velocity. Contact includes normal
force, friction, compliance, collision geometry, and numerical solver choices.

This is a **hybrid system**: continuous dynamics plus mode switches. Small
differences in foot geometry or timing can produce large rollout differences.
That is one reason long sim and real trajectories need not be bit-identical
even when the control interface matches.

### Coulomb friction intuition

A simple contact model limits tangential friction force:

```math
|F_t|\le\mu F_n,
```

where $F_n$ is normal force and $\mu$ is a friction coefficient. If requested
tangential force exceeds the limit, the foot slips. Real tires/feet also show
compliance, velocity dependence, surface variation, and wear.

## 7.7 Actuators are not ideal commands

An **actuator** converts electrical energy into mechanical motion or force.
Common robot actuators include direct-current and brushless direct-current
(BLDC) motors with transmissions and integrated servos.

An ideal simulator position actuator instantly producing any required torque
is physically impossible. Real behavior depends on:

- voltage and battery sag;
- motor torque/current limits;
- speed-torque relationship;
- gear ratio and efficiency;
- friction, backlash, and compliance;
- embedded-controller gains and rate;
- command/communication latency; and
- thermal protection.

### Better Actuator Models (BAM) in the Microduck project

**BAM** means **Better Actuator Models**, the actuator-modeling approach used
by the project for Dynamixel XL330 servos. Here it refers to a voltage-aware
servo model with its own friction behavior, not a generic RL algorithm.

That distinction has a concrete consequence: randomizing MuJoCo's ordinary
joint `dof_frictionloss` does not randomize the effective friction authority
when BAM computes it inside the actuator. The project scales BAM's
`friction_scale` instead.

## 7.8 Feedback control and proportional–derivative (PD) control

A feedback controller compares a target to a measurement. A proportional-
derivative (PD) joint controller is

```math
\tau=K_p(q_{target}-q)-K_d\dot q.
```

- proportional term pushes against position error;
- derivative term damps motion;
- $K_p$ and $K_d$ are gains.

High gains can track tightly but amplify noise, excite unmodeled dynamics, or
hit current limits. Low gains are compliant but may need target overshoot to
produce enough torque. Thus a policy's position target can intentionally lie
beyond the actual desired joint position while the physical joint remains
within limits.

## 7.9 Choosing the policy action

An RL action need not equal motor current. Common choices are:

| Action | Advantage | Risk/requirement |
| --- | --- | --- |
| torque/current | direct dynamic authority | hard transfer and safety; needs fast loop |
| joint velocity target | natural for wheels | requires matched velocity controller |
| absolute joint position | interpretable | can jump if not rate-limited |
| position delta from HOME | centered, normalized | target scale and HOME must match |
| residual on nominal controller | bounded learning scope | nominal/residual interaction |
| task-space target | meaningful geometry | needs inverse kinematics (IK) or a controller underneath |

Microduck uses 14 normalized actions that become joint-position targets around
the default pose. The lower-level actuator dynamics turn targets into motion.

## 7.10 Nested control loops

A safe robot usually has several rates and responsibilities:

```text
high-level planner / behavior selection        1–10 Hz
             |
local navigation / learned skill command      10–50 Hz
             |
RL locomotion policy                          50–200 Hz
             |
joint position/velocity control             100–1000 Hz
             |
current/field-oriented motor control          5–40 kHz
             |
mechanics and contacts                      continuous
```

These are illustrative ranges, not universal specifications. Each robot must
measure timing and stability requirements.

The fastest loop should not depend on a cloud round trip. A network outage must
not remove motor current limits, watchdogs, balance protection, or emergency
stop.

## 7.11 Sampling rate, latency, and delay

A 50 Hz policy period is

```math
T=\frac{1}{50}=0.02\text{ s}=20\text{ ms}.
```

If physics simulates at 5 ms and the action is held for four physics steps,
the policy sees a new observation every 20 ms. This hold count is called
**decimation** in many robotics simulators.

**Latency** is time between measurement/decision and effect. **Jitter** is
variation in that latency. A constant 10 ms delay and a delay varying from
0–20 ms can affect stability differently.

Deployment must define:

- when sensors are sampled;
- whether measurements have different timestamps;
- inference worst-case execution time (WCET), not only average time;
- when commands reach actuators; and
- what happens on deadline miss.

## 7.12 State estimation

The policy rarely observes the true state. A **state estimator** combines
sensor readings and a process model to estimate quantities such as orientation
or velocity.

Typical inputs:

- encoders: joint/motor position and sometimes velocity;
- IMU gyroscope: angular velocity;
- IMU accelerometer: specific force, including gravity effects;
- contact/force sensors;
- cameras, depth sensors, or light detection and ranging (LiDAR); and
- motor current/voltage/temperature.

An estimator may use complementary filtering, a Kalman-filter family,
optimization, or a learned model. Every estimate has bandwidth, bias, noise,
delay, and frame conventions.

### Observation is not state

An observation $o_t$ is what the policy receives. Simulator state $s_t$ may
also contain exact ground-truth velocity, contact forces, terrain parameters,
or future commands unavailable on hardware.

Using richer information for a training-only critic is called **asymmetric
actor-critic** or **privileged critic** training. It is valid only if the actor
input remains deployable.

## 7.13 System identification before randomization

**System identification** estimates model parameters from measured input-output
data. For an actuator:

1. command a safe excitation;
2. log target, encoder position/velocity, current, voltage, and timestamps;
3. fit delay, gain, friction, damping, or other parameters;
4. validate on a held-out motion; and
5. randomize around the residual uncertainty.

Randomization is not a substitute for calibration. A wildly wrong nominal
model plus a huge randomization range can teach unnecessarily conservative or
unphysical behavior.

## 7.14 Safety is a separate objective and mechanism

A reward penalty is a preference in expectation. It is not a hard guarantee.
Hardware safety needs independent mechanisms:

- mechanical stops and suitable structure;
- current, voltage, speed, position, and temperature limits;
- watchdog and command timeout;
- communication validity and sequence checks;
- bounded policy outputs and rate limits;
- fall/tilt detection;
- physical emergency stop;
- tether/test stand during early trials; and
- a known rollback controller/artifact.

Constrained RL can represent expected costs separately from reward. For
example, [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
optimizes return under expected constraints. Its theoretical/empirical
properties do not replace electrical or realtime interlocks.

## 7.15 Exercises

1. A policy uses 5 ms physics and decimation 4. Compute its rate.
2. Explain each term in the rigid-body equation in one sentence.
3. Why might a low-$K_p$ servo need a command outside the desired physical
   angle, and why must the real joint still have a safety limit?
4. Choose an action representation for a wheel and for a servo-height joint.
5. Draw the frames for a body IMU mounted rotated 90° about yaw. Which transform
   must the observation pipeline apply?
6. Give one constant bias, one random noise source, and one latency source on a
   real robot.
7. Explain why a reward penalty cannot serve as an emergency stop.
8. Design a system-identification trial that is informative but safe.

Continue with the [Microduck software and control architecture](08_microduck_software_and_control_architecture.md),
where these concepts become concrete interfaces.

## 7.16 Folded solutions

<details>
<summary>Show answers to Section 7.15</summary>

1. The policy period is $5\text{ ms}\times4=20\text{ ms}$, so its rate is
   $1/0.020=50$ Hz.
2. In $M(q)\ddot q+C(q,\dot q)\dot q+g(q)=\tau+J^Tf_{ext}$: $M$ maps joint
   acceleration to inertial torque; $C\dot q$ contains velocity-dependent
   Coriolis/centrifugal effects; $g$ is gravity torque; $\tau$ is commanded or
   actuator torque; and $J^Tf_{ext}$ maps external/contact forces into joint
   torque.
3. With low proportional gain, static load may leave a position error; a target
   beyond the desired angle can create enough torque to hold the actual angle.
   The physical joint still needs independent limits because target overshoot
   can otherwise request collision, overload, or hard-stop impact.
4. A wheel naturally accepts bounded velocity or torque/current target under a
   lower wheel loop. A height servo naturally accepts a bounded position target
   with rate/limit enforcement. The exact choice follows measured actuator and
   controller semantics.
5. Define sensor frame $S$ and body frame $B$. For a sensor mounted +90° about
   body yaw, apply the calibrated rotation $R_{BS}$ to every sensor-frame vector
   before constructing body-frame observations. Test known gravity and angular-
   rate directions; do not repair a fixed mounting transform with randomization.
6. Constant bias: gyro zero offset. Random noise: encoder quantization or IMU
   measurement noise. Latency: filtering plus bus/queue/inference delay.
7. A reward penalty changes expected optimization preference. It can be traded
   against positive reward, has approximation error, and acts only when the
   policy runs. An E-stop must independently and deterministically remove or
   bound actuator authority.
8. Clamp the robot in a fixture, begin at low voltage/current, and command a
   small multisine or step sequence inside accepted travel. Log command,
   encoder, current, voltage, load, and synchronized timestamps. Fit delay,
   gain, damping/friction on one sequence; validate on a different sequence;
   stop on current, temperature, position, velocity, or watchdog limits.

</details>
