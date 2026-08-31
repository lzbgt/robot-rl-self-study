# 7. Robot Dynamics, Control, and State Estimation

A reinforcement learning (RL) policy becomes a robot controller only through
mechanics, electronics, sensors, timing, and lower-level control loops. This
chapter supplies the robotics background needed to understand those
interfaces.

### Why learning does not replace robotics

Robot control grew from mechanics, feedback, estimation, numerical
optimization, and realtime systems. A learned policy enters an existing causal
chain: sensors estimate a delayed physical state; software computes a bounded
target; actuator electronics create torque; contacts feed forces back into the
mechanism. The policy can approximate a difficult control law, but it cannot
make frames, energy, latency, observability, or actuator limits disappear.

Modern alternatives are therefore usually compositional: inverse dynamics,
whole-body control, or Model Predictive Control (MPC) for explicit structure;
reinforcement learning for hard-to-model behavior; imitation for motion style;
learned residuals for missing physics; and control-barrier/supervisory logic for
constraints. “Classical versus learning” is rarely the most useful system
boundary.

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

### Forward kinematics and the geometric Jacobian

**Forward kinematics** maps joint configuration to a point or link pose:

```math
x=f(q).
```

For a small joint change $\Delta q$, first-order Taylor expansion gives

```math
f(q+\Delta q)\approx f(q)+J(q)\Delta q,
```

where

```math
J(q)=\frac{\partial f}{\partial q}
```

is the Jacobian. Dividing by a small time interval yields

```math
\dot x=J(q)\dot q.
```

The transpose maps an end-effector wrench $F$ into generalized joint torque:

```math
\tau_{ext}=J(q)^TF.
```

This follows from equal virtual work:

```math
F^T\delta x=F^TJ\delta q=(J^TF)^T\delta q.
```

The Jacobian can lose rank at a **singularity**, where some Cartesian direction
requires unbounded or unavailable joint velocity. A task-space RL action still
needs inverse kinematics or a controller that handles these geometric limits.

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

Mechanical power is

```math
P=F^Tv
\quad\text{or}\quad
P=\tau^T\dot q.
```

Energy is the integral of power over time. A torque may be within its static
limit yet demand excessive power at high speed; sustained current can also
overheat a motor. This is why action, torque, speed, voltage, and temperature
limits describe different safety constraints.

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

### Where the dynamics equation comes from

Let kinetic energy be $T(q,\dot q)$ and potential energy be $U(q)$. The
Lagrangian is $\mathcal{L}=T-U$. For each generalized coordinate $q_i$, the
Euler–Lagrange equation is

```math
\frac{d}{dt}\frac{\partial\mathcal{L}}{\partial\dot q_i}
-\frac{\partial\mathcal{L}}{\partial q_i}
=Q_i,
```

where $Q_i$ collects actuator and external generalized forces. Expanding these
equations for linked rigid bodies produces the inertia matrix, velocity terms,
and gravity vector. This derivation explains useful structure:

- $M(q)$ is symmetric and positive definite away from constrained/redundant
  coordinates;
- gravity comes from the gradient of potential energy;
- contact forces enter through $J^TF$ because of virtual work; and
- actuator/friction models add nonconservative generalized forces.

A simulator numerically integrates these accelerations. With semi-implicit
Euler for a simple coordinate:

```math
\dot q_{k+1}=\dot q_k+\Delta t\,\ddot q_k,
```

```math
q_{k+1}=q_k+\Delta t\,\dot q_{k+1}.
```

Changing timestep changes integration error and contact behavior even when the
continuous equation is unchanged.

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

A unit quaternion $q=[w,x,y,z]$ satisfies

```math
w^2+x^2+y^2+z^2=1.
```

For an axis-angle rotation with unit axis $u$ and angle $\theta$:

```math
q=\left[\cos\frac{\theta}{2},\ u\sin\frac{\theta}{2}\right].
```

Because $q$ and $-q$ encode the same rotation, a naive Euclidean error
$\lVert q-q^*\rVert$ can call identical orientations far apart. A sign-
invariant similarity uses $|q^Tq^*|$, and a shortest-path interpolation flips
one quaternion when the dot product is negative.

## 7.6 Contacts make robot dynamics hybrid

A walking robot alternates between continuous flight/swing motion and discrete
contact events. Impact can abruptly change velocity. Contact includes normal
force, friction, compliance, collision geometry, and numerical solver choices.

This is a **hybrid system**: continuous dynamics plus mode switches. Small
differences in foot geometry or timing can produce large rollout differences.
That is one reason long sim and real trajectories need not be bit-identical
even when the control interface matches.

An ideal unilateral contact has gap $\phi(q)\geq0$ and normal force
$\lambda_n\geq0$. They satisfy complementarity:

```math
\phi(q)\lambda_n=0.
```

Either bodies are separated and normal force is zero, or the gap is closed and
a nonnegative contact force may exist. Impacts add velocity-level rules and
friction adds tangential constraints. Engines approximate or regularize these
nonsmooth conditions differently, so solver iteration count, compliance, and
collision geometry are part of the learned environment.

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

### A minimal voltage-to-torque derivation

For a simple direct-current motor,

```math
V=Ri+L\frac{di}{dt}+k_e\omega,
```

```math
\tau_m=k_t i.
```

$V$ is applied voltage, $i$ current, $R$ resistance, $L$ inductance,
$k_e\omega$ back electromotive force, and $k_t$ torque constant. At steady
current, ignoring inductance:

```math
i\approx\frac{V-k_e\omega}{R},
\qquad
\tau_m\approx\frac{k_t}{R}(V-k_e\omega).
```

Available torque falls as speed increases because back electromotive force
consumes more of the voltage budget. Battery sag lowers $V$; heating changes
$R$; gearing transforms torque/speed and adds friction/backlash. An ideal
position actuator with unlimited torque misses this causal chain.

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

### Gain meaning from a one-joint closed loop

For one ideal joint with inertia $I$, viscous damping $b$, fixed target
$q^*$, and error $e=q-q^*$, substitute PD torque into
$I\ddot q+b\dot q=\tau$:

```math
I\ddot e+(b+K_d)\dot e+K_pe=0.
```

Divide by $I$ and compare with the standard second-order form

```math
\ddot e+2\zeta\omega_n\dot e+\omega_n^2e=0.
```

This gives

```math
\omega_n=\sqrt{\frac{K_p}{I}},
\qquad
\zeta=\frac{b+K_d}{2\sqrt{IK_p}}.
```

$\omega_n$ is a nominal natural frequency and $\zeta$ a damping ratio. Larger
$K_p$ raises response frequency; $K_d$ raises damping. Torque saturation,
sample delay, coupled links, contact, friction, and motor dynamics invalidate
the ideal formula quantitatively, but it supplies a starting hypothesis.

Run [`examples/pd_joint_and_kalman.py`](examples/pd_joint_and_kalman.py).
Its `simulate_joint` maps the equation to an integration loop and changes only
feedback delay; it is a teaching model, not a servo-identification substitute.

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

A pure delay $T_d$ contributes frequency-dependent phase lag

```math
\phi(\omega)=-\omega T_d
```

in radians. At angular frequency $\omega=20$ rad/s, a 20 ms delay adds
$-0.4$ rad, about $-23^\circ$. Feedback that arrives too late can reinforce
motion instead of damping it.

Sampling at frequency $f_s$ cannot uniquely represent a sinusoid at or above
the Nyquist frequency $f_s/2$. Real control needs margin well below that bound
because filters, zero-order action holds, inference, bus transport, and
actuators all add phase lag. “The policy runs at 50 Hz” is therefore incomplete
without the bandwidth of the behavior it must stabilize.

Deployment must define:

- when sensors are sampled;
- whether measurements have different timestamps;
- inference worst-case execution time (WCET), not only average time;
- when commands reach actuators; and
- what happens on deadline miss.

An explicit timestamped loop is preferable to assuming every field is “now”:

```python
sample = read_sensors_with_timestamp()
state = estimator.propagate_and_update(sample)
observation = contract.encode(state, command, previous_action)
action = actor(observation)
if clock.now() - sample.timestamp > maximum_observation_age:
    enter_fallback("stale observation")
else:
    send_rate_limited_target(action)
```

Training delay randomization should reproduce the measured distribution and
age semantics of this loop. Random delay can teach robustness; it does not
repair missing timestamps or an unbounded queue.

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

### Scalar Kalman update before the matrix version

Suppose a prior state estimate is Gaussian with mean $\hat x^-$ and variance
$P^-$, and a direct measurement $z=x+v$ has independent noise variance $R$.
The Kalman gain is

```math
K=\frac{P^-}{P^-+R}.
```

The posterior is

```math
\hat x^+=\hat x^-+K(z-\hat x^-),
```

```math
P^+=(1-K)P^-.
```

The term $z-\hat x^-$ is the **innovation**: what the sensor says beyond the
prediction. If prior uncertainty is large relative to sensor noise, $K$ is
near 1 and the estimate moves toward the measurement. If the sensor is noisy,
$K$ is small and the prior dominates.

The matrix Kalman filter repeats this idea with a dynamics prediction,
covariances, and an observation matrix. An extended Kalman filter linearizes
nonlinear motion/sensors with Jacobians. All require calibrated noise and model
assumptions; a covariance number is not automatically honest uncertainty.

The runnable
[`scalar_kalman_update`](examples/pd_joint_and_kalman.py) function implements
these three equations. Change prior and measurement variances before running
it and predict the gain direction.

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

### Least squares as the simplest identification map

Suppose a low-speed joint model is

```math
\tau_t=I\ddot q_t+b\dot q_t+\tau_c\,\mathrm{sign}(\dot q_t).
```

For each measured time step, form a feature row

```math
\varphi_t=[\ddot q_t,\ \dot q_t,\ \mathrm{sign}(\dot q_t)]
```

and parameters $\theta=[I,b,\tau_c]^T$. Stacking samples gives

```math
y=\Phi\theta+\epsilon.
```

Ordinary least squares minimizes squared residuals:

```math
\hat\theta=\arg\min_\theta\lVert\Phi\theta-y\rVert_2^2.
```

When $\Phi^T\Phi$ is invertible,

```math
\hat\theta=(\Phi^T\Phi)^{-1}\Phi^Ty.
```

The formula does not rescue uninformative data. If velocity barely changes,
inertia and damping may be impossible to separate. **Persistent excitation**
means the safe input explores enough frequencies/directions to identify the
chosen parameters. Derivative estimates amplify noise, current is not always
torque, and friction is nonlinear; validate on held-out trajectories before
turning a fit into randomization ranges.

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

### Expected constraints, barrier filters, and hard interlocks

A constrained Markov Decision Process can state

```math
\max_\pi J_R(\pi)
\quad\text{subject to}\quad
J_{C_i}(\pi)\leq d_i.
```

This limits an expected discounted cost. A rare catastrophic violation can
still be compatible with an acceptable expectation.

A control barrier function instead defines a safe set
$\mathcal{C}=\{x:h(x)\geq0\}$ and filters a proposed action by requiring, in a
continuous-time model,

```math
\dot h(x,u)+\alpha(h(x))\geq0.
```

A small quadratic program can choose the closest action satisfying this local
condition when the model and constraint are valid. Barrier certificates have
stronger structure than a reward penalty, but sensor error, model error,
infeasible constraints, discretization, and actuator saturation still need
handling. Electrical current cutoffs and physical emergency stop remain a
separate final authority.

As of 2026, safe robot learning is best treated as layers: train with costs and
randomization, filter commands where a verified model permits, enforce
realtime limits independently, and evaluate tail failures. Calling one layer
“safe RL” does not prove the whole robot safe.

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
9. A two-link endpoint has Jacobian
   $J=\begin{bmatrix}1&0\\1&1\end{bmatrix}$ and joint velocity
   $\dot q=[0.2,-0.1]^T$. Compute endpoint velocity. Then map force
   $F=[3,4]^T$ to joint torque.
10. A motor has $V=8$ V, $R=4\ \Omega$, $k_e=0.2$ V·s/rad,
    $k_t=0.2$ N·m/A, and $\omega=10$ rad/s. Ignoring inductance, compute
    current and torque. What happens if battery voltage falls?
11. For $I=0.02$, $b=0.05$, $K_p=8$, and $K_d=0.8$, compute the ideal natural
    frequency and damping ratio. Name two effects omitted by this model.
12. Compute phase lag from a 15 ms delay at $\omega=30$ rad/s in radians and
    degrees. Why is average inference time insufficient?
13. A scalar prior has mean 0, variance 0.25; a measurement is 0.4 with
    variance 0.01. Compute Kalman gain, posterior mean, and posterior variance.
14. Why can quaternion $q$ and $-q$ break a naive squared pose loss? Give a
    sign-invariant alternative.
15. Explain what makes a least-squares identification matrix singular or
    ill-conditioned. Propose a safe excitation improvement.
16. Distinguish an expected constrained-RL cost, a control-barrier filter, and
    a motor-driver overcurrent trip by the guarantee each can and cannot offer.

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
9. Endpoint velocity is

   ```math
   \dot x=J\dot q=
   \begin{bmatrix}1&0\\1&1\end{bmatrix}
   \begin{bmatrix}0.2\\-0.1\end{bmatrix}
   =\begin{bmatrix}0.2\\0.1\end{bmatrix}.
   ```

   Joint torque is

   ```math
   \tau=J^TF=
   \begin{bmatrix}1&1\\0&1\end{bmatrix}
   \begin{bmatrix}3\\4\end{bmatrix}
   =\begin{bmatrix}7\\4\end{bmatrix}.
   ```

10. Back electromotive force is $k_e\omega=2$ V. Current is
    $(8-2)/4=1.5$ A and torque is $0.2(1.5)=0.3$ N·m. Lower battery voltage
    reduces the current and torque available at the same speed; if voltage
    approaches back electromotive force, motoring torque approaches zero in
    this simplified model.
11. $\omega_n=\sqrt{8/0.02}=20$ rad/s. The damping ratio is
    $(0.05+0.8)/(2\sqrt{0.02\times8})=0.85/0.8=1.0625$, slightly overdamped in
    the ideal unsaturated model. Omissions include torque/current saturation,
    discrete delay, link coupling, contact, friction nonlinearities, motor
    electrical dynamics, and sensor noise.
12. $\phi=-\omega T_d=-30(0.015)=-0.45$ rad, about
    $-25.8^\circ$. Worst-case time and jitter determine whether individual
    cycles lose enough phase margin or miss deadlines; a safe mean can hide
    rare long delays.
13. $K=0.25/(0.25+0.01)=0.961538$. The posterior mean is
    $0+K(0.4)=0.384615$, and posterior variance is
    $(1-K)0.25\approx0.009615$. The confident measurement dominates.
14. They represent the same orientation but
    $\lVert q-(-q)\rVert^2=4$ for a unit quaternion. Use
    $1-|q^Tq^*|$, $1-(q^Tq^*)^2$, or flip the target sign so the dot product is
    nonnegative before computing a shortest-path error.
15. If excitation does not vary acceleration, velocity, or direction
    independently, feature columns become dependent or nearly so and several
    parameter combinations explain the same trace. Add bounded multi-frequency
    or multisine input across safe amplitudes/directions, measure synchronized
    outputs, and validate the resulting condition number and held-out error.
16. Expected constrained RL shapes a policy so average discounted cost stays
    under a learned/estimated budget; it allows estimation error and possibly
    rare violations. A barrier filter modifies proposed actions to keep a
    modeled state inside a mathematically defined safe set when its assumptions
    and feasibility hold. A motor-driver trip directly removes/bounds
    electrical authority when measured current crosses a threshold, independent
    of policy reward; it does not by itself guarantee balance or geometric
    collision avoidance.

</details>
