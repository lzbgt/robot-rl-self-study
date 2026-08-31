# 13. Customization Labs

These labs turn the book into a project course. Work on a branch, keep one
question per experiment, and preserve tests/configuration with every result.

The labs are ordered from reproduction to architecture change. Do not skip the
early ones because later work needs their habits: artifact identity, pure
tests, fixed evaluation, and deployable contracts.

Every lab notebook should contain seven fields:

```text
claim:        one falsifiable sentence
change:       exact code/config treatment
controls:     what remains fixed
mechanism:    why the change should affect the claim
measurement: raw cases plus aggregate/uncertainty
acceptance:   threshold chosen before seeing the result
decision:     keep, revise, or reject—with evidence
```

Use the live
[`tasks` package](https://github.com/pollen-robotics/microduck_rl/tree/main/src/mjlab_microduck/tasks),
[`tests` directory](https://github.com/pollen-robotics/microduck_rl/tree/main/tests),
and [`scripts`](https://github.com/pollen-robotics/microduck_rl/tree/main/scripts)
alongside these instructions. Start a focused branch and record the baseline:

```bash
git switch -c study/<short-lab-name>
git rev-parse HEAD
git status --short
uv run --with pytest pytest tests/
```

Do not mix unrelated local changes into a lab patch. A result is reproducible
only if its source commit, dirty diff, lockfile, command, seeds, and artifacts
are recoverable.

## Lab 0: reproduce before modifying

Goal: prove that you can run and explain the existing system.

1. Run all central processing unit (CPU) tests.
2. Run a 64-environment, five-iteration velocity smoke train.
3. Load an existing trained velocity checkpoint.
4. Record a 10-second rollout.
5. Export Open Neural Network Exchange (ONNX) and run the CPU rehearsal with a
   fixed forward command.
6. Draw the 61D observation and 14D action layout from memory.

Deliverable: the lab report template from Chapter 9 plus one paragraph
explaining why playback direction changes are command sampling, not random
actions.

Acceptance evidence must include:

- clean test summary and five-iteration log;
- resolved environment/agent configuration;
- checkpoint and export digests;
- one frozen 61D observation with checkpoint/ONNX 14D action comparison;
- a finite 10-second video with task/command identity; and
- measured policy-loop throughput in the CPU rehearsal.

Explain each arrow in this chain from memory, then verify it in code:

```text
task ID -> factory -> managers -> 61D raw observation
-> embedded normalizer -> deterministic actor -> 14D action
-> HOME/scale/order map -> actuator target
```

If one arrow cannot be explained, reproduction is incomplete even if the
viewer looks convincing.

## Lab 1: change a command distribution

Question: does more turn-in-place data improve turning without hurting forward
walking?

The current setting is:

```python
TURN_IN_PLACE_FRACTION = 0.15
```

If one rollout batch contains $N=N_{env}N_{step}$ transitions and command
buckets were independent per transition, expected turn samples would be
$Np_{turn}$. Commands persist for seconds, so transitions inside one command
segment are correlated; the true number of independent command episodes is
much smaller. Log both transition count and resample/episode count.

Changing $p_{turn}$ also changes the mixture objective:

```math
J=p_{turn}J_{turn}+p_{stand}J_{stand}+p_{general}J_{general}.
```

An improvement in $J_{turn}$ may accompany regression elsewhere because
training capacity and samples are reallocated. This is a multi-objective trade,
not a free augmentation.

Experiment:

1. Keep a baseline run and fixed evaluation battery.
2. Change only this fraction, for example to `0.25`.
3. Add/update a configuration test that checks the expected value.
4. Run the CPU suite and five-iteration smoke test.
5. Train both settings for the same budget and seed policy.
6. Compare turn-in-place success, translation drift, forward speed tracking,
   and total command coverage.

Do not conclude that 25% is universally better. Increasing one bucket reduces
the fraction of all other experience.

Use paired training seeds and a table such as:

| Variant | Train seed | Turn success | Turn drift | Forward MAE | Severe failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.15 | 101 | measure | measure | measure | classify |
| 0.25 | 101 | measure | measure | measure | classify |

Repeat over several seeds. Predeclare a decision such as “retain 0.25 only if
the paired turn-success estimate improves, its uncertainty is reported, and
forward mean absolute error (MAE) plus severe-failure rate remain within chosen
non-inferiority margins.” The margins must come from use requirements, not be
moved after results appear.

Also assert the mixture is valid:

```python
def test_command_bucket_probabilities_are_valid():
    cfg = make_microduck_velocity_env_cfg()
    twist = cfg.commands["twist"]
    assert 0.0 <= twist.rel_turn_in_place_envs <= 1.0
    assert 0.0 <= twist.rel_standing_envs <= 1.0
    assert (
        twist.rel_turn_in_place_envs + twist.rel_standing_envs
    ) <= 1.0
```

Adapt field names to the live configuration and test the resolved value; the
remaining probability belongs to general command sampling.

## Lab 2: add a small custom reward safely

Goal: learn the pure-helper → environment wrapper → configuration → test
pattern.

Suppose a new task needs a self-negating trunk-height error:

```python
# In tasks/mdp.py
def height_l1_from_values(height: torch.Tensor, target: float) -> torch.Tensor:
    """Return a nonpositive penalty; zero only at the target."""
    return -torch.abs(height - target)


def trunk_height_l1_penalty(
    env: ManagerBasedRlEnv,
    target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    origin_z = env.scene.terrain.env_origins[:, 2]
    height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - origin_z,
        nan=0.0,
    )
    return height_l1_from_values(height, target)
```

Wire it in the relevant task factory:

```python
cfg.rewards["trunk_height_l1"] = RewardTermCfg(
    func=microduck_mdp.trunk_height_l1_penalty,
    weight=0.2,  # positive: the function already returns <= 0
    params={"target": 0.095},
)
```

Test the math and sign without a simulator:

```python
def test_height_l1_penalty():
    h = torch.tensor([0.095, 0.105, 0.075])
    out = height_l1_from_values(h, target=0.095)
    torch.testing.assert_close(out, torch.tensor([0.0, -0.01, -0.02]))


def test_height_penalty_weight_is_positive():
    cfg = make_my_task_env_cfg()
    assert cfg.rewards["trunk_height_l1"].weight > 0.0
```

Trace units and sign. The helper returns meters with a negative sign. A weight
of 0.2 reward units per meter gives

```math
r_h=-0.2|h-h^*|.
```

At a 2 cm error the contribution is $-0.004$ per step. Over 1,000 undiscounted
logged steps that could sum to about $-4$, but Proximal Policy Optimization
(PPO)'s effect depends on competing terms and state visitation. Compute observed
reward mass after the smoke/full run rather than assuming `0.2` is large or
small.

The absolute-value subgradient is constant away from zero:

```math
\frac{\partial r_h}{\partial h}
=-0.2\,\mathrm{sign}(h-h^*),
\qquad h\ne h^*.
```

The policy gradient does not differentiate directly through simulator height;
the equation still reveals objective geometry: unlike a narrow Gaussian, L1
error does not vanish far from target. That can help recovery but can also
dominate phases where target height is inappropriate.

Before keeping the reward, answer:

- Is the target measured from the actual current robot model?
- Is height meaningful in every task phase?
- Could a fallen pose match the height?
- Does this term block motion the task physically requires?
- How large is its weighted mass relative to the positive objective?

This example teaches wiring; it is not a recommendation to add a height
penalty to the main walking task.

Extend the test matrix:

```python
def test_height_penalty_is_symmetric_and_monotone():
    near = height_l1_from_values(torch.tensor([0.090, 0.100]), 0.095)
    far = height_l1_from_values(torch.tensor([0.075, 0.115]), 0.095)
    torch.testing.assert_close(near[0], near[1])
    torch.testing.assert_close(far[0], far[1])
    assert torch.all(far < near)
```

Add a configuration-level test that the selected trunk and terrain-relative
height are correct. Then audit stable counterexamples: prone, supine, and side
poses can sometimes match height. If the true goal is *progress to upright*, a
bounded potential difference in tilt plus a terminal pose criterion may be
harder to farm than continuous absolute height.

## Lab 3: design a new task from a template

Choose the closest family:

| Intended behavior | Starting point |
| --- | --- |
| locomotion | velocity |
| episodic trick ending in a pose | stand-up |
| commanded two-state transition | sit-stand |
| dynamic maneuver | roulade |

Before code, complete a task specification:

```text
initial-state classes and probabilities:
command and resampling semantics:
actor-observable variables:
critic-only variables:
14D action interpretation:
success state and minimum hold time:
true termination versus truncation:
positive objective terms:
cost/constraint terms:
five cheapest plausible exploits:
curriculum frontier:
deployment owner and skill-handoff states:
```

Translate it into a task-specific Markov Decision Process objective:

```math
J(\theta)=\mathbb{E}_{s_0,c,\xi,\pi_\theta}
\left[\sum_{t=0}^{H-1}\gamma^t
\sum_i w_i f_i(s_t,a_t,s_{t+1},c;\xi)\right],
```

where $\xi$ contains randomized physical/sensor conditions. Circle every
quantity unavailable to the actor and confirm it appears only in critic,
reward, reset, or evaluation logic—not in deployed observation.

Implementation checklist:

1. Write the task's observable success criterion and plausible exploits.
2. Verify target states in physics before Proximal Policy Optimization (PPO).
3. Create `microduck_<name>_env_cfg.py` by building on the closest factory.
4. Put custom Markov Decision Process (MDP) functions in `tasks/mdp.py`, grouped
   with the task.
5. Preserve the 61D actor command layout and 14D action layout.
6. Select the correct walk/all-collisions/roller model.
7. Give the runner a distinct `experiment_name`.
8. Register train/play variants in `tasks/__init__.py`.
9. Add a backlash twin only if it mirrors the base model correctly.
10. Write config tests for dimensions, signs, selectors, gates, and task
    registration.
11. Run CPU tests and the five-iteration smoke train.
12. Evaluate the main behavior, every stable flop, and likely reward hacks.

Minimal registration shape:

```python
register_mjlab_task(
    task_id="Mjlab-MyTask-Flat-MicroDuck",
    env_cfg=make_microduck_my_task_env_cfg(),
    play_env_cfg=make_microduck_my_task_env_cfg(play=True),
    rl_cfg=MicroduckMyTaskRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

Add invariant tests before reward tuning:

```python
def test_my_task_family_contract():
    cfg = make_microduck_my_task_env_cfg()
    assert expected_actor_observation_dimension(cfg) == 61
    assert expected_action_dimension(cfg) == 14
    assert all_penalty_weights_have_expected_sign(cfg)
    assert selectors_resolve_on_configured_robot(cfg)

def test_my_task_is_registered():
    assert "Mjlab-MyTask-Flat-MicroDuck" in list_registered_task_ids()
```

The helper names above are pseudocode; use existing repository test utilities
and patterns instead of inventing a second configuration system. A valuable
test fails when someone changes the selected model, command order, or reward
sign—not only when the file fails to import.

Define an acceptance ladder:

```text
physics target is supportable
-> manager construction and 61/14 contracts pass
-> five-iteration run is finite and exports
-> skill appears in nominal cases across seeds
-> stable-flop/exploit battery is rejected
-> randomization boundaries retain performance
-> deployment rehearsal meets timing/interface contract
```

## Lab 4: add a curriculum from evidence

Do not begin by inventing ten stages. First train the easier slice and identify
the measured frontier.

Example goal: widen a pose command only after small commands track well.

```python
cfg.curriculum["pose_range"] = CurriculumTermCfg(
    func=microduck_mdp.pose_command_range_curriculum,
    params={
        "command_name": "head_pose",
        "range_stages": [
            {"step": 0, "ranges": ((-0.05, 0.05),) * 4},
            {"step": 500 * 24, "ranges": ((-0.2, 0.2),) * 4},
        ],
    },
)
```

For the real head, use per-joint ranges rather than the identical illustrative
ranges above because yaw, pitch, and roll have different limits.

Test that:

- stage steps use environment-step units;
- the live command-manager term changes;
- the first and last ranges are intended;
- the policy sees nonzero samples before a reward is introduced; and
- metrics do not collapse at the stage boundary.

The schedule above is iteration-based because repository curriculum steps are
$k=24\times\text{iteration}$. At 500 iterations, the stage begins at step
12,000 regardless of environment count. Log the active range and empirical
command histogram so a metric jump can be separated from a logging-weight
jump.

A configuration test should exercise boundary values:

```python
def test_pose_range_curriculum_boundaries(env):
    # Pseudocode: invoke the curriculum through the initialized manager.
    apply_curriculum(env, step=500 * 24 - 1)
    before = env.command_manager.get_term_cfg("head_pose").ranges
    apply_curriculum(env, step=500 * 24)
    after = env.command_manager.get_term_cfg("head_pose").ranges
    assert before == ((-0.05, 0.05),) * 4
    assert after == ((-0.2, 0.2),) * 4
```

Use the live helper signature and per-joint physical ranges in production.
Critically, mutate/read the manager copy after initialization; changing
`env.cfg` may not change the running term.

For a competence-triggered alternative, define a held-out metric $m_k$, entry
threshold $u$, exit threshold $l<u$, and patience $P$:

```text
advance after m_k >= u for P consecutive evaluations
do not regress unless m_k <= l for P consecutive evaluations
```

The gap between $u$ and $l$ is hysteresis; it prevents noisy oscillation between
stages. Save curriculum state in checkpoints. Compare scheduled and
competence-triggered designs at equal transition budgets rather than assuming
adaptive pacing is automatically superior.

## Lab 5: obstacle avoidance—choose the architecture first

The current velocity policy cannot avoid obstacles. There are two different
projects hidden inside the phrase “add obstacle avoidance.”

### Option A: perception/planning above locomotion

```text
camera/depth/ToF
      v
local obstacle model + planner
      v
safe local [vx, vy, yaw_rate] command
      v
existing 61D Microduck walking policy
```

Depth cameras, range rays, and time-of-flight (ToF) sensors differ in field of
view, minimum range, sunlight sensitivity, latency, and missing-data behavior.
Choose a sensor the physical robot can reproduce; a perfect simulator depth
map is not a deployable observation.

This is the recommended first project. It preserves the proven fast motor
policy and observation contract. The planner can be classical, learned, or
cloud-assisted, but a local collision/stop layer must remain available when
network service is absent or stale.

Build a simulation harness that varies obstacles and checks:

- planner command expiry;
- stop distance;
- minimum clearance;
- collision rate;
- progress to goal; and
- behavior on missing/stale perception.

The locomotion policy still needs robustness to the planner's command changes,
but it does not need pixels.

Start with stopping geometry. For speed $v$, end-to-end reaction delay $\tau$,
conservative deceleration $a_{stop}>0$, and margin $m$, a one-dimensional
stopping bound is

```math
d_{stop}=v\tau+\frac{v^2}{2a_{stop}}+m.
```

The first term is distance traveled before braking begins; the second follows
from $v_f^2=v^2-2a_{stop}d$ with $v_f=0$. Measure $a_{stop}$ across surfaces,
battery states, and command transitions. Perception range must exceed this
bound plus uncertainty at every allowed speed.

A small local planner can score candidate twists without modifying the motor
policy:

```python
def choose_twist(candidates, local_map, goal, now):
    valid = []
    for twist in candidates:
        path = rollout_kinematic_model(twist, horizon_s=1.0)
        clearance = minimum_clearance(path, local_map)
        if clearance < stopping_margin(twist, now):
            continue
        score = goal_progress(path, goal) - turn_cost(twist)
        valid.append((score, twist))
    return max(valid)[1] if valid else ZERO_TWIST
```

This resembles a sampled Model Predictive Control (MPC) or dynamic-window
planner. Its model need not predict joint contacts; it proposes a safe local
velocity inside the locomotion policy's trained command envelope. Verify that
command acceleration is also within training experience.

Do not use time-to-collision alone at zero relative velocity or with invalid
depth. Missing/old perception must produce an explicit conservative state, not
an infinity that looks safe. Record source timestamp and map age in every
planner decision.

### Option B: an end-to-end exteroceptive policy

```text
proprioception + command + depth/rays/features
      v
new actor architecture
      v
joint actions
```

This requires a new versioned policy family because extra features break the
61D hot-swap contract. Define:

- sensor model and hardware availability;
- feature dimensions/order/units and missing-data semantics;
- model architecture and inference budget;
- obstacle/goal scene generator;
- collision, clearance, progress, and deadlock objectives;
- domain randomization for sensor noise/dropout and obstacle geometry;
- runtime contract/version negotiation; and
- safety behavior independent of learned perception.

Do not merely add boxes and a collision penalty. Without pre-contact
observation, the policy learns only collision reaction.

One compact exteroceptive representation is $K$ body-frame range rays plus a
validity mask:

```math
x^{ray}_i=\mathrm{clip}\left(\frac{d_i}{d_{max}},0,1\right),
\qquad m_i\in\{0,1\}.
```

Never encode “missing” as ordinary maximum distance unless training and safety
semantics intentionally mean “assume clear.” A mask lets the network
distinguish no return from a measured far surface. Include ray angles/frame,
height, update rate, age, clipping, and self-geometry exclusions in the schema.

Possible network architectures include:

- concatenate a small ray vector with proprioception before a multilayer
  perceptron;
- encode a depth image with a convolutional/transformer encoder, then fuse a
  compact feature with proprioception; or
- train a perception encoder separately and freeze/distill it for bounded
  real-time inference.

The simple concatenated baseline is easiest to test. Image encoders can use
richer geometry but demand more data, augmentation, compute, and latency
matching.

An illustrative objective is

```math
r_t=w_p\left(\Phi_{goal}(s_{t+1})-\Phi_{goal}(s_t)\right)
-w_c\mathbf 1[collision]
-w_n\max(0,d_{safe}-d_{min})
-w_a\lVert a_t-a_{t-1}\rVert_2^2.
```

Progress difference avoids paying forever for standing near a goal. Collision
is a severe event; near-clearance shaping supplies earlier signal; action
change discourages jitter. Audit the shortcut of freezing far from obstacles:
without progress or a time cost, it may be safest and highest-return.

Train with procedurally varied widths, shapes, heights, materials, motion,
sensor dropout, latency, and spawn/goal relationships, then reserve generator
seeds and geometry families for test. Curriculum can begin with sparse static
obstacles, but evaluation must include clutter, narrow passages, dead ends,
dynamic crossings, missing data, and the exact stop behavior.

### Lab 5 acceptance matrix

| Case | Required evidence |
| --- | --- |
| clear path | progress/tracking is not materially degraded |
| static obstacle | no collision; minimum clearance and stop/progress pass |
| narrow passage | bounded success/failure, no oscillatory deadlock |
| dynamic crossing | prediction/reaction remains within latency envelope |
| stale/missing sensor | local safe stop occurs before stopping bound |
| unseen geometry | uncertainty and failure classes reported |
| cloud/network loss | local planner/safety behavior remains available |

Option A and B should be compared on end-to-end latency, failure severity,
data/compute cost, schema complexity, and physical evidence—not on novelty.

## Lab 6: measure one sim-to-real parameter

Choose one subsystem, such as a single XL330 step response.

1. Define a safe bench input and collect timestamped command, position,
   velocity, current/voltage, and load conditions.
2. Replay the same input in the Better Actuator Models (BAM) testbench
   simulation.
3. Compare rise time, overshoot, steady-state error, and decay/friction.
4. Fit a nominal parameter using training data.
5. Validate it on a different input.
6. Estimate a realistic randomization range from repeated trials.
7. Record the model version and evidence.

The lesson is broader than one actuator: fit first, randomize uncertainty
second.

Use at least two excitation families, for example small bidirectional steps for
friction/deadband and larger safe sweeps for speed/voltage behavior. Split by
trajectory, not individual adjacent samples, because neighboring timestamps are
strongly correlated.

For candidate parameters $\theta$, define a weighted residual:

```math
L(\theta)=\sum_t
\left(
w_q(q_t-\hat q_t(\theta))^2
+w_v(\dot q_t-\widehat{\dot q}_t(\theta))^2
+w_i(i_t-\hat i_t(\theta))^2
\right).
```

Weights nondimensionalize signals or express priorities; otherwise a
large-numeric-unit channel dominates. Report each physical residual separately
even when optimizing one scalar loss.

A transparent coarse-to-fine search is a good beginner baseline:

```python
best = None
for friction in friction_grid:
    for motor_gain in gain_grid:
        prediction = replay_sim(command, friction, motor_gain)
        loss = weighted_residual(measurement, prediction)
        if best is None or loss < best.loss:
            best = Result(loss, friction, motor_gain)
```

Then validate `best` on held-out command shapes, directions, loads, and battery
states. Inspect residual versus time; one scalar score can hide a phase lag
that is disastrous in closed loop. If many parameter pairs fit equally well,
the experiment did not identify them separately—design a new excitation rather
than reporting false precision.

## Lab 7: create a policy acceptance battery

Write a headless evaluator for one task that runs a matrix of seeds and command
cases. Output a machine-readable summary:

```json
{
  "task": "Mjlab-Velocity-Flat-MicroDuck",
  "checkpoint": "model_2000.pt",
  "cases": 100,
  "successes": 92,
  "falls": 5,
  "nonfinite": 0,
  "mean_linear_tracking_error_mps": 0.04,
  "mean_yaw_tracking_error_radps": 0.09
}
```

Define success before running. Keep the raw per-case data so an aggregate
number can be audited. Add video sampling for human-visible quality.

Prefer newline-delimited JavaScript Object Notation (JSONL) for raw rows and a
separate summary. Each row should include policy/config hashes, train/evaluation
seeds, scenario identifier (ID), command, reset/disturbance parameters, success, continuous
metrics, failure class, and video/log reference. JavaScript Object Notation
(JSON) is a portable structured-text format; JSONL keeps one independently
parseable record per line.

Evaluator pseudocode:

```python
for case in fixed_case_manifest:
    env.reset(seed=case.seed, overrides=case.reset)
    apply_command(case.command)
    trace = rollout(policy, horizon=case.horizon)
    row = {
        **case.identity(),
        "success": success_rule(trace),
        "metrics": compute_metrics(trace),
        "failure_class": classify_failure(trace),
    }
    append_jsonl("episodes.jsonl", row)
```

The manifest must be read-only during final assessment. Validate the evaluator
itself with synthetic traces: known success, exact threshold, fall, timeout,
and nonfinite cases. An off-by-one time or frame convention can reverse labels.

Report Wilson intervals for per-slice success, seed-level values for algorithm
comparisons, and a lower-tail metric for severe behavior. A hundred simulation
episodes from one checkpoint do not equal a hundred independent training
seeds; state which uncertainty is estimated.

## Lab 8: build a real-time budget and stale-data test

Goal: prove that the deployment loop implements the timing distribution the
policy expects.

1. Draw the path from sensor capture through observation, inference, safety,
   transport, and actuator application.
2. Instrument each boundary with a monotonic timestamp and sequence number.
3. Run at representative central processing unit load and record at least
   several minutes of cycles.
4. Report median, 95th, 99th, maximum observed duration, missed deadlines, and
   information age—not only average inference time.
5. Calculate on-wire bus utilization including framing/turnaround/retries and
   compare it with a captured trace.
6. Inject a missed inference cycle, frozen sensor timestamp, late command, and
   lost cloud connection.
7. Verify the real-time board/local safety state reaches the predeclared safe
   response without cloud help.

Use `python examples/realtime_data_age.py` to explain why a latest-value
mailbox is appropriate for reactive state. Then implement the same semantics
with explicit overwrite/drop counters in the runtime.

The deliverable is a timing table:

| Stage | Nominal budget | p50 | p99 | max observed | deadline action |
| --- | ---: | ---: | ---: | ---: | --- |
| sensor and transport | choose | measure | measure | measure | invalidate stale |
| observation | choose | measure | measure | measure | safe fallback |
| inference | choose | measure | measure | measure | hold, ramp, or disable |
| safety/send | choose | measure | measure | measure | local stop |
| end-to-end age | `< trained/tested bound` | measure | measure | measure | reject cycle |

“No misses observed” is evidence over a stated duration/load, not proof that a
deadline can never be missed. A hard real-time claim needs a stronger platform
and worst-case analysis; the safety state is still required.

## Lab 9: design and test one skill handoff

Choose walking → standing, walking → recovery, or recovery → walking. Write a
finite-state machine with explicit guards:

```text
ACTIVE_A
  -> HANDOFF_REQUESTED
  -> ENTER_HANDOFF_SET
  -> VERIFY_BUNDLE_AND_STATE
  -> ACTIVE_B
  -> ACCEPT or FALLBACK
```

Define the handoff set $\mathcal H_{A\rightarrow B}$ by measurable bounds such
as pose, speed, contacts, action magnitude, sensor age, and command validity:

```math
\mathcal H_{A\rightarrow B}
=\{s: |\mathrm{tilt}(s)|<\epsilon_R,
\lVert\dot q\rVert<\epsilon_v,
\mathrm{contacts}(s)\in C,
\mathrm{age}(s)<\tau_{max}\}.
```

State how policy B initializes previous action/history and how unrelated
command slots are cleared. Test:

- request at a valid handoff state;
- request outside the set;
- wrong policy schema/hash;
- stale command during transition;
- B fails to make progress before timeout; and
- repeated switch requests.

Measure target discontinuity, saturation, peak tilt, contacts, transition time,
success, and fallback. Do not hide a discontinuity with an untrained blend;
train or validate any transition trajectory the system applies.

## Capstone: a complete evidence-backed customization

Your capstone should include:

```text
problem statement and non-goals
architecture choice
physics assumptions and measurements
observation/action/command contract
reward and termination rationale
likely exploits
unit and configuration tests
five-iteration smoke evidence
training configuration and checkpoints
fixed evaluation battery and videos
ONNX export/interface inspection
CPU deployment rehearsal
sim-to-real risk register
known limitations and next experiment
```

The capstone is complete when another person can reproduce the result and
understand where it is safe to generalize—not when one attractive video exists.

Recommended milestones:

| Review | Required artifact | Stop condition |
| --- | --- | --- |
| problem review | specification, non-goals, hazard/reward-hack list | capability is not observable/trainable |
| physics review | equilibrium/actuator/sensor evidence | target or model is implausible |
| software review | schema, pure/config tests, registered task | invariant or clean sync fails |
| smoke review | finite five-iteration export/reload | integration failure |
| learning review | multi-seed curves and fixed validation | main skill absent or exploit dominates |
| assessment review | locked checkpoint and untouched battery | acceptance or severe-tail criterion fails |
| deployment review | parity, timing, safety/fault injection | contract/deadline/safe state fails |

Grade evidence, not novelty:

```text
20% problem/task specification and causal reasoning
15% physics/model measurement
15% tests and reproducible software
20% evaluation design and statistics
15% deployment/safety contract
10% diagnosis of limitations and alternatives
 5% clarity of handoff
```

A negative result with a well-falsified hypothesis, preserved artifacts, and a
clear next experiment can score higher than a cherry-picked apparent success.

## Where to continue learning

After this book, read code and experiments in this order:

1. the velocity task and its focused tests;
2. sit-stand for command-conditioned state transitions;
3. stand-up for reset distributions and recovery shaping;
4. roulade for dynamic-task reward lessons;
5. backlash for observation/physics consistency; and
6. roller tasks for passive joints and architecture-specific indices.

Return to the [book index](README.md) whenever a new experiment crosses from
task design into training or deployment.

## Folded reference outcomes

<details>
<summary>Show reference outcomes for Labs 0–4</summary>

**Lab 0 — reproduction.** A complete result binds a pinned source/lock and
resolved configuration to the checkpoint, exported graph, frozen observation
corpus, parity output, rehearsal log, and video by content hashes. It explains
that the environment's sampled command changes intention while the
deterministic policy conditionally responds; it does not call the learned
action random. CPU tests and smoke training answer different claims and both
are retained.

**Lab 1 — command mixture.** A strong analysis first proves the resolved
probability changed and other task configuration did not. It logs actual
command episode counts, pairs baseline/treatment seeds, and reports turn plus
forward/idle slices with uncertainty and severe failures. A result that improves
turn success but violates the predeclared forward non-inferiority margin is a
tradeoff, not an unqualified improvement. Exact-zero training remains explicit.

**Lab 2 — reward helper.** The pure tensor tests establish zero at target,
symmetry, monotonic magnitude, nonfinite handling policy, and returned sign.
The configuration test establishes a positive weight for the self-negating
helper and correct trunk/terrain selectors. The run audit shows every weighted
penalty nonpositive and reports its observed mass. The write-up checks prone,
supine, and side-pose counterexamples and rejects the term if matching height
rewards a stable flop or blocks required motion.

A pure reward helper and wiring test can follow this pattern:

```python
def limit_proximity_cost(q, lower, upper, margin):
    distance = minimum(q - lower, upper - q)
    return maximum(0.0, margin - distance) / margin

def test_limit_cost_is_zero_away_from_limits():
    assert limit_proximity_cost(0.0, -1.0, 1.0, 0.1) == 0.0

def test_limit_cost_grows_near_upper_limit():
    assert limit_proximity_cost(0.95, -1.0, 1.0, 0.1) > 0.0
```

Adapt this to Torch and repository conventions. The helper returns a
nonnegative cost, so the configured reward weight should be negative. This
contrasts deliberately with the self-negating height helper.

**Lab 3 — new task.** The submission contains the pre-code worksheet, measured
supportability of target states, the closest-template rationale, registered
train/play configuration, 61D/14D or explicitly versioned contract, selectors
resolved on the actual model, sign/gate tests, and five-iteration export/reload.
Evaluation includes intended success, every plausible stable flop, reset
classes, and a named exploit taxonomy. A novel task with no deployment owner or
handoff state remains incomplete.

**Lab 4 — curriculum.** Boundary tests show stage selection immediately before
and at each step, using the initialized manager rather than a stale config
copy. Logs contain active range/weight and empirical sample distribution.
Skill metrics are aligned to boundaries. If collapse repeats at a boundary,
the outcome is “schedule outpaces competence”; delaying/staggering one stage is
the next controlled test. A competence-triggered variant persists hysteresis,
patience, and stage state in checkpoint provenance.

</details>

<details>
<summary>Show reference outcomes for Labs 5–9</summary>

**Lab 5 — obstacles.** Option A retains the 61D motor-policy contract and
demonstrates timestamped perception, an explicit invalid-data state, commands
inside the trained envelope, measured stopping-distance assumptions, local
expiry/stop behavior, and held-out-layout metrics. Option B versions the schema,
documents every ray/image feature and validity mask, matches inference latency,
trains pre-contact information with progress/clearance/collision objectives,
and tests unseen geometries plus sensor dropout. In either option, rendering a
box without a causal observation and objective is rejected. Cloud loss must not
remove local stop authority.

**Lab 6 — identification.** Raw timestamped bench traces, safe excitation,
calibration, and load/battery/temperature conditions are preserved. Candidate
parameters are fit on complete trajectories and tested on different command
families. The report shows position/velocity/current residuals over time, not
only one loss. Correlated or unidentifiable parameters are labeled; the next
excitation is designed to separate them. Randomization centers on validated
nominal behavior and covers measured residual/repeatability rather than an
arbitrary wide interval.

**Lab 7 — evaluator.** A locked case manifest and tested success classifier
produce raw JSONL rows, a reproducible summary, per-slice counts/intervals,
training-seed identity, failure severity, and synchronized sample videos. A
nonfinite simulator case is classified separately from a policy fall. The
summary is regenerated from raw rows by one command, and the final assessment
manifest was not used for tuning.

**Lab 8 — timing.** The trace measures capture-to-actuation age and every
intermediate stage under representative load. It reports distribution tails,
deadline misses, bus trace versus analytic budget, data drops/overwrites, and
the exact response to injected stalls/staleness/disconnects. A latest-value
mailbox keeps feedback age bounded, while a separate lossless/logging path
preserves events. The local safe response works with cloud and brain process
unavailable.

**Lab 9 — handoff.** The finite-state machine rejects incompatible bundles and
requests outside the handoff set, atomically resets command semantics/history,
and transitions only after measurable guards pass. Success and fallback paths
are exercised. Logs show target discontinuity, saturation, state/command age,
transition time, contacts, and final state. Any action blend is included in the
trained or explicitly validated transition distribution.

</details>

<details>
<summary>Show the capstone completion rubric</summary>

A passing capstone lets an independent learner reproduce the software pipeline
from a clean checkout, rerun a bounded evaluation, inspect raw evidence, and
state exactly what was *not* demonstrated. Its chain is:

```text
measured/declared assumptions
-> executable task and invariant tests
-> finite smoke/export evidence
-> multi-seed learning and locked checkpoint selection
-> untouched per-slice assessment with uncertainty/failures
-> one-step export parity and deployment rehearsal
-> timing/safety/fault-injection record
-> operating envelope, rollback, and next falsifiable hypothesis
```

Automatic failure conditions include an unrecoverable artifact, absent raw
evaluation rows, an observation/action schema mismatch, positive weighted
penalty due to sign error, capability claim without causal observation/reward,
unsafe severe failure hidden by an average, or physical testing before the
declared interface/safety gates pass.

A high-quality negative capstone might show that an obstacle planner cannot
stop within range once measured perception age and wet-surface deceleration are
included. Rejecting deployment, documenting the boundary, and proposing a
lower speed or better sensor is successful engineering and successful learning.

</details>
