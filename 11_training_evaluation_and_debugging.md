# 11. Training, Evaluation, and Debugging

Training is not “start PPO and wait.” A useful run is a controlled experiment
with a hypothesis, resolved configuration, checkpoints, per-term metrics, and
rollouts that test the intended behavior.

## 11.1 The end-to-end loop

```text
physics assumption check
        v
CPU config/reward tests
        v
64-env, 5-iteration smoke train
        v
full vectorized run
        v
metric and checkpoint inspection
        v
fixed evaluation battery + video
        v
one evidence-backed change
        +--------------------------> repeat
```

For a target/rest pose, add a physics-only check before training: hold its
control target from noisy initial states for several seconds and measure both
height and tilt. A fallen body can have a stable height, so height alone cannot
prove equilibrium.

## 11.2 Before launching a full run

Write a short experiment note:

```text
hypothesis:
task ID and commit:
baseline run/checkpoint:
one intentional change:
expected main metric effect:
expected visible behavior:
failure criterion:
training budget:
```

Then run:

```bash
uv run --with pytest pytest tests/

uv run train <TASK_ID> \
    --env.scene.num-envs 64 \
    --agent.max_iterations 5
```

Do not combine reward redesign, new observations, stronger DR, a new robot
model, and new PPO hyperparameters in one experiment. Even a successful result
will not tell you which change mattered.

## 11.3 Read the training dashboard in layers

### Layer 1: pipeline health

Check:

- iteration advances;
- simulation and learning FPS are plausible;
- losses and KL are finite;
- no NaN termination spike appears;
- checkpoints continue to save; and
- GPU memory remains stable.

### Layer 2: episode behavior

The main velocity episode lasts 20 seconds, or 1,000 policy steps at 50 Hz. A
mean episode length near 1,000 suggests most episodes reach timeout. It does
not prove walking; standing still may also survive.

A rising episode length can be good for locomotion and irrelevant for a fixed
short trick. Interpret it against the task.

### Layer 3: reward signs

For every penalty contribution:

```text
Episode_Reward/<penalty> <= 0
```

If a penalty is positive, stop and inspect its function/weight sign. PPO will
farm a double-negated violation.

### Layer 4: the main skill

Track the term that directly represents the task:

```text
walking      linear/angular tracking and air time
stand-up     upright/height completion and landing quality
sit-stand    commanded transition completion
spin         commanded yaw-rate profile and grounded support
```

Total reward can rise because action rate fell while locomotion remained
absent. Main-task metrics and video must agree.

### Layer 5: policy distribution

Watch entropy/noise scale, KL, and action statistics. Early entropy collapse
can freeze a do-nothing solution. Persistently large action noise can hide a
useful mean policy. These are diagnostic signals; do not tune them before
checking the task specification.

## 11.4 Weighted logs and reward mass

`Episode_Reward/<term>` is the weighted contribution. If a term's weight is
zero, its log is zero even when the underlying behavior is poor. If a
curriculum changes the weight, a jump in the log may reflect the coefficient,
not behavior.

Compare reward mass:

```math
\text{mass}_i \approx |w_i|\,\mathbb{E}[|f_i|]
```

Suppose one task has 8 points/step of positive reward and another has 2. An
action-rate weight of `-0.1` is four times weaker relative to the first task,
even though the text of the configuration is identical.

## 11.5 Evaluation is a separate experiment

Do not judge only the final checkpoint or one random viewer episode. Create a
battery that covers the state distribution you care about.

Example velocity battery:

| Case | Command | What to measure |
| --- | --- | --- |
| idle | `[0, 0, 0]` | stable stance, head bias, jitter |
| forward | `[+0.2, 0, 0]` | achieved speed, drift, falls |
| backward | `[-0.2, 0, 0]` | tracking and foot clearance |
| lateral | `[0, +0.15, 0]` | side tracking and collisions |
| turn in place | `[0, 0, +0.6]` | yaw rate, translation drift |
| combined | `[+0.2, 0, +0.5]` | curve tracking |
| head command | fixed twist + head deltas | per-joint error and gait effect |
| disturbance | selected push cases | recovery and settling time |

The default `play` command generator is useful for qualitative variety, but a
fixed battery is better for checkpoint comparison. The deployment rehearsal in
`scripts/infer_policy.py` allows explicit initial velocity commands and
keyboard changes.

## 11.6 Watch video and compute state metrics

Video answers geometric questions metrics can miss:

- Which body or collision geometry touched?
- Did a “forward roll” go over the head or shoulder?
- Did the feet step or slide?
- Is the head tracking target or serving as a counterweight?

State metrics answer questions video can mislead:

- exact trunk tilt and height;
- achieved versus commanded speed;
- peak angular rate and acceleration;
- contact timing and impact force;
- per-spawn success rate; and
- end-state clusters.

Use both. A smooth camera angle can hide lateral drift, and a scalar threshold
can split one visibly identical behavior cluster.

## 11.7 Checkpoint selection

The last checkpoint is not automatically best. A policy can discover the skill
and later trade it away when a curriculum adds difficulty or regularization.

Evaluate checkpoints around:

- first visible skill discovery;
- every curriculum boundary;
- sudden main-metric changes;
- entropy or KL events; and
- the final iteration.

Load an exact local checkpoint:

```bash
uv run play <TASK_ID> \
    --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_2000.pt \
    --num-envs 1
```

Preserve the resolved `params/` directory with the selected model.

## 11.8 Common failure patterns

### The robot does nothing

Likely causes:

- attempt penalties or smoothness costs dominate before skill discovery;
- positive task rewards are too sparse or flat;
- command distribution rarely includes the desired case;
- a tracking standard deviation is too tight at initial errors; or
- standing earns most of the same positive stack as moving.

Measure per-term contributions before increasing entropy or learning rate.

### The robot moves violently

Check whether an arrival state pays a per-step jackpot, whether impact/contact
is under-specified, and whether action-rate regularization has begun. For a
small robot, 3.5–5.5 rad/s body rotation may be physically natural during a
tumble; penalize impact and thrash rather than importing human-scale angular
speed intuition.

### It finds the beginning but not the finish

The successful frontier may almost never appear in on-policy data. Add
reverse-curriculum resets near later maneuver states and consolidate each
slice before widening.

### It camps at a waypoint

Per-step waypoint rewards create local jackpots. For an episodic landing task,
prefer a fixed final target with progress/landing shaping and let RL discover
the path.

### It reaches the goal too fast and crashes

An early-arrival state that keeps paying makes speed profitable. Use a slewed
target or rate-limited progress so being ahead of the intended transition does
not pay extra.

### Joints park at hard limits

The default limit cost may activate only near the last portion of the range.
Add a qpos-side limit-proximity penalty for the offending joints rather than
shrinking the wide command range needed by low-stiffness servos.

### Training becomes NaN

Do not merely lower the learning rate. Determine whether the first nonfinite
value originated in physics, a sensor, reward math, observation normalization,
or PPO. Inspect `Episode_Termination/nan_state`, contact conditions, and a
minimal reset/step reproduction.

### Simulation succeeds and hardware fails

Audit the interface before redesigning rewards:

```text
joint names/order and signs
HOME offsets
units and coordinate frames
observation order and normalization
sensor delay/noise/filtering
action scaling and filtering
policy frequency and missed deadlines
actuator voltage, friction, saturation, backlash
command-slot writes
```

In-sim playback applies the checkpoint normalizer and can hide an ONNX export
mistake. Runtime all-zero commands can select “stand” when an application
expected a trick flag. These interface failures look like bad learning.

## 11.9 Curriculum diagnosis

Plot main metrics against curriculum stages. If a metric drops at exactly the
same step every run, inspect the stage before changing PPO.

Example reasoning:

```text
observation: air-time reward rises through iteration 700
event: action-rate cost doubles at iteration 750
result: air-time collapses and never recovers

hypothesis: smoothness tax arrived before stepping consolidated
experiment: delay only that stage, preserve every other setting
```

Curricula should introduce challenge after competence, not according to an
arbitrary desire to finish sooner.

## 11.10 A disciplined reward change

Use this sequence:

1. Name the visible failure precisely: “left hip roll remains within 3 degrees
   of its hard limit for 70% of forward steps.”
2. Confirm it from state data, not one frame.
3. Identify the cheapest exploit in the current objective.
4. Add or change one term/gate.
5. Write a pure-function test and a configuration wiring/sign test.
6. Run the five-iteration smoke test.
7. Compare a fixed checkpoint battery with the baseline.
8. Keep or revert based on the intended behavior, not total reward alone.

## 11.11 Lab: produce an evidence-backed run review

Choose one trained checkpoint and write:

```text
Task and checkpoint:
Resolved configuration path:
Main behavior observed:
Failure rate and evaluation cases:
Main weighted reward terms:
All penalty signs valid? yes/no
Episode length interpretation:
Visible exploit or compromise:
Measured state evidence:
Next single hypothesis:
```

Avoid “it works.” Prefer statements such as “tracks +0.2 m/s on 18/20 flat
starts, but falls in 6/20 turn-in-place trials after the first direction
change.” That tells the next engineer what to reproduce.

Continue with
[export, deployment, and sim-to-real](12_deployment_and_sim2real.md).

## 11.12 Folded lab rubric

<details>
<summary>Show a reference evidence-backed review</summary>

An acceptable answer names a reproducible case and separates facts from its
next hypothesis. For example:

```text
Task/checkpoint:       exact task + checkpoint SHA-256
Evaluation battery:   20 fixed flat starts at +0.2 m/s; 20 turn-in-place cases
Observed result:      18/20 forward successes; 14/20 turn successes
Raw failure classes:  4 forward falls after direction reversal; 2 initial slips
Penalty signs:        every weighted penalty <= 0 (attach metric export)
Episode length:       timeouts are success only for continuous walking cases
Visible compromise:  large translation drift while tracking yaw
State evidence:       median/p90 yaw error and x-y drift, synchronized to video
Hypothesis:           turn-in-place samples remain underrepresented
Single next change:   change only its explicit command bucket fraction
```

Those numbers are illustrative. Your report must derive them from the selected
checkpoint. A strong answer also includes representative success/failure video,
resolved configuration, seed list, evaluator commit, and the result that would
falsify the proposed hypothesis.

</details>
