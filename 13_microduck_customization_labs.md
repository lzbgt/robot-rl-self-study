# 13. Customization Labs

These labs turn the book into a project course. Work on a branch, keep one
question per experiment, and preserve tests/configuration with every result.

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

## Lab 1: change a command distribution

Question: does more turn-in-place data improve turning without hurting forward
walking?

The current setting is:

```python
TURN_IN_PLACE_FRACTION = 0.15
```

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

Before keeping the reward, answer:

- Is the target measured from the actual current robot model?
- Is height meaningful in every task phase?
- Could a fallen pose match the height?
- Does this term block motion the task physically requires?
- How large is its weighted mass relative to the positive objective?

This example teaches wiring; it is not a recommendation to add a height
penalty to the main walking task.

## Lab 3: design a new task from a template

Choose the closest family:

| Intended behavior | Starting point |
| --- | --- |
| locomotion | velocity |
| episodic trick ending in a pose | stand-up |
| commanded two-state transition | sit-stand |
| dynamic maneuver | roulade |

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
<summary>Show the minimum acceptable result for Labs 0–7</summary>

- **Lab 0:** a pinned commit/config/checkpoint reproduces the baseline evaluator
  within declared stochastic uncertainty. A video alone is insufficient.
- **Lab 1:** the resolved command distribution changes exactly one intended
  bucket/range; exact zero remains explicitly sampled; a configuration test
  locks the values.
- **Lab 2:** the helper's sign and boundary behavior pass as pure tensor tests,
  its configuration weight has the intended sign, and every logged weighted
  penalty remains nonpositive.
- **Lab 3:** the new task inherits the closest proven template, preserves the
  61D/14D deployment family contract when intended, resolves all joint/sensor
  selectors, and passes a five-iteration export smoke.
- **Lab 4:** the stage changes only after measured competence, uses environment
  steps correctly, and does not create a repeatable metric cliff at its
  boundary.
- **Lab 5:** either the modular planner proves local stop/clearance/freshness
  while retaining 61D, or a new exteroceptive schema is explicitly versioned;
  simply rendering obstacles is rejected.
- **Lab 6:** nominal actuator parameters are fit on one trace and validated on
  held-out traces; randomization represents residual uncertainty rather than a
  guess.
- **Lab 7:** a machine-readable per-case result, aggregate with uncertainty,
  failure taxonomy, and sampled videos can be regenerated from one command.

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

Adapt this to Torch and repository conventions; it illustrates separation of
the mathematical value from simulator-manager data extraction.

</details>
