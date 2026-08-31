# 11. Training, Evaluation, and Debugging

Training is not “start Proximal Policy Optimization (PPO) and wait.” A useful
run is a controlled experiment with a hypothesis, resolved configuration,
checkpoints, per-term metrics, and rollouts that test the intended behavior.

Three processes must remain separate:

```text
optimization: use training experience to change parameters
selection:    use validation cases to choose a checkpoint/configuration
assessment:   use untouched test cases to estimate final capability
```

If the same ten trials are repeatedly watched while rewards and checkpoints
are tuned, those trials have become validation data. Reporting them as an
unseen test exaggerates confidence even though no supervised labels were used.

Deep reinforcement learning (RL) is unusually variable because data depends on
the changing policy, resets and disturbances are random, and nonlinear
optimization amplifies early differences. The evaluation methods in this
chapter build on
[Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
and its
[`rliable` reference implementation](https://github.com/google-research/rliable).
The repository was archived in 2025, but the paper's lessons—intervals,
performance profiles, robust aggregate metrics, and paired comparisons—remain
valuable. Robot evaluation adds command slices, physical failure taxonomies,
and explicit safety tails.

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

Do not combine reward redesign, new observations, stronger domain randomization
(DR), a new robot model, and new PPO hyperparameters in one experiment. Even a
successful result will not tell you which change mattered.

Think in terms of variables:

- the **treatment** is the one intended change;
- controlled variables include commit, task, simulator, training budget, and
  evaluation cases;
- nuisance variables include learner seed, reset randomness, device timing,
  and nondeterministic kernels; and
- outcomes are predeclared metrics and failure classes.

For a paired seed design, baseline and treatment use the same seed set:

```text
baseline: 101, 202, 303, 404, 505
treatment: 101, 202, 303, 404, 505
```

Analyze per-seed differences. Pairing does not make five seeds “large,” but it
removes some variation caused by comparing unrelated initializations. Never
discard a failed seed because its curve is ugly unless the predeclared rule
identifies an infrastructure-invalid run and is applied equally to both arms.

Training budget should be expressed in at least three units:

```math
N_{transition}=N_{env}N_{step}N_{iteration},
```

wall-clock time, and compute/device type. Equal iterations with different
environment counts are not equal sample budgets; equal transitions on devices
with different speed are not equal wall budgets. State which comparison the
hypothesis requires.

## 11.3 Read the training dashboard in layers

### Layer 1: pipeline health

Check:

- iteration advances;
- simulation and learning frames per second (FPS) are plausible;
- losses and Kullback–Leibler (KL) divergence are finite;
- no not-a-number (NaN) termination spike appears;
- checkpoints continue to save; and
- graphics processing unit (GPU) memory remains stable.

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

The Kullback–Leibler (KL) divergence estimates how far the updated policy moved
from the behavior policy that collected the rollout. A small value does not
prove learning, and a large value does not identify whether the environment is
correct; it only diagnoses the optimization step. Read it beside clipping
fraction, value loss, entropy, gradient norm, and task performance.

Learning curves are temporally correlated: checkpoint 1,001 descends from
checkpoint 1,000 and is not an independent replicate. A moving average can
make a plot readable, but it does not create more independent evidence. Keep
raw data, state the smoothing window, and compare independent seeds for
uncertainty.

When an anomaly begins, locate the first iteration rather than inspecting only
the final corrupted state. Save:

```text
last known-good checkpoint and metrics
first bad checkpoint and metrics
active curriculum stage
first nonfinite tensor/term if relevant
representative rollout immediately before and after
```

This brackets the causal event and makes a reduced reproduction possible.

## 11.4 Weighted logs and reward mass

`Episode_Reward/<term>` is the weighted contribution. If a term's weight is
zero, its log is zero even when the underlying behavior is poor. If a
curriculum changes the weight, a jump in the log may reflect the coefficient,
not behavior.

Compare reward mass:

```math
M_i=\mathbb{E}[w_i f_i],
\qquad
A_i=\mathbb{E}[|w_i f_i|].
```

$M_i$ is signed mean contribution; $A_i$ is absolute activity. Cancellation
can make $M_i$ small even when a signed shaping term is very active, so both
are useful. A rough relative share is

```math
S_i=\frac{A_i}{\sum_j A_j+\epsilon}.
```

This is a diagnostic, not a causal attribution: changing one term changes the
policy and therefore every $f_j$. Still, it catches orders-of-magnitude mistakes
that raw weights hide.

Suppose one task has 8 points/step of positive reward and another has 2. An
action-rate weight of `-0.1` is four times weaker relative to the first task,
even though the text of the configuration is identical.

Log unweighted features or behavior metrics when a curriculum can set a term's
weight to zero. Otherwise the dashboard erases the measurement exactly when it
would reveal whether the policy is ready for that term to turn on.

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

### 11.5.1 Define the sampling hierarchy

Do not confuse four sample counts:

```text
training seeds
  checkpoints within a seed
    evaluation scenarios/commands
      repeated episodes within a scenario
```

Twenty episodes from one checkpoint measure conditional rollout variability;
they are not twenty independently trained policies. For a claim about the
algorithm or reward recipe, the independent unit is normally the training
seed. For a claim about one chosen deployable checkpoint, repeated physical
conditions may be the relevant unit, but the claim must stay checkpoint-specific.

Use a table with one row per raw episode, not only an aggregate dashboard:

```text
policy_hash, train_seed, checkpoint_iteration, scenario_id,
eval_seed, command, reset_class, disturbance, success,
tracking_error, fall_time, energy_proxy, failure_class, video_id
```

This structure permits regrouping without rerunning and prevents a mean from
hiding which command failed.

### 11.5.2 Continuous metrics

For command $c_t$ and measured velocity $v_t$, useful tracking metrics include

```math
\mathrm{MAE}=\frac{1}{T}\sum_{t=1}^{T}|v_t-c_t|,
\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{T}\sum_{t=1}^{T}(v_t-c_t)^2}.
```

Mean absolute error (MAE) is easy to interpret and less dominated by spikes.
Root mean squared error (RMSE) emphasizes large errors. Report both only when
each answers a stated question; a dozen redundant metrics increases the chance
of selecting a flattering one after seeing results.

Robot-specific measures can include:

```math
\mathrm{drift}=\lVert p_T-p_0-\text{desired displacement}\rVert_2,
```

```math
E_{mechanical}=\sum_{t,j}|\tau_{t,j}\dot q_{t,j}|\Delta t,
```

plus peak tilt, slip distance, impact impulse, settling time, action variation,
and deadline misses. $E_{mechanical}$ is an absolute joint-work proxy; real
battery energy additionally depends on voltage/current, regeneration, motor
loss, and electronics.

Average metrics can conceal unsafe tails. Report a high quantile such as the
95th percentile tracking error and a worst-slice result. For a “larger is
better” score $X$, lower-tail conditional value at risk at fraction $\alpha$
is the mean of the worst $\alpha$ fraction:

```math
\mathrm{CVaR}^{lower}_{\alpha}(X)
=\mathbb{E}[X\mid X\le q_{\alpha}],
```

where $q_\alpha$ is the $\alpha$ quantile. Conditional value at risk (CVaR)
makes repeated near-failures visible even when mean tracking is strong.

### 11.5.3 Binary success and uncertainty

If $k$ of $n$ trials succeed, the estimate is $\hat p=k/n$. The familiar
standard error

```math
\sqrt{\frac{\hat p(1-\hat p)}{n}}
```

is useful intuition but a normal interval behaves badly near zero/one and at
small $n$. The Wilson interval is a better elementary default. With standard
normal critical value $z$, its center and half-width are

```math
c=\frac{\hat p+z^2/(2n)}{1+z^2/n},
```

```math
h=\frac{z}{1+z^2/n}
\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}.
```

Report $[c-h,c+h]$. Eighteen successes in 20 trials means 90%, but the
approximate 95% Wilson interval is about 70%–97%. This is far less certainty
than the point estimate visually suggests. Zero failures is also not proof of
zero risk: with 20/20 successes the lower Wilson bound is only about 84%.

For robot safety, a success label should be accompanied by failure classes and
severity. One gentle stop and one high-energy face impact are both “failures”
in a binomial count but demand different engineering responses.

### 11.5.4 Means, medians, interquartile means, and bootstrap intervals

The arithmetic mean uses every value but can be dominated by rare extreme
runs. The median is robust but discards much rank information in small samples.
The interquartile mean (IQM) averages the middle 50% after sorting, combining
some robustness with better sample efficiency than the median on multi-task
benchmarks:

```math
\mathrm{IQM}(x)=\mathrm{mean}\{x_i:q_{0.25}\le x_i\le q_{0.75}\}.
```

Do not mechanically replace every robot metric with IQM. For one homogeneous
command slice, show all seed values, mean/median, and an interval. IQM is most
useful when aggregating comparable normalized scores over multiple tasks or
conditions.

A nonparametric bootstrap approximates estimator uncertainty:

1. sample the independent units with replacement;
2. compute the statistic on that resample;
3. repeat many times; and
4. take suitable percentiles of the bootstrap statistics.

If episodes are nested within training seeds, resample seeds as clusters or use
a hierarchical bootstrap. Resampling every episode as independent produces an
interval that is too narrow because all episodes from one checkpoint share the
same learned policy.

For baseline $A$ and treatment $B$ trained with matching seeds, bootstrap the
paired differences $d_s=B_s-A_s$. The probability of improvement estimate

```math
\widehat{P}(B>A)=\frac{1}{S}\sum_{s=1}^{S}\mathbf{1}[B_s>A_s]
```

is intuitive, but with few seeds its uncertainty must also be shown.

Run the standard-library example:

```bash
python examples/evaluation_statistics.py
```

It computes mean, median, IQM, a seeded bootstrap interval, a paired
probability of improvement, and a Wilson success interval. Alter one outlier
and observe which summaries move.

### 11.5.5 Stratify before aggregating

An overall score answers performance under a particular mixture of cases. If
90% of trials are easy forward motion and 10% are turn-in-place, the overall
mean mostly answers the forward question. Preserve named slices:

```text
command bucket x reset class x terrain x disturbance x hardware condition
```

Report per-slice sample count and uncertainty. Then, if one deployment-weighted
aggregate is needed, publish its weights:

```math
J_{deploy}=\sum_k \omega_k J_k,
\qquad \sum_k\omega_k=1.
```

The worst credible slice can be more important than $J_{deploy}$ for safety.
Test values inside the training support, near its boundaries, and just outside
it. This distinguishes interpolation, edge robustness, and extrapolation; the
word “robust” should not combine them.

Random evaluation and adversarial search are complementary. Random trials
estimate performance under a declared distribution. A grid or optimization
over push direction, timing, friction, delay, and command can locate failure
boundaries. Once found and used for tuning, those cases become validation
cases; retain a fresh final assessment set.

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

Synchronize video frames with state/action logs using a shared simulation step
or timestamp. A useful review panel places commanded/measured twist, trunk
tilt, contacts, action extrema, reward terms, and curriculum state under the
video. Then “the foot slipped here” becomes a queryable event rather than a
memory of one playback.

Create a failure taxonomy before counting:

```text
F0 no failure
F1 initial-state collapse
F2 tracking loss without fall
F3 slip then fall
F4 self-collision
F5 joint/action saturation
F6 numerical/contact-solver failure
F7 deadline or stale-command safety stop
```

Keep raw notes so classes can be refined, but freeze the taxonomy before the
final test. Separate policy behavior (`F2`) from invalid simulation (`F6`) and
runtime infrastructure (`F7`); pooling them obscures which subsystem needs a
change.

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

Selecting the maximum of many noisy checkpoint estimates creates a winner's
curse. If 100 equivalent checkpoints are each evaluated on five noisy trials,
the best observed one is likely lucky. A disciplined protocol is:

1. define a validation battery and selection rule, such as lowest validation
   tracking error subject to zero severe failures;
2. choose once using that battery;
3. lock the checkpoint content hash; and
4. evaluate it on a larger untouched test battery.

When training several seeds, decide whether the deployed artifact is the best
validation seed or a representative fixed-seed recipe. Report algorithm-level
seed distribution separately from the selected-checkpoint test. They answer
different questions.

Learning performance also has a time axis. Compare curves at matched
transition budgets, not just final return. The area under a learning curve up
to budget $B$,

```math
\mathrm{AUC}(B)=\int_0^B J(n)\,dn,
```

summarizes sample efficiency but depends on metric scaling and interpolation.
Always show the curve and final performance beside the area under the curve
(AUC), because a fast weak plateau and a slow strong result can have similar
areas.

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
prefer a fixed final target with progress/landing shaping and let reinforcement
learning (RL) discover the path.

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

In-sim playback applies the checkpoint normalizer and can hide an Open Neural
Network Exchange (ONNX) export mistake. Runtime all-zero commands can select
“stand” when an application expected a trick flag. These interface failures
look like bad learning.

### Debug with competing hypotheses

Do not stop at the first plausible story. For “turning policy falls,” write
alternatives and discriminating evidence:

| Hypothesis | Prediction | Cheap test |
| --- | --- | --- |
| turn commands were too rare | forward strong, turn slice weak across seeds | command-count and sliced evaluation |
| yaw observation sign is wrong | response consistently turns opposite | named-state observation/action probe |
| slip penalty blocks pivoting | applied feet remain planted while target yaw rises | contact/slip/action trace and weight ablation |
| actuator saturation | target error and voltage/torque limit co-occur | saturation telemetry |
| checkpoint mismatch | metrics cannot reproduce saved video | artifact hashes and manifest replay |

Prefer a test whose outcomes separate hypotheses. Increasing training time
tests none of these cleanly: it may mask data scarcity, intensify a reward
exploit, or simply waste budget.

For numerical failures, bisect along two dimensions:

- **time**: find the first bad iteration/step; and
- **complexity**: reduce worlds, terrain, events, sensors, or reward terms while
  preserving the failure.

Remove one causal branch at a time. If disabling a reward hides a NaN only
because it no longer reads a bad sensor, the origin may still be the sensor;
trace the first nonfinite value rather than the final stack frame.

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

## 11.11 Exercises and evidence lab

1. A study trains 2,048 worlds for 3,000 iterations with 24 rollout steps.
   Calculate transitions. What additional quantities are needed to compare its
   resource budget with another study?
2. A penalty helper returns a nonpositive value, has weight `-0.2`, and its
   logged weighted contribution is positive. Diagnose the error.
3. Term A has signed mean contribution `+1.5` and absolute activity `1.5`.
   Term B has signed mean `0` but absolute activity `2.0`. What can and cannot
   be concluded?
4. Why do 100 episodes from one training seed not equal 100 independent
   algorithm runs? Which question can those episodes still answer well?
5. Using $z=1.96$, calculate or approximate the Wilson interval for 18
   successes in 20 trials. Interpret it in plain language.
6. Baseline seed scores are `[0.4, 0.7, 0.5, 0.8, 0.2]`; treatment scores are
   `[0.5, 0.6, 0.7, 0.9, 0.6]` in matching seed order. Compute paired
   differences and the observed probability of improvement.
7. Explain why selecting the best of 100 checkpoints on five trials and
   reporting those same five trials is biased. Propose a repair.
8. Design strata for evaluating idle, forward, turning, and push recovery.
   Give one continuous metric, one binary criterion, and one failure class for
   each.
9. A mean tracking score improves while worst-10% score falls sharply. What
   engineering conclusion is justified? What is not?
10. A smoothed learning curve has 1,000 plotted points. How many independent
    training replicates does that imply?
11. Construct three competing hypotheses for a policy that walks in the viewer
    but fails after ONNX export. Give one discriminating test for each.
12. When should a numerical-failure run be excluded from a comparison, and what
    must be reported?
13. Run `python examples/evaluation_statistics.py`, change one score to an
    extreme outlier, and explain the response of mean, median, and IQM.
14. Choose one trained checkpoint and produce this report:

    ```text
    task, commit, resolved config, checkpoint hash:
    selection rule and validation battery:
    untouched test battery and raw-data path:
    train seeds and evaluation seeds:
    per-slice sample counts:
    main estimates with uncertainty:
    severe failures and synchronized videos:
    reward signs and main reward masses:
    observed capability boundary:
    next hypothesis and falsifying result:
    ```

Avoid “it works.” Prefer “tracks +0.2 m/s on 18/20 flat starts, but falls in
6/20 turn-in-place trials after the first direction change.” The second form
states a population, condition, numerator, denominator, and failure trigger.

## 11.12 Folded solutions and report rubric

<details>
<summary>Show solutions to Exercises 1–7</summary>

1. The count is
   $2048\times3000\times24=147{,}456{,}000$ transitions. Also report device,
   wall time, power/accounting assumptions if relevant, simulator and policy
   rates, environment count, update/minibatch schedule, and whether both studies
   count the same kind of transition. Equal transitions do not imply equal wall
   time or optimization updates.
2. Multiplying a nonpositive helper by a negative weight double-negates it into
   reward. Use a positive weight for that helper or rewrite it as a nonnegative
   cost with a negative weight. The episode contribution should then be
   nonpositive on violations.
3. A contributes positive reward consistently on average. B is active but its
   signed values cancel; it may be potential shaping or vary across cases. B's
   larger absolute activity means it strongly changes instantaneous returns,
   but neither statistic alone proves causal influence on the learned policy.
4. All 100 episodes share one learned parameter vector and training history;
   they estimate rollout/reset variability conditional on that checkpoint.
   They can estimate that checkpoint's success probability under a declared
   deployment scenario well, but not variability of the training algorithm.
5. Substitution gives center about 0.838 and half-width about 0.134, hence
   approximately `[0.70, 0.97]`. “18/20” is compatible with a much lower true
   success rate than 90%; more independent trials are needed for a narrow
   operational guarantee.
6. Differences are `[+0.1, -0.1, +0.2, +0.1, +0.4]`. Four of five are
   positive, so the observed probability of improvement is $4/5=0.8$. With
   only five pairs the uncertainty is large; show all values or a paired
   interval rather than claiming an 80% universal probability.
7. Noise helps one of many checkpoints win, and reusing the same five trials
   conditions the report on that luck. Select with a fixed validation battery,
   lock the checkpoint hash, then assess once on a larger untouched test
   battery. Report the selection procedure and number of candidates.

</details>

<details>
<summary>Show solutions to Exercises 8–13</summary>

8. One valid design is:

   | Slice | Continuous | Binary | Failure class |
   | --- | --- | --- | --- |
   | idle | action variation/head bias | stays upright 20 s | jitter-induced slip |
   | forward | velocity MAE/drift | reaches timeout within error bound | forward fall |
   | turn | yaw MAE/translation drift | completes commanded angle | pivot slip |
   | push | peak tilt/settling time | recovers without fall | unrecovered fall |

   Thresholds, command values, pushes, horizon, and sample counts must be fixed
   before final evaluation.
9. The change improved central/average behavior while worsening tail behavior
   under the tested distribution. That is a tradeoff requiring severity and
   slice investigation; it does not justify “better” or “safer” without an
   application-defined utility/constraint.
10. None. The 1,000 points may come from one correlated trajectory through
    parameter space. Independent replicate count is the number of independently
    trained seeds, not checkpoints or smoothing samples.
11. Examples: missing normalizer predicts mismatched normalized features—compare
    training-side and ONNX outputs on fixed observations. Wrong action order
    predicts named joints respond to the wrong indices—run one-hot probes.
    Wrong command-slot writes predict a correct policy response to unintended
    intention—log and compare the complete 61-vector. Artifact mismatch is a
    fourth hypothesis tested by hashes.
12. Exclude only under a predeclared, treatment-independent infrastructure rule,
    such as a confirmed machine outage before training began. Algorithmic NaNs
    are outcomes, not inconvenient data. Report every exclusion, raw count,
    reason, affected arm/seed, evidence, and analysis with failures included
    when possible.
13. The mean moves directly with the outlier. The median often does not move
    until rank order crosses the center. IQM drops the outer quarters and is
    robust if the changed score remains outside the middle half. This robustness
    is useful, but a catastrophic robot failure must still be reported rather
    than trimmed out as statistical inconvenience.

</details>

<details>
<summary>Show a reference evidence-backed review rubric</summary>

An acceptable answer names a reproducible case, reports uncertainty, and
separates observations from the next hypothesis. For example:

```text
Task/checkpoint:       exact task + Secure Hash Algorithm 256-bit (SHA-256)
                       checkpoint digest
Selection:             predeclared validation rule and candidate count
Test battery:          20 untouched flat +0.2 m/s; 20 turn-in-place cases
Observed result:       18/20 forward; 14/20 turn, with Wilson intervals
Raw failure classes:  4 turn falls after reversal; 2 initial slips
Penalty signs:        every weighted penalty <= 0 (attach metric export)
Episode length:       timeout is success only for continuous-walking cases
Visible compromise:  large translation drift while tracking yaw
State evidence:       median/p90 yaw error and drift, synchronized to video
Capability boundary: tested commands/terrain only; no obstacle claim
Hypothesis:           turn-in-place samples remain underrepresented
Single next change:   change only its explicit command-bucket fraction
Falsifier:            paired turn metric/tail does not improve across seeds
```

Those numbers are illustrative. A strong submission includes raw episode rows,
representative success and failure video, resolved configuration, seed list,
evaluator commit, slice definitions, and the result that would falsify the
proposed hypothesis.

</details>

Continue with
[export, deployment, and sim-to-real](12_deployment_and_sim2real.md).
