# 17. Research Literacy, Reproduction, and Capstone Projects

Being current in reinforcement learning does not mean collecting paper titles.
It means being able to state what was tested, reproduce a meaningful slice,
measure uncertainty, and know which conclusion the evidence does not support.

## 17.1 A source hierarchy

Prefer sources in this order:

1. peer-reviewed paper or clearly versioned preprint;
2. official project page and repository from the authors;
3. released configuration, code, checkpoints, data, and evaluation scripts;
4. independent reproductions that disclose deviations; and
5. summaries, talks, or social posts only as pointers.

An arXiv identifier proves a document exists; it does not prove peer review,
correctness, reproducibility, or relevance to your robot. An open repository
proves some artifact is visible; it does not prove it matches the paper result.

## 17.2 Read a paper in three passes

### Pass 1: scope

Read title, abstract, figures, evaluation tables, limitations, and conclusion.
Write one sentence:

```text
The paper claims X on task/benchmark Y under protocol Z.
```

If you cannot fill Y and Z, do not repeat X yet.

### Pass 2: mechanism

Read method and algorithm. Identify:

- inputs available during training and deployment;
- outputs/action representation;
- objective and constraint;
- data collection policy;
- model architecture and memory;
- simulator/hardware and rates; and
- differences from the strongest baseline.

### Pass 3: evidence

Read experiment details, appendices, and repository configuration. Record:

- tasks and split;
- environment steps, data size, and compute;
- number of seeds and hardware trials;
- metric aggregation and uncertainty;
- hyperparameter tuning budget;
- ablations;
- failure examples; and
- artifact availability.

## 17.3 Claim taxonomy

Separate these claim types:

- **theoretical**: follows from stated assumptions and proof;
- **algorithmic**: mechanism is defined and implemented;
- **benchmark empirical**: result holds on evaluated benchmark/protocol;
- **sim-to-sim**: transfers between simulators;
- **sim-to-real**: evaluated on stated hardware conditions;
- **generalization**: evaluated on a specified held-out distribution;
- **scaling**: behavior changes with data/model/compute range;
- **systems**: latency, throughput, reliability, or resource result.

A benchmark empirical claim is not a theoretical guarantee. A ten-minute
hardware video is not a reliability distribution. “Zero-shot sim-to-real” means
no real-world policy fine-tuning under that setup; it does not mean no hardware
calibration, system identification, or engineering.

## 17.4 Baselines answer “better than what?”

A useful baseline should be:

- relevant to the task and information setting;
- implemented competently;
- given a comparable tuning/compute budget;
- evaluated with the same metric and held-out cases; and
- simple enough to reveal whether complexity is necessary.

Robot baselines can include:

- zero/random policy for plumbing only;
- hand-designed or PD controller;
- MPC or state machine;
- behavior cloning;
- standard PPO/SAC implementation;
- prior method with authors' configuration; and
- an oracle using privileged information, clearly labeled as an upper bound.

A new RL controller that has not beaten an accepted scripted balance baseline
is not ready for hardware because its training reward is high.

## 17.5 Ablation studies identify causes

An **ablation** removes or changes one component while holding other factors
fixed.

Examples:

- BAM actuator model on versus ideal actuator;
- observation history 1 versus 4;
- privileged critic on versus off;
- action-rate curriculum versus full penalty from iteration 0;
- calibrated randomization versus arbitrary wide randomization;
- perception confidence supplied versus omitted.

An ablation needs multiple seeds when training is variable. Comparing one
lucky run with one unlucky run does not identify a cause.

## 17.6 Random seeds and uncertainty

A seed initializes stochastic components: network weights, environment resets,
commands, minibatch order, and sometimes GPU kernels. Deep RL can produce
meaningfully different outcomes across seeds.

For returns $x_1,\ldots,x_n$, the sample mean is

```math
\bar{x}=\frac{1}{n}\sum_i x_i.
```

The sample standard deviation is

```math
s=\sqrt{\frac{1}{n-1}\sum_i(x_i-\bar{x})^2}.
```

Neither says everything about skewed failures. Also report median, quantiles,
success rate, and named failure categories.

### Bootstrap confidence interval intuition

To estimate uncertainty without assuming a normal distribution:

1. sample $n$ results with replacement from the observed $n$ results;
2. compute the chosen statistic;
3. repeat many times; and
4. take appropriate percentiles of the bootstrap statistics.

This quantifies sampling uncertainty given the observed runs. With only three
seeds, the interval will be uncertain because the evidence is genuinely weak.

The
[RLiable evaluation paper](https://arxiv.org/abs/2108.13264) explains why
point estimates from a few RL runs can mislead and proposes interval estimates,
performance profiles, and robust aggregate metrics.

## 17.7 Best checkpoint bias

If you evaluate 100 checkpoints on the same test conditions and report only the
best, you have tuned on the test set. Similarly, choosing the best of many seeds
and hiding the rest estimates luck, not expected performance.

Define selection before final testing:

```text
training set: optimizer updates
validation set: checkpoint/hyperparameter selection
test set: one final locked evaluation
hardware acceptance: separately defined safety/performance protocol
```

## 17.8 Reproducibility levels

### Repeatability

Same team, code, data, environment, and procedure obtains consistent results.

### Reproducibility

Another team uses provided artifacts/procedure and obtains a compatible result.

### Replicability

Another team independently implements the idea and obtains evidence supporting
the claim.

Terminology varies by field, so define what you mean. In all cases preserve:

- source commit and dirty diff;
- lockfile/container and driver/CUDA versions;
- exact resolved environment/agent configuration;
- seeds and command lines;
- raw and aggregated metrics;
- checkpoints and model hashes;
- evaluator source/version;
- rollout video and failure labels; and
- hardware/calibration revisions.

The classic
[Deep Reinforcement Learning That Matters](https://arxiv.org/abs/1709.06560)
paper documents sensitivity to implementation and experimental choices. The
lesson remains current: algorithm labels alone do not define an experiment.

## 17.9 Reproduction card template

```markdown
# Reproduction: <paper/result>

## Claim being tested
One scoped sentence.

## Primary sources
Paper version, official repository commit/release, model/data identifiers.

## Original protocol
Task, observations, actions, data, algorithm, compute, seeds, metrics.

## Our deviations
Every hardware/software/config difference and expected consequence.

## Success criterion chosen before running
Numerical metric, uncertainty, and qualitative failure boundary.

## Baselines and ablations
What is held fixed and what changes.

## Results
All seeds/trials, not only best; confidence intervals and failure categories.

## Interpretation
What evidence supports, does not support, and next discriminating experiment.

## Artifacts
Commits, lockfiles, commands, configs, logs, checkpoints, videos, hashes.
```

## 17.10 Capstone A: tabular foundations

Goal: show you understand learning before deep networks.

1. Modify `examples/tabular_q_learning.py` to implement SARSA.
2. Compare SARSA and Q-learning in a grid with a costly cliff.
3. Sweep $\epsilon$ and three seeds.
4. Plot or tabulate return, cliff entries, and convergence episodes.
5. Explain why on-policy exploration can produce a safer route during
   learning.

Deliverable: two-page report plus runnable code and raw seed results.

## 17.11 Capstone B: PPO mechanism study

Goal: connect the clipped objective to observed optimization.

1. Use `examples/ppo_clip_demo.py` to map advantage sign and probability ratio
   to the surrogate objective.
2. Add cases at ratios 0.5 through 1.5.
3. Predict the flat regions before running.
4. In a small standard control environment, compare two clip values while
   holding seeds and budget fixed.
5. Record approximate KL, entropy, return, and update stability.

Deliverable: reproduction card explaining what clipping does and does not
guarantee.

## 17.12 Capstone C: reproduce the Microduck pipeline

Goal: demonstrate end-to-end robot-RL understanding without claiming a
five-iteration policy is trained.

1. run CPU tests and task registry;
2. run a 64-environment, five-iteration smoke train;
3. inspect resolved environment and agent configuration;
4. play zero agent and one available checkpoint;
5. classify the observation, action, command, and reward terms;
6. export via the official normalizer-baking path;
7. inspect ONNX shape and run CPU deployment rehearsal; and
8. state why random-looking viewer direction is command sampling, why obstacles
   are not avoided, and why body-pose intent is not trained by the main recipe.

Deliverable: exact commands, log/artifact paths, one annotated rollout, and a
contract diagram.

## 17.13 Capstone D: modern locomotion ablation

Choose one:

- actuator calibration versus randomization width;
- privileged critic versus symmetric actor/critic information;
- observation history versus feed-forward policy;
- PPO versus SAC at matched environment steps and wall time; or
- existing planner + velocity policy versus perceptive end-to-end policy.

Before running:

1. identify one primary paper and official codebase;
2. freeze the baseline contract;
3. define at least three training seeds;
4. define held-out physics/commands;
5. define human-visible failure categories; and
6. estimate compute and stopping rule.

Deliverable: a mini-paper with negative results preserved.

## 17.14 Capstone E: Jump Rover staged handoff

This capstone starts with evidence, not RL installation.

### Phase 1: hardware truth

- finish and measure mechanics;
- validate motor/servo sign, range, current, and thermal behavior;
- validate realtime watchdog, limits, stop, and communication;
- accept a tethered classical/scripted balance/drive baseline.

### Phase 2: digital twin

- record mass, inertia, geometry, contacts, actuator response, delay;
- create simulator and four-action environment;
- replay hardware excitation in simulation;
- define held-out identification trials.

### Phase 3: learning

- train bounded residual or command policy in simulation;
- compare against the frozen baseline;
- export a versioned model with schema and WCET;
- perform sim-to-sim, processor-in-loop, bench, tether, then floor gates.

### Phase 4: autonomy

- onboard perception produces confidence-bearing world state;
- local planner/behavior tree selects exact skills;
- cloud agent proposes semantic plans only;
- realtime MCU retains motor safety and network-loss response.

Deliverable: signed gate evidence and rollback artifact at every phase. The
project-specific handoff document in the Jump Rover repository contains the
current interfaces and blockers.

## 17.15 A weekly research-study routine

For one paper per week:

1. Monday: scope pass and claim sentence.
2. Tuesday: derive one central equation in your own words.
3. Wednesday: inspect official code/config corresponding to that equation.
4. Thursday: run one tiny reproduction or dependency-free analogy.
5. Friday: write evidence, limitations, and one project-relevant experiment.

Reading ten abstracts is less useful than tracing one claim through equation,
implementation, configuration, and result.

## 17.16 Final self-assessment

You are ready to begin independent robot-RL research when you can:

- reject an impossible task definition before training;
- derive a Bellman or policy-gradient update and implement a small version;
- justify algorithm choice from interaction/data constraints;
- separate simulator truth, actor observation, critic privilege, and runtime
  sensor data;
- design baselines and ablations before seeing results;
- report uncertainty and failure clusters;
- preserve a deployable observation/action/timing contract; and
- say “the evidence does not establish that” without losing enthusiasm for the
  next experiment.

Continue with the [detailed paper seminars](18_detailed_paper_seminars.md), then
use [Chapter 20](20_glossary_and_worked_problems.md) for terminology and worked
checks before returning to the capstone whose deliverable you cannot yet
produce.

## 17.17 Folded capstone reference checks

<details>
<summary>Show reference mechanisms and completion criteria for Capstones A–E</summary>

These are not single “correct” experiment results. They are reference checks
that make an incomplete or unsupported submission visible.

**Capstone A — SARSA.** The on-policy target uses the action actually selected
by the exploratory policy at the next state:

```math
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha
\left[r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\right].
```

A minimal update helper is:

```python
def sarsa_update(q, s, a, reward, next_s, next_a, alpha, gamma):
    target = reward + gamma * q[next_s][next_a]
    q[s][a] += alpha * (target - q[s][a])
```

Q-learning instead uses `max(q[next_s])`. Near a costly cliff, SARSA learns
about the consequences of its continuing exploration, so it can prefer a
route with more clearance while training. Completion requires all requested
seeds and cliff-entry counts; one attractive trajectory is not the result.

**Capstone B — PPO clipping.** For positive advantage, the surrogate becomes
flat above ratio $1+\epsilon$; for negative advantage it becomes flat below
$1-\epsilon$. Clipping the objective does not impose a hard bound on the final
network update because all samples share parameters and the optimizer takes
multiple minibatch steps. A complete report shows the predicted piecewise
curve, measured approximate KL and entropy, both clip settings under the same
budget, and every seed.

**Capstone C — Microduck pipeline.** A passing submission distinguishes a
smoke test from learned locomotion, records the actor input as 61 values and
the action as 14 values, uses the official export path that bakes observation
normalization into ONNX, and rehearses that artifact on CPU. It explicitly
states that the main actor has no obstacle sensor or global-goal input and that
its body-pose reward has zero weight. Missing any one of configuration,
checkpoint, normalizer, command ordering, or timing leaves the deployment
contract unproved.

**Capstone D — locomotion ablation.** The causal comparison changes exactly one
named factor, holds training and evaluation budgets fixed, uses at least three
seeds, and reports held-out conditions plus failure categories. If PPO and SAC
receive different wall time, replay warm-up, observations, or reward code,
label those as confounds rather than attributing the entire difference to the
algorithm name. A negative result with complete artifacts passes; a best-seed
claim without uncertainty does not.

**Capstone E — Jump Rover.** Each phase needs a measurable exit gate. Examples
are a recorded watchdog stop bound and thermal envelope for hardware truth;
held-out excitation error for the digital twin; matched-baseline success and
processor-in-loop worst-case execution time for learning; and demonstrated
safe behavior under cloud timeout, stale perception, and invalid skill IDs for
autonomy. A cloud-generated plan is never evidence that the real-time motor
safety layer works. Every gate must name its artifact and rollback version.

</details>
