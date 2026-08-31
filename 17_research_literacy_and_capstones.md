# 17. Research Literacy, Reproduction, and Capstone Projects

Being current in reinforcement learning does not mean collecting paper titles.
It means being able to state what was tested, reproduce a meaningful slice,
measure uncertainty, and know which conclusion the evidence does not support.

By the end of this chapter, you should be able to:

- turn an abstract claim into a falsifiable experiment;
- distinguish the training run, evaluation episode, and physical robot as
  different experimental units;
- find confounds in an apparently fair algorithm comparison;
- calculate and interpret seed-level uncertainty without pseudoreplication;
- trace one equation into source code, configuration, and logged evidence;
- preserve a reproduction artifact another person can actually exercise; and
- propose a capstone whose stopping rule and acceptance criteria exist before
  its results.

Research literacy is not habitual skepticism. It is disciplined curiosity:
**what exact observation would make you revise the current explanation?**

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

### Publication status is evidence metadata

A document can be an unreviewed preprint, a revised preprint, a workshop paper,
an accepted conference paper, a journal article, or a withdrawn/corrected work.
Record status and version rather than using “paper” as if all stages were equal.
A later version may change the method, tasks, or conclusions while retaining the
same arXiv identifier.

For fast-moving robot foundation policies, make a dated statement:

```text
Verified 2026-09-01:
- paper version: arXiv v3, dated YYYY-MM-DD
- official repository: organization/project at commit <hash>
- checkpoint: exact model identifier and digest
- evaluation claim used: table/figure/section
- publication status: preprint / venue + year
```

“State of the art” is not a permanent property stored in a model name. It is a
comparison under a protocol at a date. If data, robot, action space, or test
distribution differs, reproduce the comparison before carrying the label into
your project.

### Artifact openness has levels

Do not collapse these statements:

- source code is visible;
- training code runs;
- exact configuration and dependencies are present;
- weights are downloadable;
- training data and mixture are accessible;
- the official evaluator reproduces a reported number; and
- a new team independently obtains a compatible result.

Each is stronger than the previous one in a different way. A code license can
also differ from a base-model or dataset license. Record all three before
calling a policy reusable.

The [Association for Computing Machinery artifact review and badging
policy](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
separates artifact availability, artifact evaluation, and result validation.
This is a useful mental model even when a robotics venue does not award badges.

## 17.2 Read a paper in three passes

### Pass 1: scope

Read title, abstract, figures, evaluation tables, limitations, and conclusion.
Write one sentence:

```text
The paper claims X on task/benchmark Y under protocol Z.
```

If you cannot fill Y and Z, do not repeat X yet.

Add the population and comparator:

```text
Under protocol Z, method X improved metric M over baseline B
on evaluated population Y; the evidence does not yet cover U.
```

The final clause $U$ is the **exclusion boundary**. For example: “The paper
reports manipulation success on two arms; it does not evaluate dynamic biped
balance.” Writing the boundary prevents an impressive adjacent result from
silently becoming evidence for your robot.

### Pass 2: mechanism

Read method and algorithm. Identify:

- inputs available during training and deployment;
- outputs/action representation;
- objective and constraint;
- data collection policy;
- model architecture and memory;
- simulator/hardware and rates; and
- differences from the strongest baseline.

Trace the data flow, not just named blocks:

```text
raw sensor -> preprocessing -> observation schema -> model -> action adapter
           -> safety/controller -> plant -> metric logger
```

Ask which block owns normalization, history, privileged information, action
clipping, and termination. A paper diagram may omit these because they are
implementation details; in sim-to-real work they can determine the result.

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

Now write three sentences:

1. **Support:** “The experiment supports ...”
2. **Non-support:** “It does not establish ...”
3. **Discriminator:** “The next experiment that separates explanations A and B
   is ...”

This forces reading to produce a testable project decision instead of a longer
summary.

### Worked claim dissection

Suppose an abstract says, “Our policy robustly transfers to real robots.” A
research note should unpack it:

| Word | Question needed before reuse |
| --- | --- |
| policy | actor only, estimator plus actor, or full system? |
| robustly | mean success, worst case, intervention, or no falls? |
| transfers | zero policy updates, or calibration and real adaptation? |
| real robots | how many units, revisions, trials, surfaces, and operators? |

If the study used one robot, 20 hand-reset trials, and nominal voltage, a fair
restatement is valuable but narrower. Precision is not hostility to the paper;
it is how its evidence becomes usable.

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

### Quantifiers are part of the claim

Compare:

- “the policy avoided obstacles”;
- “the policy avoided all 12 training layouts”;
- “across 200 preregistered held-out trials, it avoided contact in 184”; and
- “its lower 95% confidence bound exceeded the accepted baseline by 5
  percentage points.”

The first can describe one video. The later statements expose population,
count, uncertainty, and decision rule. Prefer words such as **on the evaluated
conditions**, **for the sampled seeds**, and **within the measured operating
envelope**. Avoid “always,” “safe,” “general,” and “solved” unless the evidence
supports their unusually broad quantifiers.

### Turn a question into variables

“Does history help?” is not yet an experiment. Define:

- **treatment:** observation history length 4 instead of 1;
- **outcomes:** held-out success, falls, energy, and inference time;
- **controlled factors:** simulator, reward, network budget, optimizer, steps,
  seeds, and evaluator;
- **population:** declared command/physics distribution;
- **unit:** independently trained seed, not each timestep;
- **hypothesis:** history improves partial observability enough to offset added
  optimization and latency cost; and
- **decision rule:** a predeclared effect and uncertainty threshold.

Draw a small causal sketch before training:

```text
history length ---> state information ---> success
       |                    |
       +--> model size -----+--> optimization
       +--> inference time ------> stale action ---> success
```

If increasing history also increases parameter count and latency, success
cannot automatically be attributed to memory. Match capacity, measure timing,
or explicitly label those pathways as part of the treatment.

## 17.4 Baselines answer “better than what?”

A useful baseline should be:

- relevant to the task and information setting;
- implemented competently;
- given a comparable tuning/compute budget;
- evaluated with the same metric and held-out cases; and
- simple enough to reveal whether complexity is necessary.

Robot baselines can include:

- zero/random policy for plumbing only;
- hand-designed or proportional–derivative (PD) controller;
- Model Predictive Control (MPC) or state machine;
- behavior cloning;
- standard Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC)
  implementation;
- prior method with authors' configuration; and
- an oracle using privileged information, clearly labeled as an upper bound.

A new reinforcement learning (RL) controller that has not beaten an accepted
scripted balance baseline is not ready for hardware because its training reward
is high.

### Fairness is more than equal environment steps

Algorithm A and B can receive equal environment interactions yet unequal:

- gradient updates and replay reuse;
- network parameters and observation history;
- wall time and graphics processing unit memory;
- expert demonstrations or offline pretraining;
- privileged critic inputs;
- hyperparameter search trials; and
- checkpoint-selection opportunities.

There is no single universally fair budget. Report at least environment steps,
wall-clock time, compute hardware, gradient updates, and external data. Then say
which resource the comparison holds fixed and why that matches the intended
deployment question.

Use two baseline roles:

1. a **sanity baseline** establishes that task wiring and metric are meaningful;
2. a **competitive baseline** asks whether the new method adds value beyond a
   credible existing choice.

An oracle with privileged terrain may expose the cost of partial observability,
but it is not deployable. A hand controller may be deployable but incapable of
the full task. Label each role instead of ordering every method on one axis.

### Baseline integrity checklist

Before accepting a comparison, verify that every method:

- sees the declared observation and no accidental privilege;
- uses the same action bounds and control frequency;
- encounters the same reset, termination, and evaluation seeds;
- has a competent hyperparameter source or tuning allocation;
- is evaluated from a frozen checkpoint, without online test adaptation unless
  that adaptation is the treatment; and
- is allowed to fail visibly rather than having failures filtered from video.

## 17.5 Ablation studies identify causes

An **ablation** removes or changes one component while holding other factors
fixed.

Examples:

- Better Actuator Models (BAM) actuator model on versus ideal actuator;
- observation history 1 versus 4;
- privileged critic on versus off;
- action-rate curriculum versus full penalty from iteration 0;
- calibrated randomization versus arbitrary wide randomization;
- perception confidence supplied versus omitted.

An ablation needs multiple seeds when training is variable. Comparing one
lucky run with one unlucky run does not identify a cause.

### One-at-a-time ablations can miss interactions

Suppose actuator realism $A$ and domain randomization $D$ each have two
levels. A $2\times2$ factorial design measures:

| Actuator realism | Randomization | Mean held-out success |
| --- | --- | ---: |
| off | off | measure |
| on | off | measure |
| off | on | measure |
| on | on | measure |

The effect of realism when randomization is off is

```math
\Delta_{A\mid D=0}=\bar y_{1,0}-\bar y_{0,0},
```

and when randomization is on it is

```math
\Delta_{A\mid D=1}=\bar y_{1,1}-\bar y_{0,1}.
```

Their difference is an interaction:

```math
I_{A,D}=\Delta_{A\mid D=1}-\Delta_{A\mid D=0}.
```

If realism helps only with calibrated randomization, a one-at-a-time study from
the “both off” corner may reach the wrong design conclusion. Factorial studies
cost more; use them for interactions your mechanism actually predicts.

### Mechanistic and deletion ablations differ

Deleting a module asks whether the complete system needs it. A mechanistic
ablation changes the quantity the explanation depends on. If a paper says
history helps because it estimates velocity, useful tests include:

- history length while matching model capacity;
- explicit velocity with no history;
- shuffled or time-reversed history;
- sensor delay swept independently; and
- a probe for velocity plus a control intervention.

These distinguish “more parameters” from “temporal information.” A good
ablation attacks the causal story, not merely a software checkbox.

## 17.6 Random seeds and uncertainty

A seed initializes stochastic components: network weights, environment resets,
commands, minibatch order, and sometimes graphics processing unit (GPU)
kernels. Deep RL can produce meaningfully different outcomes across seeds.

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

### First identify the experimental unit

If one trained policy runs 1,000 evaluation episodes, those episodes estimate
that policy's conditional performance. They are not 1,000 independent training
runs. Treating every episode as an independent algorithm replicate is
**pseudoreplication** and produces falsely narrow uncertainty.

A useful hierarchy is:

```text
algorithm/configuration
  -> independent training seed
       -> checkpoint selected by fixed rule
            -> evaluation scenario
                 -> repeated episode / hardware trial
```

Variation can arise at every level. Report seed-to-seed variability and, for
physical robots, unit/day/operator variation when those are in the intended
population. Ten retries on the same robot in the same marked start pose do not
establish reliability across builds and rooms.

The standard error of a seed mean is

```math
\mathrm{SE}(\bar x)=\frac{s}{\sqrt n},
```

but a normal interval is fragile for few, skewed seeds. It also answers only
about the mean under the sampled training process, not the probability of a
rare catastrophic hardware outcome.

### Pair seeds and scenarios when possible

If methods A and B use the same environment seeds and evaluation scenarios,
analyze paired differences

```math
d_i=x_i^{(A)}-x_i^{(B)}.
```

Common scenario difficulty then cancels. Report the distribution and interval
of $d_i$, not two unrelated error bars. Pairing is valid only when the pairing
is meaningful and does not couple training in a way that changes the method.

For physical tests, construct a balanced trial order so battery temperature,
floor wear, and operator learning do not all favor the method tested second.
Randomize or counterbalance order and log it.

### Success rates are binomial only under assumptions

For $k$ successes in $n$ independent Bernoulli trials,

```math
\hat p=k/n.
```

A Wilson interval behaves better than the simple
$\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}$ near 0, 1, or small $n$. But neither
fixes correlated trials, changing conditions, or post-selected retries. Define
what counts as a trial, success, intervention, and excluded run before testing.

Report numerator and denominator—“18/20”—not only 90%. State whether a human
reset, safety catch, or retry converts an attempt into failure or exclusion.

### Bootstrap confidence interval intuition

To estimate uncertainty without assuming a normal distribution:

1. sample $n$ results with replacement from the observed $n$ results;
2. compute the chosen statistic;
3. repeat many times; and
4. take appropriate percentiles of the bootstrap statistics.

This quantifies sampling uncertainty given the observed runs. With only three
seeds, the interval will be uncertain because the evidence is genuinely weak.

For a paired comparison, resample paired seed/scenario records together. For a
hierarchical robot study, a **stratified bootstrap** may first resample robots
or seeds, then episodes within them. Resampling 10,000 correlated timesteps as
if independent manufactures certainty.

Minimal seed-level code is:

```python
import random

def bootstrap_mean_interval(seed_scores, repeats=20_000, rng_seed=0):
    rng = random.Random(rng_seed)
    n = len(seed_scores)
    estimates = []
    for _ in range(repeats):
        sample = [rng.choice(seed_scores) for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    return estimates[int(0.025 * repeats)], estimates[int(0.975 * repeats)]
```

The percentile bootstrap is pedagogically simple, not universally optimal.
With very few seeds it cannot reveal modes never sampled.

The
[RLiable evaluation paper](https://arxiv.org/abs/2108.13264) explains why
point estimates from a few RL runs can mislead and proposes interval estimates,
performance profiles, and robust aggregate metrics.

The interquartile mean (IQM) sorts scores and averages the middle 50%. It is
less dominated by extreme runs than the mean while using more information than
the median. Across multiple tasks, normalize only with defensible reference
scores; arbitrary normalization can reverse comparisons.

A performance profile asks, for threshold $\tau$, what fraction of task-run
scores exceed it:

```math
F(\tau)=\frac{1}{N}\sum_{i=1}^{N}
\mathbb 1[x_i>\tau].
```

One curve reveals whether improvement is broad or concentrated in easy tasks.
The official `rliable` repository is now archived, so pin its commit if using
it and keep a small independent statistic check in your analysis tests.

### Effect size comes before significance

Ask whether the difference matters physically. A 0.5% mean reward gain may be
statistically detectable yet irrelevant; one fewer fall per 100 missions may
be operationally important. Predeclare a **smallest effect of practical
interest** in task units: success points, centimeters of clearance, joules, or
interventions per hour.

Power calculations require an expected effect and variability. If those are
unknown, run a labelled pilot to estimate logistics, not to make a definitive
claim. Do not repeatedly inspect results and stop when a favorable interval
first appears unless a valid sequential design was declared.

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

### The winner's curse

Suppose every checkpoint has the same true performance $\mu$, but measured
validation score is

```math
\hat J_j=\mu+\epsilon_j.
```

Selecting $j^*=\arg\max_j\hat J_j$ also selects a checkpoint with unusually
positive noise. Its validation score is biased upward even though the individual
measurements were unbiased. More checkpoints, seeds, and hyperparameter trials
create more opportunities to select luck.

The remedy is not to ban checkpoint selection. Define it on validation data,
then estimate the chosen procedure once on a locked test set. The unit being
evaluated is the entire procedure—training, selection, and adaptation—not an
imaginary checkpoint chosen with test knowledge.

### Multiple questions create false discoveries

If you test many metrics, tasks, seeds, and subgroup cuts, some favorable result
can occur by chance. Mark one or a small number of primary outcomes before
running. Treat exploratory findings as hypotheses for a new confirmatory run.
Where formal hypothesis tests are appropriate, disclose the family and apply a
multiple-comparison method; do not hide the unsuccessful comparisons.

Robot learning often benefits more from intervals, effect sizes, failure
clusters, and replication than a ritual threshold on one $p$-value. The key
is that analysis choices cannot secretly depend on which story looks best.

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
- lockfile/container and driver/NVIDIA Compute Unified Device Architecture
  (CUDA) versions;
- exact resolved environment/agent configuration;
- seeds and command lines;
- raw and aggregated metrics;
- checkpoints and model hashes;
- evaluator source/version;
- rollout video and failure labels; and
- hardware/calibration revisions.

The definitions above follow the current Association for Computing Machinery
convention: repeatability is same team/setup, reproducibility is different team
with the same setup/artifacts, and replicability uses an independently developed
setup. State the convention because older literature sometimes swaps the last
two words.

### Provenance is a directed graph

A policy result depends on more than one commit:

```text
source commit + dirty diff + dependency lock + simulator asset revision
          + resolved config + seed + dataset/checkpoint inputs
                                |
                                v
                           training log
                                |
                 checkpoint-selection rule
                                |
                                v
                     exported deployment artifact
                                |
                  evaluator + scenario manifest
                                |
                                v
                         report + raw trials
```

Hash large immutable artifacts and record human-readable identifiers. A hash
proves byte identity, not correctness. Preserve the conversion program that
made an Open Neural Network Exchange artifact, including observation
normalization, because a checkpoint hash alone cannot reconstruct deployment.

### A runnable artifact needs an outer and inner loop

The **outer loop** creates the environment, downloads/verifies assets, and runs
a smoke test. The **inner loop** reproduces one table/figure from existing logs
without retraining. A reviewer with limited compute should be able to exercise
the inner loop and verify artifact consistency before attempting the expensive
outer run.

At minimum include:

```text
README with exact commands and expected output
LICENSES and data/model usage conditions
dependency lock or immutable container recipe
small test fixture and checksum
resolved experiment configurations
raw per-seed/per-trial data
analysis script that regenerates the reported table
known deviations, failures, and compute estimate
```

The [machine-learning reproducibility program
report](https://www.jmlr.org/papers/v22/20-303.html) documents why checklists,
code submission, and independent reproduction are complementary. A completed
checklist is evidence of disclosure, not proof that the result is true.

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

### Claim-to-evidence ledger

Maintain a ledger while reading and experimenting:

| Claim identifier | Evidence | Assumptions | Status | Next discriminator |
| --- | --- | --- | --- | --- |
| C1: history reduces falls | seeds, test set, interval | matched latency | open | delay sweep |
| C2: policy fits target | replay and timing log | normalizer exact | supported | hardware gate |

Use statuses such as `proposed`, `implemented`, `smoke-tested`, `supported in
simulation`, `supported on hardware`, `contradicted`, and `inconclusive`.
“Implemented” is not evidence of performance. “No significant difference” is
not evidence of equivalence unless the interval excludes effects that matter.

### Trace an equation into code

For one central method equation, make a four-column trace:

| Mathematical symbol | Tensor/config | Source location | Logged check |
| --- | --- | --- | --- |
| $\epsilon$ clip | `clip_param` | loss function | clip fraction |
| $\gamma$ | `gamma` | return estimator | return fixture |
| $r_t$ | reward sum | environment manager | per-term mass |
| $a_t$ | actor output | action adapter | min/max/histogram |

Then answer:

1. Does the implementation use the same sign, reduction, and normalization as
   the equation?
2. Is a symbol scheduled or mutated during training?
3. Does the deployment path add filtering, clipping, or coordinate conversion?
4. Which unit test or log would expose a mismatch?

A paper may describe one policy-gradient update while the code additionally
normalizes advantages, clips value loss, schedules learning rate, and limits
gradient norm. Those additions do not automatically invalidate the paper, but
they are part of the reproducible algorithm.

### A cheap reproduction ladder

Do not begin by spending the full paper budget. Climb:

1. **Static audit:** schemas, shapes, signs, dependencies, and config resolution.
2. **Unit analogy:** a tiny script that demonstrates the central equation.
3. **Released evaluation:** official checkpoint with official evaluator.
4. **Smoke train:** short run proving the update and artifacts execute.
5. **Small matched study:** enough seeds to expose gross effects.
6. **Full reproduction:** declared budget and protocol.
7. **Extension:** only after the reproduction boundary is understood.

At each rung define expected output and a stop condition. This saves compute
while making negative results interpretable.

### Rules shared by every capstone

Before the first result exists, commit a short protocol containing:

- question, mechanism, primary outcome, and practical effect threshold;
- baseline, treatment, controlled factors, and known confounds;
- experimental unit, seeds/scenarios/trials, and selection rule;
- compute/hardware budget and early-stop conditions;
- failure/intervention definitions and safety authority;
- artifact paths, licenses, and expected reproducibility level; and
- what result would falsify the favored explanation.

A capstone can pass with a negative result. It cannot pass by changing its goal
after seeing an attractive video.

## 17.10 Capstone A: tabular foundations

Goal: show you understand learning before deep networks.

1. Modify `examples/tabular_q_learning.py` to implement
   State–Action–Reward–State–Action (SARSA).
2. Compare SARSA and Q-learning in a grid with a costly cliff.
3. Sweep $\epsilon$ and three seeds.
4. Plot or tabulate return, cliff entries, and convergence episodes.
5. Explain why on-policy exploration can produce a safer route during
   learning.

Deliverable: two-page report plus runnable code and raw seed results.

**Acceptance gate:** a fresh clone runs both algorithms, regenerates the table,
and shows paired seed results. The discussion must separate learning return from
cliff-entry risk; “SARSA is safer” is too broad if only one exploration schedule
was tested.

## 17.11 Capstone B: PPO mechanism study

Goal: connect the clipped objective to observed optimization.

1. Use `examples/ppo_clip_demo.py` to map advantage sign and probability ratio
   to the surrogate objective.
2. Add cases at ratios 0.5 through 1.5.
3. Predict the flat regions before running.
4. In a small standard control environment, compare two clip values while
   holding seeds and budget fixed.
5. Record approximate Kullback–Leibler (KL) divergence, entropy, return, and
   update stability.

Deliverable: reproduction card explaining what clipping does and does not
guarantee.

**Acceptance gate:** the report predicts the clipped piecewise objective before
showing optimizer results, records all seeds, and does not describe clipping as
a hard Kullback–Leibler constraint. At least one case should show that network
updates can move the aggregate policy farther than an individual sample's
flat surrogate suggests.

## 17.12 Capstone C: reproduce the Microduck pipeline

Goal: demonstrate end-to-end robot-RL understanding without claiming a
five-iteration policy is trained.

1. run central processing unit (CPU) tests and task registry;
2. run a 64-environment, five-iteration smoke train;
3. inspect resolved environment and agent configuration;
4. play zero agent and one available checkpoint;
5. classify the observation, action, command, and reward terms;
6. export via the official normalizer-baking path;
7. inspect Open Neural Network Exchange (ONNX) shape and run CPU deployment
   rehearsal; and
8. state why random-looking viewer direction is command sampling, why obstacles
   are not avoided, and why body-pose intent is not trained by the main recipe.

Deliverable: exact commands, log/artifact paths, one annotated rollout, and a
contract diagram.

**Acceptance gate:** start from a clean dependency sync, run the central
processing unit tests, preserve the resolved configuration, and label the
five-iteration run only as a smoke test. The exported artifact must pass a
one-step observation/action comparison against the simulation path. Record any
missing checkpoint or external credential as a scoped limitation rather than
substituting an unrelated model.

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

**Acceptance gate:** the treatment changes one declared factor or reports an
explicit factorial interaction. Evaluate a frozen selection procedure on
locked conditions, include timing/resource cost, and publish per-seed raw
values. A conclusion such as “no evidence at this budget” is valid; “methods
are equal” requires an equivalence margin and adequate evidence.

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
- export a versioned model with schema and worst-case execution time (WCET);
- perform sim-to-sim, processor-in-loop, bench, tether, then floor gates.

### Phase 4: autonomy

- onboard perception produces confidence-bearing world state;
- local planner/behavior tree selects exact skills;
- cloud agent proposes semantic plans only;
- realtime microcontroller unit (MCU) retains motor safety and network-loss
  response.

Deliverable: signed gate evidence and rollback artifact at every phase. The
project-specific handoff document in the Jump Rover repository contains the
current interfaces and blockers.

**Acceptance gate:** no phase inherits a green status from a presentation or
simulation video. Each requirement points to a dated measurement, test log,
revision, owner, and expiration/retest condition. A failed realtime or
mechanical gate blocks dynamics-sensitive RL but does not block software-only
schema, replay, cloud-timeout, and mission-validation tests.

## 17.15 Threats to validity and research ethics

Before interpreting a result, audit five kinds of validity.

### Internal validity: did the treatment cause the change?

Confounds, inconsistent tuning, changing simulator assets, lucky selection, and
different data can explain an apparent improvement. A controlled ablation,
paired conditions, and preserved provenance strengthen internal validity.

### Construct validity: did the metric represent the goal?

Training reward may rise while the maneuver fails. “Upright” based only on body
height can count a tilted robot. “No collision” can ignore a human safety catch.
Define observable success and failure from the physical intent, then test the
metric against adversarial examples and human-labelled videos.

### External validity: where should the result generalize?

A policy tested on one simulator, robot, floor, lighting setup, or operator has
not established performance outside that population. Name the target
population, sample held-out factors from it, and avoid claiming beyond them.
Generalization to novel textures is not generalization to novel dynamics.

### Statistical-conclusion validity: is the evidence strong enough?

Few seeds, correlated trials, unstable metrics, multiple selection, and
unreported exclusions can make the estimated effect unreliable. Show raw
points, intervals, denominators, selection, and sensitivity to reasonable
aggregation choices.

### Systems validity: was the deployable system evaluated?

A graphics processing unit benchmark with preloaded images does not establish
camera-to-actuator latency. A simulation actor with privileged state does not
establish onboard perception. Evaluate the exported artifact, preprocessing,
normalizer, action adapter, queue, safety governor, and target compute as one
timed path.

### Ethics and safety affect the dataset

Stopping unsafe trials protects people and hardware, but it also censors what
would have happened. Predefine intervention criteria and count every
intervention as an outcome; never delete it as an “incomplete episode.” Obtain
consent and define access/retention for identifiable camera, audio, and operator
data. Report environmental and financial compute cost when it affects who can
reproduce the work.

A **preregistration** is a time-stamped protocol committed before results. It
does not freeze all scientific thought; it separates confirmatory analysis from
later exploration. Amendments are allowed when disclosed with time and reason.

## 17.16 A weekly research-study routine

For one paper per week:

1. Monday: scope pass and claim sentence.
2. Tuesday: derive one central equation in your own words.
3. Wednesday: inspect official code/config corresponding to that equation.
4. Thursday: run one tiny reproduction or dependency-free analogy.
5. Friday: write evidence, limitations, and one project-relevant experiment.

Reading ten abstracts is less useful than tracing one claim through equation,
implementation, configuration, and result.

Add a sixth step: preserve the paper version, repository commit, and dated
claim ledger. On the next week, begin by checking whether a new version changed
the result you relied on.

## 17.17 Final self-assessment

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

## 17.18 Exercises

1. Classify this claim and narrow it: “Our learned controller is robust because
   it succeeds in 47 of 50 trials on one robot and one floor.”
2. One trained policy is evaluated for 500 episodes. Another seed is never
   trained. What is the experimental unit for an algorithm-training claim, and
   what do the 500 episodes estimate?
3. Explain why treating 24 rollout steps from each of 4,096 parallel
   environments as 98,304 independent algorithm replicates is
   pseudoreplication.
4. Paired held-out scores for methods A and B are $A=[0.8,0.6,0.9]$ and
   $B=[0.7,0.65,0.75]$. Compute paired differences and their mean. What does
   three pairs fail to establish?
5. Explain the winner's curse when selecting the best of 100 checkpoints.
6. In a factorial ablation, outcomes for
   $(A,D)=(0,0),(1,0),(0,1),(1,1)$ are $0.40,0.50,0.55,0.80$.
   Compute both conditional effects of $A$ and their interaction.
7. Why should a report say “18/20 successes” rather than only “90% success”?
   Name two trial-dependence problems that still invalidate a binomial model.
8. Trace Proximal Policy Optimization clip parameter $\epsilon$ from equation
   to configuration, source computation, and one logged diagnostic.
9. Method A and B use equal environment steps, but B performs ten times as many
   gradient updates and uses demonstrations. Is the comparison invalid? Give a
   precise way to report it.
10. Draft a primary outcome and falsification rule for the claim “adding a
    depth map teaches Microduck obstacle avoidance.”
11. A repository releases a checkpoint but omits observation normalization and
    the evaluator. Which openness claims are justified, and what cannot yet be
    reproduced?
12. A new 2026 preprint claims state-of-the-art vision-language-action control.
    List the minimum checks before adding that claim to a JumpRover decision.

Continue with the [detailed paper seminars](18_detailed_paper_seminars.md), then
use [Chapter 20](20_glossary_and_worked_problems.md) for terminology and worked
checks before returning to the capstone whose deliverable you cannot yet
produce.

## 17.19 Folded capstone reference checks

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
safe behavior under cloud timeout, stale perception, and invalid skill
identifiers (IDs) for
autonomy. A cloud-generated plan is never evidence that the real-time motor
safety layer works. Every gate must name its artifact and rollback version.

</details>

## 17.20 Folded exercise solutions

<details>
<summary>Show worked answers to Section 17.18</summary>

1. It is a benchmark/hardware empirical claim on a narrow evaluated population,
   not general robustness. A precise version is: “On one robot revision and one
   floor under the stated 50-trial protocol, the controller succeeded in
   47/50 attempts.” Report intervention/failure definitions and an interval;
   robustness to other robots, surfaces, voltage, and disturbances remains open.

2. The independently trained seed is the unit for variation in the training
   procedure. The 500 episodes estimate that one frozen policy's conditional
   performance over sampled evaluation cases. They improve precision about that
   policy but do not measure how often retraining produces a good policy.

3. Parallel steps share one policy, optimizer history, simulator code, reward,
   and often correlated initializations. They are samples used to make one
   update, not independent re-runs of the algorithm. Seed-level conclusions
   require independently initialized training procedures.

4. Paired differences $A-B$ are $[0.10,-0.05,0.15]$, with mean
   $(0.10-0.05+0.15)/3\approx0.0667$. Three pairs show the observed direction
   but provide weak uncertainty coverage and cannot establish robustness to
   unsampled training seeds, scenarios, or physical conditions.

5. Every validation score contains noise. Taking the maximum selects both good
   underlying performance and unusually favorable noise, so the selected
   validation score overestimates future performance. Select by a fixed
   validation rule, then evaluate the chosen procedure once on locked test data.

6. With $D=0$, the effect of $A$ is $0.50-0.40=0.10$. With $D=1$, it
   is $0.80-0.55=0.25$. The interaction is $0.25-0.10=0.15$: actuator
   realism has a larger observed effect when randomization is on.

7. The fraction exposes sample size; 90% could mean 9/10 or 900/1,000 and those
   have different uncertainty. Dependence arises if repeated trials share an
   overheated battery/robot or if nearly identical reset scenes make outcomes
   correlated. Adaptive retries and changing conditions are additional issues.

8. In the equation, $\epsilon$ defines the ratio clip range
   $[1-\epsilon,1+\epsilon]$. In configuration it may be `clip_param`; trace
   that field into the surrogate-loss `clamp`/`clip` call. Log clip fraction and
   approximate Kullback–Leibler divergence to check whether updates frequently
   hit the flat region and how far the aggregate policy moves.

9. It is a valid comparison of two complete procedures at equal interaction
   budget if the extra updates and demonstrations are disclosed as part of B.
   It is not a clean claim that the algorithmic update alone caused the gain.
   Report environment steps, gradient updates, wall time, compute, and external
   demonstration count; add matched-data/update ablations if that causal claim
   matters.

10. One primary outcome could be collision-free goal completion over a locked
    set of unseen obstacle geometries, with all attempts and interventions
    counted. Falsify the favored mechanism if the depth policy does not improve
    paired collision-free completion over the planner/velocity baseline by the
    predeclared practical margin, or if improvement disappears when texture is
    changed while geometry is fixed. Also measure age and stopping clearance.

11. It justifies “checkpoint bytes are available,” subject to working access
    and license. It does not establish an exercisable deployment artifact,
    exact inference behavior, or reproduction of reported metrics. Recover the
    normalizer/action adapter and official evaluation protocol before comparison.

12. Verify paper version/status/date; official author repository and commit;
    code, weights, data mixture, evaluator, and licenses; observation/action
    schema and embodiments; baselines, task splits, trials/seeds, uncertainty,
    and tuning budget; target compute latency/memory; and whether the reported
    protocol overlaps JumpRover's task. Label an unreproduced preprint claim as
    provisional and compare it with a small matched baseline.

</details>
