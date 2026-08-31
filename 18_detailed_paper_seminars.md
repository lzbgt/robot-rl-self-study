# 18. Detailed Paper Seminars: Impactful Robot-Intelligence Research

This chapter is a guided reading course, not a list of fashionable titles. The
selected works introduced ideas that materially shaped open robot-learning
practice, provided unusually useful experimental evidence, or opened a major
current direction. They span locomotion, world models, large-scale real-robot
reinforcement learning (RL), imitation, cross-embodiment data, foundation
policies, and language-guided planning.

“Most impactful” is necessarily a judgment. The selection criteria are:

- the work changed how important robot-learning problems are formulated;
- the central mechanism can be explained and tested;
- the evidence is substantial and scoped clearly;
- a primary paper and preferably official artifacts exist; and
- the idea teaches something transferable to Microduck, Jump Rover, or another
  real robot.

For each seminar, read the paper itself after this guide. This chapter explains
the map; it does not replace methods, appendices, or code.

## 18.1 Seminar 1 — Massively parallel Proximal Policy Optimization (PPO) for locomotion

### Primary work

- [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning (2021)](https://arxiv.org/abs/2109.11978)
- [Official `legged_gym`](https://github.com/leggedrobotics/legged_gym)
- [Official Robotic Systems Lab reinforcement learning
  (RSL-RL)](https://github.com/leggedrobotics/rsl_rl)

### The problem before the paper

Legged locomotion policies could take hours or days to train, which made reward,
curriculum, and dynamics experiments slow. Physics simulation was often split
between central processing unit (CPU) environments while neural-network
updates ran on a graphics processing unit (GPU), creating data movement and
throughput bottlenecks.

### Central idea

Run thousands of physics environments in parallel on one GPU and collect short
on-policy fragments from all of them:

```text
environment 1:     o0 a0 r0 ... o23 a23 r23
environment 2:     o0 a0 r0 ... o23 a23 r23
...
environment 4096:  o0 a0 r0 ... o23 a23 r23
                              |
                              v
                    one large PPO batch
```

The time horizon per environment can be short because the aggregate batch is
large. For Microduck:

```math
4096\times24=98,304\text{ transitions per iteration}.
```

This changes wall-clock economics. It becomes reasonable to discard stale PPO
data because almost 100,000 fresh transitions arrive each iteration.

### Curriculum insight

The work also uses a terrain curriculum framed like a game: environments move
toward harder terrain when the policy makes sufficient progress and toward
easier terrain when it fails. This concentrates samples near the current
competence frontier.

The general principle is not “always increase difficulty.” It is:

```math
\text{sample where success is neither nearly zero nor nearly certain}.
```

If every rollout fails instantly, the learning signal cannot distinguish useful
partial behavior. If every rollout succeeds, it supplies little pressure to
improve robustness.

### What the evidence establishes

The paper reports dramatic wall-clock training speed in its ANYmal/Isaac Gym
setup, analyzes components in the massively parallel regime, and demonstrates
sim-to-real locomotion. The open stack influenced a broad family of legged-robot
projects.

### What it does not establish

- PPO is always more sample efficient than off-policy methods.
- Any 4,096-environment recipe fits a graphics processing unit (GPU) with
  8 gigabytes (GB) of memory.
- A fast training curve transfers without actuator/system identification.
- Reward and curriculum values copy across robot size or morphology.

### Code-reading route

In RSL-RL, trace:

```text
runner -> rollout storage -> returns/advantages -> PPO minibatches
       -> actor/critic update -> new rollout
```

In Microduck, match this to `num_steps_per_env=24`, environment count, PPO
epochs, and minibatches. Calculate samples and optimizer exposures before
launching a run.

### Reproduction exercise

Run the same small locomotion task at 64, 256, and the largest feasible
environment count. Keep total environment transitions approximately fixed.
Report:

- simulator frames/second;
- GPU memory;
- wall time;
- policy updates;
- return across at least three seeds; and
- whether optimization behavior changes with batch size.

This tests the throughput insight without pretending to reproduce ANYmal.

## 18.2 Seminar 2 — Rapid Motor Adaptation

### Primary work

- [Rapid Motor Adaptation (RMA) for Legged Robots (2021)](https://arxiv.org/abs/2107.04034)
- [Official project page](https://ashish-kmr.github.io/rma-legged-robots/)

### The problem

A robust policy trained under domain randomization must act reasonably across
many dynamics. But the best action on ice differs from the best action on high
friction. If the policy can infer the current condition, it can adapt rather
than compromise.

The true environment parameter vector might contain friction, payload, motor
strength, and terrain compliance:

```math
e_t=[\mu, m_{payload}, k_{motor}, \ldots].
```

Those quantities are easy to read from simulation but usually unavailable on
the real robot.

### Architecture

RMA trains a base policy with a compact environment encoding:

```math
a_t=\pi(o_t,z_t).
```

During a privileged training phase, an encoder can derive $z_t$ from simulator
environment information. A deployment adaptation module instead predicts it
from recent observation/action history:

```math
\hat z_t=g(o_{t-H:t},a_{t-H:t-1}).
```

The history contains the consequence of prior actions. For example, the same
motor command produces different acceleration under different payload or
friction.

### Why a latent instead of explicit parameter regression?

The policy needs a control-useful summary, not necessarily an accurate human
parameter report. Several physical changes can have similar short-term effects,
and some are not separately identifiable from proprioception. A learned latent
can represent equivalence classes relevant to action.

### Training logic

The idea is teacher-student distillation across information sets:

```text
simulation parameters -> privileged encoder -> target latent
history                -> adaptation module -> predicted latent
                                           |
observation + latent -> base policy -> action
```

The deployment module learns to reproduce a latent whose use was already made
valuable by the base policy.

### Evidence and impact

The paper evaluates simulation-trained RMA on a Unitree A1 over diverse real
terrains without real-world policy fine-tuning. The architecture popularized
fast history-based adaptation as a practical alternative/complement to fixed
robustness.

### Limitations and audit questions

- Adaptation can only infer properties that affect observed history.
- The first moments after reset have insufficient history.
- Latent prediction can lag a sudden surface change.
- A training distribution missing a real failure mode cannot confer magic
  adaptation.
- History length, encoder latency, and sensor bias affect deployment.

### Microduck experiment

Randomize one identifiable factor, such as actuator strength. Train:

1. a feed-forward robust baseline;
2. the same actor with observation history; and
3. a privileged-latent teacher plus history student.

Evaluate on held-out fixed strengths and a within-episode strength change. Plot
tracking error versus time after the change. This distinguishes average
robustness from adaptation speed.

## 18.3 Seminar 3 — Dreamer and physical robot world models

### Primary works

- [DayDreamer: World Models for Physical Robot Learning (2022)](https://arxiv.org/abs/2206.14176)
- [DreamerV3: Mastering Diverse Domains through World Models (2023)](https://arxiv.org/abs/2301.04104)
- [Official DreamerV3 code](https://github.com/danijar/dreamerv3)

### The problem

Model-free RL may need many environment interactions. That is tolerable in a
fast simulator but costly on physical hardware. A learned world model can use
each transition to predict many aspects of future experience, then train a
policy through imagined rollouts.

### Recurrent state-space model

Dreamer-family methods use a compact latent state containing a deterministic
memory and a stochastic component. At a high level:

```math
h_t=f(h_{t-1},z_{t-1},a_{t-1}),
```

```math
z_t\sim q(z_t\mid h_t,o_t).
```

- $h_t$ carries recurrent history;
- $z_t$ represents uncertain current information;
- posterior $q$ incorporates the real observation.

The model learns to predict observations/features, rewards, and continuation.
A prior predicts $z_t$ without seeing $o_t$, which is what imagined rollouts
need.

### Model-learning objective in words

The objective balances:

1. reconstruct/predict information from observations;
2. predict reward and whether the episode continues; and
3. make the predictive prior agree with the observation-informed posterior
   without collapsing the representation.

The divergence term is commonly based on Kullback–Leibler (KL) divergence. Here
it asks how different two latent distributions are—not whether the robot's
physical path is close in meters.

### Imagination

Starting from a posterior latent inferred from real data:

```text
latent now -> actor samples action -> world model predicts latent next
           -> predicted reward/value -> repeat in imagination
```

Actor and critic learn from these compact imagined trajectories. No future
pixels must be rendered during policy learning.

### DayDreamer evidence

DayDreamer applies the approach online to four physical robot settings,
including legged recovery/walking, visual manipulation, and visual navigation.
Its importance is showing that world-model learning can operate directly on
real robot experience rather than only simulated/game benchmarks.

### DreamerV3 evidence

DreamerV3 emphasizes a single configuration across more than 150 evaluated
tasks and introduces scale-robust transformations/normalization. It provides
strong general evidence for the family, not proof that one model captures every
contact-rich robot accurately.

### Failure analysis

Inspect separately:

- reconstruction/prediction quality;
- reward prediction calibration;
- continuation/termination prediction;
- latent rollout error versus horizon;
- imagined policy success versus real rollout success; and
- uncertainty under out-of-distribution actions.

A visually plausible prediction can have the wrong contact timing, while an
unrecognizable decoded image might still retain adequate control information.

### Project exercise

Before training a world model on a robot, fit a one-step predictor to logged
proprioception. Evaluate one-step, 5-step, and 25-step open-loop errors on held-
out action sequences. Compare against a persistence baseline
$\hat s_{t+1}=s_t$. This establishes whether learning dynamics adds signal
before adding actor optimization.

## 18.4 Seminar 4 — Temporal Difference Learning for Model Predictive Control, second generation (TD-MPC2)

### Primary work

- [TD-MPC2: Scalable, Robust World Models for Continuous Control (2023)](https://arxiv.org/abs/2310.16828)
- [Official code](https://github.com/nicklashansen/tdmpc2)

### How it differs from Dreamer

Both learn latent dynamics, but TD-MPC2 emphasizes local trajectory
optimization at decision time. It learns an encoder, latent dynamics, reward,
value functions, and a policy prior. Candidate action sequences are refined and
scored in latent space.

Conceptually, for horizon $H$:

```math
\text{score}(a_{t:t+H-1})=
\sum_{k=0}^{H-1}\gamma^k\hat r(z_{t+k},a_{t+k})
+\gamma^H\hat V(z_{t+H}).
```

The learned value estimates what happens after the short explicit planning
horizon. This is a common model-based pattern: model near-term consequences,
bootstrap the distant future.

### Planning loop

1. encode current observation/history into latent state;
2. initialize a population of action sequences, helped by the learned policy;
3. roll sequences through latent dynamics;
4. score them with predicted reward and terminal value;
5. update the proposal distribution toward high-scoring sequences;
6. execute only the first action; and
7. observe and replan.

### Evidence

The paper reports one set of hyperparameters across 104 online RL tasks in four
domains and investigates scaling a multi-task agent. This supports the method's
breadth within that suite.

### Engineering tradeoff

A feed-forward PPO actor performs one network pass. TD-MPC2 performs several
model/value evaluations over candidate sequences per action. Compare:

```math
\text{data efficiency} \quad\text{versus}\quad
\text{runtime planning cost and model risk}.
```

For a 20 Hz arm, planning may fit comfortably. For a 100 Hz balance loop on a
small processor, measure worst-case execution before choosing it.

### Reproduction exercise

Use the official code on one supported benchmark. Then halve and double the
planning horizon while keeping training checkpoint fixed. Measure return and
inference latency. Explain why a longer horizon can either help foresight or
hurt through compounded model error and missed deadline.

## 18.5 Seminar 5 — QT-Opt and large-scale real-world RL

### Primary work

- [QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation (2018)](https://arxiv.org/abs/1806.10293)

QT-Opt is the paper's method name, not a full term that the authors expand
letter by letter. It combines learned Q-values with numerical action
optimization, which is the mechanism worth remembering.

### Why this older paper remains important

QT-Opt is an influential counterexample to the idea that robot policies must
always learn in simulation or by imitation. It scales off-policy Q-learning to
hundreds of thousands of real grasp attempts and closed-loop visual control.

### Task formulation

The observation is camera-based; the continuous action describes gripper motion
and gripper command; sparse success indicates a completed grasp. The learned
Q-function estimates long-horizon success probability/value from observation
and action.

Continuous action prevents enumerating $\arg\max_aQ(s,a)$. QT-Opt uses the
Cross-Entropy Method (CEM), a sampling optimizer:

1. sample action candidates from a distribution;
2. evaluate each with Q;
3. retain an elite high-value fraction;
4. refit the action distribution to elites; and
5. repeat, then execute a high-value action.

This makes Q-learning possible without a separate deterministic actor, at the
cost of several Q evaluations per control decision.

### Distributed data and training

The system separates robot collection, replay storage, distributed training,
and deployment. Off-policy learning lets old grasp attempts remain useful as
new policies collect better data.

### Evidence

The paper reports use of more than 580,000 real grasp attempts and high success
on its unseen-object evaluation. It documents learned closed-loop behaviors
such as regrasping and object repositioning that were not individually
programmed.

### Limitations and modern reading

- The data/robot fleet budget is far beyond a hobby project.
- Grasp success is narrower than general manipulation.
- The exact camera/action distribution defines generalization.
- A percentage on unseen objects does not describe every failure severity.
- CEM inference has a compute/latency cost.

The lasting lesson is a systems one: replay-based RL can turn a growing robot
fleet log into policy improvement when task reward is reliable and collection
is automated.

### Small-scale exercise

In simulation, compare a discrete grid over a 2D continuous action with CEM
optimization of the same learned/known Q surface. Measure solution quality
versus Q evaluations. This isolates action optimization from the cost of
training a deep visual Q-function.

## 18.6 Seminar 6 — Action Chunking with Transformers (ACT) and Diffusion Policy

### Primary works

- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (2023)](https://arxiv.org/abs/2304.13705)
- [Official ACT code](https://github.com/tonyzhaozh/act)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (2023)](https://arxiv.org/abs/2303.04137)
- [Official Diffusion Policy code](https://github.com/real-stanford/diffusion_policy)

### Shared problem: demonstrations are temporally and multimodally structured

A one-step squared-error behavior-cloning policy assumes a single average
action label. Manipulation demonstrations often have:

- temporally coordinated submotions;
- several valid approaches;
- pauses and contact transitions;
- human timing variability; and
- errors that compound when predicting one action at a time.

Both ACT and Diffusion Policy predict **action chunks**, but represent their
distribution differently.

### ACT architecture

ACT uses a conditional variational autoencoder (CVAE) with a transformer-style
policy. During training, an encoder sees the demonstration action sequence and
infers a latent style $z$. The policy/decoder predicts an action chunk
conditioned on observation and $z$.

The CVAE objective has two parts:

```math
L=L_{action}+\beta D_{KL}
\bigl(q_\phi(z\mid o,a_{t:t+H})\|p(z)\bigr).
```

- $L_{action}$ makes predicted chunks match demonstrations;
- KL regularization makes the training latent distribution compatible with a
  simple prior available at deployment;
- $\beta$ controls the tradeoff.

ACT also temporally ensembles overlapping chunk predictions. If several past
inferences predicted the action for the current time, the controller combines
them with age-dependent weights. This can smooth transitions without making
the policy completely open loop.

### Diffusion Policy architecture

Diffusion starts with a noisy action trajectory $x_K$ and repeatedly predicts
how to remove noise conditioned on observation:

```math
x_{k-1}=\text{denoise}_\theta(x_k,o,k)+\text{scheduled noise}.
```

Training corrupts real action sequences with known Gaussian noise and asks the
network to predict the noise (or an equivalent denoising target):

```math
L(\theta)=
\mathbb{E}_{x_0,k,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_k,o,k)\|^2\right].
```

This is supervised generative modeling. It can retain distinct modes because
sampling begins from different noise rather than forcing one mean action.

At deployment, receding-horizon control predicts a sequence but executes only
a prefix before observing again.

### Evidence

ACT reports fine-grained low-cost bimanual tasks with relatively small
demonstration sets. Diffusion Policy reports results across 12 tasks from four
manipulation benchmarks and real-robot evaluations. Each supports action
sequence modeling in its task family.

### Comparison

| Question | ACT | Diffusion Policy |
| --- | --- | --- |
| action distribution | latent-variable decoder | iterative generative denoising |
| inference passes | typically one policy decode | several denoising steps |
| temporal mechanism | chunks + temporal ensembling | chunks + receding horizon |
| strength | efficient fine-grained imitation | expressive multimodal trajectories |
| concern | latent use/collapse, chunk tuning | inference latency, denoising schedule |

### What to reproduce

On one official simulated task:

1. train a one-step behavior cloning (BC) baseline;
2. train action chunks at two horizons;
3. hold demonstration split and image encoder fixed;
4. report task success and correction after an injected perturbation;
5. measure inference latency; and
6. visualize predicted action sequences, not only final success.

The research question is whether temporal/multimodal representation explains
the improvement—not whether one method has a newer name.

## 18.7 Seminar 7 — Robotics Transformer 1 (RT-1) and Open X-Embodiment

### Primary works

- [RT-1: Robotics Transformer for Real-World Control at Scale (2022)](https://arxiv.org/abs/2212.06817)
- [Open X-Embodiment: Robotic Learning Datasets and Robotics Transformer X (RT-X) Models (2023)](https://arxiv.org/abs/2310.08864)
- [Official Open X-Embodiment repository](https://github.com/google-deepmind/open_x_embodiment)

### RT-1 question: does robot policy performance scale with diverse data?

RT-1 studies a high-capacity transformer policy trained on a large real-robot
dataset covering many language-described tasks. Images are converted into
tokens; language conditions the network; actions are discretized into tokens.

An action dimension $a^j$ within range $[l_j,u_j]$ can be placed into one of
$B$ bins. The model predicts a categorical token rather than a raw floating
number. De-tokenization maps the selected bin back to a command.

This turns policy learning into sequence classification:

```math
L=-\sum_t\sum_j \log p_\theta
(\text{action-token}_{t,j}\mid I_{\le t},\text{instruction}).
```

The model is trained by imitation. It does not learn from a scalar reward in
the PPO sense.

### TokenLearner and efficiency

Full image patches across time create many tokens. RT-1 uses a learned spatial
token reduction so the transformer processes a smaller set of image features.
The design highlights a recurring systems problem: a robot policy needs enough
visual context while meeting inference rate.

### RT-1 evidence

The work studies data/model scaling and generalization on its Everyday Robots
fleet/task setup. Its importance is the large, diverse real-robot policy study,
not a claim that tokenized actions dominate continuous outputs everywhere.

### Open X-Embodiment question: can data transfer across robots?

Robot datasets differ in cameras, arms, grippers, action frames, rates, and
task language. Open X-Embodiment assembled standardized data from 22 robot
embodiments across 21 institutions, covering 527 skills (reported as 160,266
tasks/variations in the paper's terminology).

The central hypothesis is positive transfer:

```math
\text{target robot performance with multi-robot pretraining}
>
\text{target-only training at comparable target data}.
```

RT-X models adapt the RT-1 and Robotics Transformer 2 (RT-2) policy families
to the combined data.

### The hard part is normalization, not file concatenation

Cross-embodiment training must reconcile:

- observation names, image views, and missing sensors;
- absolute versus delta actions;
- joint versus end-effector coordinates;
- action dimension and gripper semantics;
- frequency and temporal horizon;
- language/task taxonomy; and
- dataset imbalance.

A common action space may discard embodiment-specific capability. A very loose
space may prevent shared learning. This is an interface research problem.

### Evidence and limitations

The Open X paper reports positive transfer for multiple evaluated robots and
makes standardized datasets/models available. But 22 embodiments do not
uniformly cover all robot types: dataset volume and task diversity are
imbalanced, mostly toward manipulation.

### Microduck relevance

The main lesson is not to feed manipulation actions into a biped. It is to
version a policy-family schema so related Microduck skills can share useful
representations. Microduck's stable 61D actor contract enables hot swapping;
multi-task pretraining would still need task labels, compatible actions, and
balanced sampling.

### Reproduction exercise

Choose two compatible datasets from Open X. Write a schema table before
training. Compare target-only BC with joint pretraining plus target fine-tuning,
matching target examples. Report per-task transfer, including negative
transfer. A global average can hide one task getting worse.

## 18.8 Seminar 8 — From RT-2 to open vision-language-action (VLA) and flow policies

### Primary works and artifacts

- [RT-2 (2023)](https://arxiv.org/abs/2307.15818)
- [Octo (2024)](https://arxiv.org/abs/2405.12213) and [official code](https://github.com/octo-models/octo)
- [OpenVLA (2024)](https://arxiv.org/abs/2406.09246) and [official code](https://github.com/openvla/openvla)
- [$\pi_0$ (2024)](https://arxiv.org/abs/2410.24164) and [official `openpi`](https://github.com/Physical-Intelligence/openpi)
- [$\pi_{0.5}$ (2025)](https://arxiv.org/abs/2504.16054)
- [Generalist Robot 00 Technology (GR00T) N1 (2025)](https://arxiv.org/abs/2503.14734) and [official code](https://github.com/NVIDIA/Isaac-GR00T)
- [SmolVLA (2025)](https://arxiv.org/abs/2506.01844) and [LeRobot](https://github.com/huggingface/lerobot)

This is a fast-moving area. The purpose is to understand design axes, not name
a permanent winner.

### RT-2: action tokens inside a vision-language model

RT-2 co-fine-tunes a pretrained vision-language model on web vision-language
tasks and robot trajectories. Robot actions are expressed as tokens in the same
autoregressive vocabulary as text.

The hoped-for transfer is:

```text
web vision/language knowledge (objects, concepts, relations)
                         +
robot action demonstrations (grounded motor behavior)
                         |
                         v
language-conditioned visual action tokens
```

The paper reports thousands of evaluation trials and improved semantic
generalization in its robot setup. It demonstrates that web-scale semantic
pretraining can influence robot action selection. It does not show that web
text teaches unobserved robot dynamics.

### Octo: reusable open policy initialization

Octo is trained on 800,000 trajectories from Open X-Embodiment and is designed
to accept different observation/task modalities and adapt to new action spaces.
Its transformer uses token groups and readout tokens so downstream users can
attach suitable action heads.

Octo's scientific value includes open checkpoints/code and ablations on
architecture/data choices. It frames a generalist policy as an initialization
to fine-tune, not a universal zero-shot hardware controller.

### OpenVLA: an open vision-language model (VLM)-to-action recipe

OpenVLA starts from a pretrained language backbone and visual encoders, then
trains on 970,000 Open X robot trajectories. It predicts discretized action
tokens autoregressively. The 7-billion-parameter model makes modern VLA study
possible outside the organizations that developed closed RT-2 models.

Parameter-efficient fine-tuning such as Low-Rank Adaptation (LoRA) updates
low-rank matrices rather than every full weight matrix:

```math
W'=W+BA,
```

where $A$ and $B$ have much smaller rank than $W$. This reduces trainable
parameters, but not necessarily base-model inference memory or latency.

### $\pi_0$: continuous action chunks with flow matching

$\pi_0$ combines a pretrained vision-language backbone with an action expert
that generates continuous action chunks using flow matching. Instead of
classifying each action into a discrete token, it learns a velocity field that
transforms noise into an action trajectory.

In a simplified conditional flow-matching view, interpolate between noise
$x_0$ and data action $x_1$:

```math
x_\tau=(1-\tau)x_0+\tau x_1.
```

Train a vector field $v_\theta(x_\tau,\tau,c)$, conditioned on vision/language
$c$, to predict the direction toward data. At inference, numerically integrate
the learned field from noise toward a structured action chunk.

The official `openpi` repository makes a particularly valuable systems lesson
visible: model code is only one part. Data transforms, image masks, prompt
tokenization, action padding, normalization statistics, embodiment config,
checkpoint assets, inference server, and robot adapter define actual behavior.

### $\pi_{0.5}$: heterogeneous co-training for open-world tasks

$\pi_{0.5}$ adds co-training on heterogeneous examples: robot actions,
high-level semantic subtask prediction, object detections, multi-robot data, and
web knowledge. The paper evaluates long-horizon manipulation in new homes.

The design suggests that low-level action data alone may not teach robust
semantic decomposition. Mixing high-level prediction examples gives the model
intermediate supervision. It also complicates attribution: ablations are needed
to say which data source produces which capability.

### GR00T N1: dual-system VLA for humanoid manipulation

GR00T N1 explicitly describes a dual system:

```text
System 2: vision-language reasoning/understanding
                         |
System 1: diffusion-transformer continuous action generation
```

Training mixes real robot trajectories, human video, and synthetic data. The
paper reports simulation benchmarks across embodiments and real humanoid
bimanual tasks. “Humanoid foundation model” here centers primarily on
language-conditioned manipulation; it should not be read as proof of autonomous
locomotion safety.

### SmolVLA: efficiency and asynchronous inference

SmolVLA studies a smaller, community-oriented model intended for affordable
hardware. It also separates slower perception/action prediction from action
execution using asynchronous inference.

Asynchrony introduces a control question: the predicted chunk may be based on
an older observation. Measure

```math
\text{observation age at action execution}
=\text{capture} + \text{queue} + \text{inference} + \text{transport delay}.
```

Smooth throughput is not enough if stale actions meet a disturbed robot.

### Comparing VLAs responsibly

Record:

| Axis | Questions |
| --- | --- |
| pretraining | web data, robot hours/trajectories, embodiments, licenses? |
| action | tokens, regression, diffusion/flow, horizon, frame, rate? |
| adaptation | full fine-tune, LoRA, new head, frozen backbone? |
| evaluation | same tasks, splits, cameras, demonstrations, trials? |
| runtime | parameters, memory, average/worst-case execution time (WCET), asynchronous age? |
| openness | code, weights, data, exact training recipe, robot adapter? |

Never compare one paper's percentage directly to another without protocol
alignment.

### Reproduction exercise

Use one official open VLA on a supported simulation or dataset. First run its
released checkpoint/evaluator. Then fine-tune on a deliberately small target
subset. Compare:

- from-scratch BC;
- frozen backbone + new action head;
- parameter-efficient fine-tuning; and
- full fine-tuning if compute permits.

Evaluate in-distribution and on one held-out object/layout. Report model
memory, inference time, observation age, and task success.

## 18.9 Seminar 9 — SayCan, Code as Policies, and VoxPoser

### Primary works

- [SayCan: Do As I Can, Not As I Say (2022)](https://arxiv.org/abs/2204.01691)
- [Code as Policies (2022)](https://arxiv.org/abs/2209.07753)
- [Official Code as Policies code](https://github.com/google-research/google-research/tree/master/code_as_policies)
- [VoxPoser (2023)](https://arxiv.org/abs/2307.05973)
- [Official VoxPoser code](https://github.com/huangwl18/VoxPoser)

These works focus on high-level semantic planning/grounding. They complement,
rather than replace, motor RL.

### SayCan: combine usefulness with affordance

A large language model (LLM) can score which skill description is
linguistically useful for an instruction. A learned value/affordance function
estimates whether the robot can successfully execute that skill in the current
situation.

For candidate skill $k$:

```math
\text{score}(k)
\propto
p_{LM}(k\mid\text{instruction, history})
\times
p_{afford}(\text{success}\mid s,k).
```

The product embodies “Say” times “Can.” A sponge-related skill may be useful
for cleaning, but its affordance should be low if no sponge is reachable.

This is a powerful architecture lesson: semantic plausibility and physical
feasibility are different estimators. Exact skill choices constrain the LLM's
output space.

### Code as Policies: language models compose application programming interfaces (APIs)

Instead of selecting one listed skill, a code-capable LLM writes programs that
call perception and control APIs. Programs can express loops, geometry,
conditions, and reusable functions.

The strength is compositionality and inspectable structure. The danger is that
generated code has real authority. A safe implementation needs:

- a restricted language/API;
- static validation and resource limits;
- no arbitrary shell/network/device access;
- typed physical units and bounded arguments;
- simulation/dry run;
- runtime monitor and timeout; and
- signed/approved primitives beneath it.

Generated Python with unrestricted motor access is not a safety architecture.

### VoxPoser: language to spatial value maps

VoxPoser asks an LLM/VLM to compose 3D voxel maps encoding affordances and
constraints. A model-based planner then optimizes a trajectory through those
maps.

Example maps for “place the block in the bowl without crossing the laptop”:

- attraction around bowl interior;
- grasp affordance around block;
- repulsion around laptop and table collision;
- orientation preference for gripper; and
- path smoothness/feasibility from the motion planner.

This grounds language into geometry rather than asking the LLM to emit every
joint target.

### Evidence boundary

These papers demonstrate important semantic/planning mechanisms in their robot
settings. They do not make language-model output a hard guarantee, solve
perception uncertainty universally, or remove the need for local feedback.

### Worked Jump Rover design example

Define a fixed skill registry:

```text
stop
turn_to_bearing(yaw_rad)
drive_local(distance_m, max_speed_mps)
follow_person(track_id, distance_m, max_speed_mps)
dock(dock_id)
```

For “follow Bruce,” let a cloud agent propose
`follow_person(track_id="person.bruce", ...)` only after local identity/
consent resolution maps language to an exact track identifier (ID). Local perception updates
the track; local planning checks obstacles; realtime control executes bounded
motion. If any layer becomes invalid, `stop` remains local.

### Reproduction exercise

Build a simulation-only skill selector with ten exact skills. Compare:

1. language-model likelihood only;
2. affordance score only; and
3. their product.

Construct commands where a useful object is absent. Report inappropriate skill
selection separately from motor execution failure.

## 18.10 Seminar 10 — Perceptive locomotion and visual parkour

### Primary works

- [Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild (2022)](https://arxiv.org/abs/2201.08117)
- [Extreme Parkour with Legged Robots (2023)](https://arxiv.org/abs/2309.14341)
- [SoloParkour (2024)](https://arxiv.org/abs/2409.13678)

### The problem

Proprioceptive locomotion reacts after the feet/body experience terrain.
Exteroception can see a step or gap before contact, enabling deliberate foot
placement or body posture. But real depth is incomplete, noisy, reflective,
occluded, and delayed.

### Robust fusion rather than blind trust

Perceptive locomotion needs a belief over terrain, not a perfect height map.
History and attention/gating can learn when exteroception agrees with
proprioception and when to rely more on contact evidence.

Conceptually:

```math
b_t=f_\psi(b_{t-1},o_t^{prop},o_t^{ext},a_{t-1}),
```

```math
a_t=\pi_\theta(o_t^{prop},b_t,c_t).
```

$b_t$ is a learned belief state carrying temporal context.

### Privileged teacher and student

Simulation can train a teacher from exact terrain geometry. A student receives
noisy depth/history and distills the teacher or learns from its experience.
The gap to audit is:

```text
teacher truth: exact terrain/contact/state
student input: rendered/noisy depth + proprioception
real input: sensor-specific artifacts + calibration + delay
```

### Constrained visual learning

SoloParkour explicitly formulates task reward and physical constraints, then
uses privileged experience to initialize visual off-policy learning. This is a
useful combination of Chapter 6 (off-policy), Chapter 7 (constraints), Chapter
14 (experience), and this chapter's perception hierarchy.

### Evidence

The cited works show increasingly demanding real quadruped terrain/parkour
behaviors under their sensors and obstacle protocols. They establish that
vision can participate in dynamic locomotion, including on low-cost platforms.

### Failure taxonomy

Do not report only obstacle success. Label:

- perception missed obstacle;
- perceived geometry wrong;
- goal/contact plan infeasible;
- policy chose wrong motion despite adequate perception;
- actuator/contact mismatch;
- latency/stale frame;
- body collision other than intended contact;
- recovery succeeded/failed; and
- safety intervention.

### Microduck experiment ladder

1. Add one obstacle and an oracle geometric local planner commanding the
   unchanged velocity policy.
2. Vary tracking accuracy and determine the locomotion controller's command
   response limit.
3. Add realistic perception noise/dropout to the planner input.
4. Only then compare a new depth-conditioned locomotion actor.
5. Evaluate held-out obstacle geometry and sensor faults.

This ladder determines whether failure belongs to perception, planning, command
tracking, or joint-level control.

## 18.11 Synthesis: five enduring research themes

Across these papers, the durable ideas are:

1. **Scale data generation carefully.** GPU environments, robot fleets, and
   cross-embodiment datasets change what is learnable, but interfaces and data
   balance still decide value.
2. **Exploit information asymmetry during training.** Privileged teachers,
   critics, and world models can improve learning while deployable actors use
   realistic inputs.
3. **Represent time explicitly.** History, recurrent belief, action chunks,
   and predictive models address partial observability and temporal coherence.
4. **Separate semantic intent from physical authority.** VLA/LLM planning
   belongs above bounded local skills and hard realtime safety.
5. **Evaluate the system, not the model name.** Data, normalization, control
   rate, actuator, planner, hardware limits, and failure recovery define robot
   intelligence in operation.

Continue with the
[open-source ecosystem and reproduction labs](19_open_source_robot_learning_ecosystem.md),
which turns these seminars into executable project choices.

## 18.12 Folded seminar-exercise guidance

<details>
<summary>Show reference experiment designs for Seminars 1–10</summary>

These are solution **structures**, because reproducible measurements—not a
predetermined winning number—answer the research exercises.

1. **Parallel PPO:** keep total transitions, task, network, optimizer, and
   evaluator fixed. Vary only environment count and corresponding iteration
   count. Report throughput, memory, updates, wall time, and three-seed
   uncertainty. If return changes, batch/update geometry changed learning even
   though transition count matched.
2. **RMA:** compare feed-forward robustness, history, and privileged-teacher/
   history-student variants with matched actor capacity where possible. Use
   held-out fixed dynamics plus a mid-episode change; plot tracking error versus
   time since the change to distinguish robustness from adaptation speed.
3. **World model:** split complete trajectories before creating windows, fit on
   training trajectories, and roll out held-out action sequences open loop.
   Compare one-, five-, and 25-step state error against persistence. Never leak
   overlapping future windows into validation.
4. **TD-MPC2 horizon:** freeze one checkpoint and evaluator, then vary planning
   horizon. Record success/return together with median, p95/p99, and maximum
   inference time. A horizon is unusable if it misses the control deadline even
   when its simulator return improves.
5. **QT-Opt action search:** define a known 2D Q surface with multiple peaks.
   Compare grid and cross-entropy method (CEM) using equal Q-evaluation budgets;
   repeat random CEM seeds and report distance/value regret from the known
   optimum.
6. **ACT/Diffusion:** hold dataset split and visual encoder fixed. Compare one-
   step BC and two chunk horizons, inject the same perturbation, and measure
   success, recovery time, action discontinuity, and inference age. Visualize
   full predicted chunks so “smooth” cannot hide stale open-loop action.
7. **Open X transfer:** first harmonize action frames, units, rates, cameras,
   and missing fields in a schema table. Match target examples across target-
   only and pretrain/fine-tune runs; report every task so negative transfer is
   visible.
8. **Open VLA:** reproduce the released evaluator before fine-tuning. Compare
   from-scratch BC, frozen backbone, parameter-efficient adaptation, and full
   adaptation only when compute allows. Fix target splits and report memory,
   tail latency, observation age, in-distribution success, and held-out object/
   layout success.
9. **Language planning:** construct exact skill IDs and local preconditions.
   Evaluate language model (LM) relevance, affordance-only, and their product
   on present/absent
   objects. Score invalid selection separately from execution failure and prove
   timeout/network loss retains local `stop`.
10. **Perceptive locomotion:** progress from oracle geometry + existing velocity
    policy, through noisy/dropout planning, to a new visual actor. Use the same
    held-out obstacles and label perception, planning, command tracking,
    contact, actuator, recovery, and safety failures separately.

One compact run table prevents a throughput exercise from becoming a story:

```csv
seed,num_envs,total_transitions,updates,fps,gpu_peak_mb,wall_s,eval_return
0,64,REPLACE,REPLACE,REPLACE,REPLACE,REPLACE,REPLACE
0,256,REPLACE,REPLACE,REPLACE,REPLACE,REPLACE,REPLACE
0,MAX_FEASIBLE,REPLACE,REPLACE,REPLACE,REPLACE,REPLACE,REPLACE
```

</details>
