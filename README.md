# Robot Reinforcement Learning: From First Principles to Real Robots

[![Book checks](https://github.com/lzbgt/robot-rl-self-study/actions/workflows/book-check.yml/badge.svg)](https://github.com/lzbgt/robot-rl-self-study/actions/workflows/book-check.yml)

Read the chapters in the [book map](#book-map), or download the complete
**[PDF edition](dist/robot-rl-self-study.pdf)**. GitHub collapses chapter-end
solutions so you can work independently; the PDF expands them for offline
reading and printing.

The canonical public home is
[`lzbgt/robot-rl-self-study`](https://github.com/lzbgt/robot-rl-self-study).
The Microduck case study is developed in the separate upstream project linked
below; this repository owns the general self-study book.

This is a free, beginner-friendly self-study book about **reinforcement
learning for robotics**. Microduck—a small open-source biped—is the running
hands-on project, not the boundary of the subject. You will first learn the
general ideas with small programs you can understand completely, then connect
them to a modern GPU-parallel PPO system, and finally learn how current robot
research extends or departs from that recipe.

The book assumes basic Python but no prior reinforcement learning (RL), control
theory, or robotics. A technical term is defined when it first appears and is
also indexed in Chapter 20. Equations are always followed by a plain-language
interpretation, a numerical example, and a question about what the equation
changes in an experiment.

The Microduck material matches the
[`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl)
project as of 2026-08-31. Microduck is approximately 25 cm tall, weighs about
800 g, has 14 actuated joints, runs its policy at 50 Hz, and uses MuJoCo Warp
through `mjlab` with RSL-RL's Proximal Policy Optimization (PPO).

## What “learn” means here

After completing the core track, you should be able to:

- formulate a robot task as a Markov Decision Process (MDP), while recognizing
  when the observation is only a partial view of the state;
- derive and implement return, value, Bellman backup, temporal-difference
  error, advantage, policy gradient, and the PPO clipped objective;
- explain the tradeoffs among value-based, on-policy actor-critic, off-policy
  actor-critic, model-based, offline, and imitation-learning approaches;
- reason about control rate, latency, state estimation, actuator dynamics,
  contacts, action representation, and safety boundaries;
- trace one real Microduck control step from a 61-value observation to 14
  position targets and explain what is—and is not—learned;
- run a smoke experiment, train, diagnose reward hacking, evaluate across
  seeds and conditions, export ONNX, and rehearse sim-to-real deployment;
- read a current robot-learning paper without treating its abstract, benchmark
  mean, or “state of the art” label as universal evidence; and
- design a reproducible research extension with baselines, ablations,
  uncertainty estimates, artifacts, and a hardware safety case.

## Book map

### Part I — General reinforcement-learning foundations

| Chapter | Central question | Deliverable |
| --- | --- | --- |
| [1. The RL problem](01_rl_foundations.md) | What is learned, and from what signal? | Write an MDP and distinguish command from action |
| [2. Math and neural-network toolkit](02_math_and_neural_network_toolkit.md) | What mathematics does deep RL actually use? | Work through probability, gradients, and backpropagation |
| [3. Bellman methods from tables to DQN](03_bellman_and_value_learning.md) | How can future reward be learned from one-step experience? | Implement value iteration and tabular Q-learning |
| [4. The algorithm map](04_rl_algorithm_families.md) | Why are there so many RL algorithms? | Select an algorithm family from task constraints |
| [5. PPO from equations to code](05_ppo_from_equations_to_code.md) | How does an on-policy actor-critic update safely? | Calculate GAE and the clipped PPO loss |
| [6. Off-policy and model-based control](06_off_policy_and_model_based_rl.md) | When should experience be reused or dynamics be learned? | Compare SAC, TD3, Dreamer, and TD-MPC2 |

### Part II — Robotics foundations and the Microduck laboratory

| Chapter | Central question | Deliverable |
| --- | --- | --- |
| [7. Robot dynamics, control, and estimation](07_robotics_control_and_estimation.md) | What must an RL practitioner know about a physical robot? | Draw the nested control loops and choose an action space |
| [8. Microduck architecture](08_microduck_software_and_control_architecture.md) | How do simulator, task, trainer, and runtime connect? | Trace a 61D observation to a 14D action |
| [9. Setup and first experiment](09_microduck_setup_and_first_experiment.md) | How do I obtain a trustworthy first result? | Run tests, a smoke train, playback, and a video |
| [10. Environment anatomy](10_microduck_environment_anatomy.md) | Where do task behavior and unintended shortcuts come from? | Read commands, observations, rewards, events, and curricula |
| [11. Training, evaluation, and debugging](11_training_evaluation_and_debugging.md) | How do I distinguish learning from reward hacking? | Produce an evidence-backed checkpoint report |
| [12. Deployment and sim-to-real](12_deployment_and_sim2real.md) | How does a checkpoint become safe robot behavior? | Export normalized ONNX and verify the runtime contract |
| [13. Microduck customization labs](13_microduck_customization_labs.md) | How do I change a real project without breaking transfer? | Add and test a term, curriculum, or new task |

### Part III — Modern robot learning and research practice

| Chapter | Central question | Deliverable |
| --- | --- | --- |
| [14. Demonstrations, imitation, and offline RL](14_imitation_and_offline_robot_learning.md) | How can robots learn from logged or demonstrated behavior? | Diagnose distribution shift and choose BC, IQL, or CQL |
| [15. Robust locomotion, adaptation, and sim-to-real research](15_modern_robot_locomotion_and_adaptation.md) | What ideas power recent real-robot locomotion? | Compare randomization, privileged learning, and online adaptation |
| [16. Vision, foundation policies, and hierarchical autonomy](16_vision_foundation_policies_and_hierarchy.md) | Where do perception, language, planning, and low-level control meet? | Design a safe cloud/local/real-time hierarchy |
| [17. Reproducing research and capstone projects](17_research_literacy_and_capstones.md) | How do I verify a paper and make a credible contribution? | Write a reproduction card and execute a scoped capstone |
| [18. Detailed paper seminars](18_detailed_paper_seminars.md) | What do influential robot-intelligence papers actually contribute? | Explain and reproduce ten major research directions |
| [19. Open-source ecosystem and labs](19_open_source_robot_learning_ecosystem.md) | Which project should I study, and how? | Select, pin, inspect, and reproduce an official project |
| [20. Glossary and worked problems](20_glossary_and_worked_problems.md) | Can I explain and calculate the core ideas unaided? | Check solutions and identify weak areas |

## Three study tracks

The **core theory track** is Chapters 1–7, then the small programs in
[`examples/`](examples/). It is suitable even if you do not own a robot.

The **builder track** adds Chapters 8–13 and uses the Microduck repository.
Begin with 64 simulated robots and five PPO iterations; a 20-hour run is not a
first experiment.

The **research track** adds Chapters 14–20. For each paper, record the task,
robot, observation/action contract, data source, baseline, evaluation protocol,
and limitations before reading the claimed result.

Suggested pacing is 12–16 weeks at five to eight hours per week. Do not advance
because you have read a chapter; advance when you can produce its deliverable.

## The learning loop

```text
predict what should happen
          |
          v
run a small controlled experiment
          |
          v
inspect numbers and a rollout
          |
          v
explain the result with the theory
          |
          v
change one hypothesis-backed factor
          |
          +----------> repeat
```

RL is empirical engineering. A plausible story is not evidence. A high total
reward is not proof that the intended skill works. Preserve the configuration,
seed, checkpoint, metrics, evaluator, and rollout behind each conclusion.

## A research claim is not a universal ranking

This book uses primary papers and official project repositories. “State of the
art” is treated as a claim scoped to a dated benchmark and protocol—not a title
owned forever by an algorithm. A method that is excellent for offline visual
manipulation may be a poor choice for a 100 Hz balance loop; a result obtained
with privileged simulation state does not prove the deployed camera policy has
the same information.

Each research section therefore asks:

1. What exact task and embodiment were evaluated?
2. What information was available during training and deployment?
3. Was learning online, offline, simulated, real-world, or a mixture?
4. What baseline and compute/data budget were used?
5. How many seeds and hardware trials support the result?
6. Which artifact—code, checkpoint, dataset, or protocol—is actually public?

See [SOURCES.md](SOURCES.md) for the annotated primary-source index and its
inclusion policy.

## Running the book examples

The introductory examples need only the Python standard library:

```bash
python examples/bandit_incremental_mean.py
python examples/gridworld_value_iteration.py
python examples/tabular_q_learning.py
python examples/ppo_clip_demo.py
python scripts/check_book.py
```

Inside the Microduck checkout, use its locked environment:

```bash
uv sync
uv run list-envs
uv run --with pytest pytest tests/

# A pipeline test, not a trained locomotion policy.
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 \
    --agent.max_iterations 5
```

## PDF edition

The committed [PDF](dist/robot-rl-self-study.pdf) is generated from the same 20
chapter files as the GitHub edition. The builder converts fenced `math` blocks
to native display equations, expands every folded answer at its chapter end,
embeds the fonts, wraps code and URLs, and fails validation if TeX reports a
line, table, or page overflow.

On Arch Linux, install the reproducible toolchain with:

```bash
sudo pacman -S --needed pandoc-cli texlive-xetex texlive-latexextra \
    texlive-fontsrecommended noto-fonts poppler
```

On Ubuntu or in a GitHub Actions-compatible environment:

```bash
sudo apt-get install pandoc texlive-xetex texlive-latex-extra \
    texlive-fonts-recommended fonts-noto-core poppler-utils
```

Then build and verify the complete book:

```bash
make pdf        # writes dist/robot-rl-self-study.pdf
make pdf-check  # also checks pages, contents, fonts, and the XeLaTeX log
make check      # Markdown/examples plus the complete PDF pipeline
```

MiKTeX can be used on Windows if it provides `xelatex` on `PATH`; Pandoc and
Poppler are still required. TeX Live is the tested Linux implementation. The
complete Linux, macOS, Windows/MiKTeX, verification, and troubleshooting
procedure is in [`pdf/README.md`](pdf/README.md). The source-preserving
conversion is implemented by [`scripts/build_pdf.py`](scripts/build_pdf.py).

## What the Microduck walking policy actually learns

The main velocity policy is a local, command-conditioned locomotion controller.
It learns responses to requested forward/lateral velocity, turn rate, and head
pose. Playback resamples these commands, so a viewer can resemble a random
walk; the network actions themselves are conditioned on state and command.

It is **not** an autonomous navigator. Its actor has no camera, LiDAR, forward
obstacle map, or global destination. The six body-pose command inputs are kept
for family compatibility, but the main velocity recipe does not train a
nonzero body-pose objective. Chapter 10 proves these statements from the actual
observation and reward configuration, and Chapter 16 shows two architectures
for adding perception and planning without confusing them with low-level RL.

## Primary Microduck project references

- [Microduck RL repository](https://github.com/pollen-robotics/microduck_rl)
  is the executable case study.
- [Project README](https://github.com/pollen-robotics/microduck_rl/blob/main/README.md)
  is the concise operator entry point.
- [`microduck_velocity_env_cfg.py`](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)
  is the main walking recipe and shared base.
- [`mdp.py`](https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/mdp.py)
  contains custom MDP terms.
- [`scripts/export.py`](https://github.com/pollen-robotics/microduck_rl/blob/main/scripts/export.py)
  is the mandatory ONNX export path.
- [`scripts/infer_policy.py`](https://github.com/pollen-robotics/microduck_rl/blob/main/scripts/infer_policy.py)
  rehearses the deployment observation and command contract.

## License and attribution

The book is released under the Apache License 2.0. Microduck project names,
code excerpts, and measured configuration facts are attributed to the
open-source `pollen-robotics/microduck_rl` project. Papers and external projects
remain under their own licenses; this book links to them and paraphrases their
claims rather than redistributing their content.
