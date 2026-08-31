# Primary Sources and Open-Source Study Index

Last reviewed: 2026-08-31.

This index supports the book's research chapters. It deliberately favors
primary papers and official repositories. Inclusion means the source is useful
for study and its identity/artifact was verified; it is not an endorsement of
every claim or a declaration that the method is currently best for every task.

## Inclusion and verification policy

A research entry should provide at least:

- a stable paper record (publisher, proceedings, OpenReview, or arXiv);
- identifiable authors and version/date;
- a clearly scoped empirical or theoretical claim; and
- preferably an official project page, source repository, data, or checkpoint.

For every source, readers should verify:

1. paper version and publication status;
2. task, embodiment, observation, action, and data setting;
3. training/evaluation compute and number of seeds/trials;
4. whether code/config/checkpoints/data reproduce the reported path; and
5. license and current dependency compatibility.

Links below point to papers or official author/project repositories, not search
result pages.

## Foundations and policy optimization

### Reinforcement Learning: An Introduction, second edition (2018)

- [MIT Press record](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- Richard S. Sutton and Andrew G. Barto.
- Establishes: the standard progression from bandits and tabular prediction to
  function approximation and policy gradients.
- Study caution: it is a general RL foundation, not a current robot deployment
  manual.

### Playing Atari with Deep Reinforcement Learning (2013)

- [Paper](https://arxiv.org/abs/1312.5602)
- Volodymyr Mnih et al.
- Establishes: an influential deep Q-learning result from pixel observations
  in discrete Atari action spaces.
- Does not establish: suitability of DQN for high-dimensional continuous robot
  commands.

### High-Dimensional Continuous Control Using GAE (2015)

- [Paper](https://arxiv.org/abs/1506.02438)
- John Schulman et al.
- Establishes: the generalized advantage estimator and its bias/variance
  motivation in policy-gradient control.

### Trust Region Policy Optimization (2015)

- [Paper](https://arxiv.org/abs/1502.05477)
- John Schulman et al.
- Establishes: the trust-region motivation that precedes PPO.

### Proximal Policy Optimization Algorithms (2017)

- [Paper](https://arxiv.org/abs/1707.06347)
- John Schulman et al.
- Establishes: PPO surrogate objectives and benchmark evidence for repeated
  minibatch updates on fresh policy data.
- Study caution: clipping is a practical surrogate, not a universal monotonic
  improvement guarantee.

### Addressing Function Approximation Error in Actor-Critic Methods (TD3, 2018)

- [Paper](https://arxiv.org/abs/1802.09477)
- Scott Fujimoto, Herke van Hoof, David Meger.
- Establishes: twin critics, delayed actor updates, and target smoothing as
  responses to continuous-control value error.

### Soft Actor-Critic (2018)

- [Paper](https://arxiv.org/abs/1801.01290)
- Tuomas Haarnoja et al.
- Establishes: an off-policy maximum-entropy stochastic actor-critic method for
  continuous control.

### Constrained Policy Optimization (2017)

- [Paper](https://arxiv.org/abs/1705.10528)
- Joshua Achiam et al.
- Establishes: policy optimization with expected constraints and stated
  theoretical properties.
- Does not replace: hard electrical, mechanical, or realtime safety.

## Model-based and world-model learning

### DreamerV3: Mastering Diverse Domains through World Models (2023)

- [Paper](https://arxiv.org/abs/2301.04104)
- Danijar Hafner et al.
- Establishes: latent world-model learning and imagined actor/critic training
  with one reported configuration across diverse evaluated domains.

### DayDreamer: World Models for Physical Robot Learning (2022)

- [Paper](https://arxiv.org/abs/2206.14176)
- Danijar Hafner et al.
- Establishes: online world-model learning directly on four physical robots in
  the paper's tasks, including quadruped locomotion and manipulation.
- Study caution: real-world sample efficiency does not remove reset labor,
  hardware wear, latency, or safety constraints.

### QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation (2018)

- [Paper](https://arxiv.org/abs/1806.10293)
- Dmitry Kalashnikov et al.
- Establishes: a distributed off-policy Q-learning system trained from a large
  mixture of autonomous and logged real-robot grasp attempts.
- Study caution: its data and infrastructure scale are part of the method's
  empirical setting and must accompany algorithm comparisons.

### TD-MPC2 (2023)

- [Paper](https://arxiv.org/abs/2310.16828)
- [Official code](https://github.com/nicklashansen/tdmpc2)
- Nicklas Hansen, Hao Su, Xiaolong Wang.
- Establishes: scalable latent model-predictive control plus value learning
  across the paper's online continuous-control suite.

## Imitation and offline learning

### DAgger (2010/2011)

- [Paper](https://arxiv.org/abs/1011.0686)
- Stéphane Ross, Geoffrey Gordon, Drew Bagnell.
- Establishes: dataset aggregation to address sequential imitation's state
  distribution shift.

### D4RL (2020)

- [Paper](https://arxiv.org/abs/2004.07219)
- Justin Fu et al.
- Establishes: offline-RL datasets/protocols designed to expose data-coverage
  challenges.

### Conservative Q-Learning (2020)

- [Paper](https://arxiv.org/abs/2006.04779)
- Aviral Kumar et al.
- Establishes: conservative value regularization for fixed offline data.

### Implicit Q-Learning (2021)

- [Paper](https://arxiv.org/abs/2110.06169)
- Ilya Kostrikov, Ashvin Nair, Sergey Levine.
- Establishes: offline improvement without evaluating arbitrary unseen policy
  actions during the primary value-learning step.

### ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (2023)

- [Paper](https://arxiv.org/abs/2304.13705)
- [Official code](https://github.com/tonyzhaozh/act)
- Tony Zhao et al.
- Establishes: action chunking and transformer-style imitation on the reported
  bimanual tasks.

### Diffusion Policy (2023)

- [Paper](https://arxiv.org/abs/2303.04137)
- [Official code](https://github.com/real-stanford/diffusion_policy)
- Cheng Chi et al.
- Establishes: conditional diffusion for multimodal visual action sequences on
  the paper's manipulation benchmarks.

## Sim-to-real and locomotion

### Domain Randomization for Visual Transfer (2017)

- [Paper](https://arxiv.org/abs/1703.06907)
- Josh Tobin et al.
- Establishes: a foundational visual domain-randomization demonstration for
  sim-to-real object localization/control.

### Dynamics Randomization for Robotic Control (2017)

- [Paper](https://arxiv.org/abs/1710.06537)
- Xue Bin Peng et al.
- Establishes: dynamics-randomized simulation policies transferred to a real
  object-pushing setup.

### Learning Agile and Dynamic Motor Skills for Legged Robots (2019)

- [Paper](https://arxiv.org/abs/1901.08652)
- Jemin Hwangbo et al.
- Establishes: an influential actuator-aware simulated-training/real-ANYmal
  deployment pipeline.

### Learning to Walk in Minutes (2021)

- [Paper](https://arxiv.org/abs/2109.11978)
- [Official `legged_gym`](https://github.com/leggedrobotics/legged_gym)
- Nikita Rudin et al.
- Establishes: massively parallel GPU locomotion training and terrain
  curriculum results in the reported setup.

### Rapid Motor Adaptation (2021)

- [Paper](https://arxiv.org/abs/2107.04034)
- [Official project page](https://ashish-kmr.github.io/rma-legged-robots/)
- [Author-linked training code](https://github.com/antonilo/rl_locomotion)
- Ashish Kumar et al.
- Establishes: a base policy plus history-based adaptation module for the
  evaluated quadruped conditions.

### Robust Perceptive Locomotion in the Wild (2022)

- [Paper](https://arxiv.org/abs/2201.08117)
- Takahiro Miki et al.
- Establishes: learned fusion of proprioception and unreliable exteroception
  for reported real quadruped deployments.

### Extreme Parkour with Legged Robots (2023)

- [Paper](https://arxiv.org/abs/2309.14341)
- Cheng et al.
- Establishes: depth-conditioned parkour behaviors on the paper's low-cost
  quadruped and obstacle protocol.

### Wheeled-Legged Navigation and Locomotion (2024)

- [Paper](https://arxiv.org/abs/2405.01792)
- Joonho Lee et al.
- Establishes: an integrated hierarchical learned locomotion/navigation system
  evaluated in the stated urban missions.

### SoloParkour (2024)

- [Paper](https://arxiv.org/abs/2409.13678)
- Elliot Chane-Sane et al.
- Establishes: constrained visual locomotion initialized from privileged
  experience on Solo-12.

### MuJoCo Playground (2025)

- [Paper](https://arxiv.org/abs/2502.08844)
- [Official code](https://github.com/google-deepmind/mujoco_playground)
- Establishes: an open GPU-accelerated MuJoCo suite with reported robot-learning
  and sim-to-real examples.

### Fast off-policy humanoid locomotion (2025 preprint)

- [Paper](https://arxiv.org/abs/2512.01996)
- Younggyo Seo et al.
- Establishes: reported FastSAC/FastTD3 recipes at massive simulation scale on
  the stated GPUs, humanoids, and protocols.
- Study caution: a training-time headline is hardware/configuration dependent.

### PACE sim-to-real (active project; 2025 paper record in project)

- [Official code and documentation](https://github.com/leggedrobotics/pace-sim2real)
- Establishes: an open, measurement-driven workflow for identifying actuator
  and joint dynamics on supported legged systems.
- Study caution: active APIs and platform-specific models must be versioned.

## Generalist robot policies and data

### RT-1: Robotics Transformer for Real-World Control at Scale (2022)

- [Paper](https://arxiv.org/abs/2212.06817)
- Anthony Brohan et al.
- Establishes: a transformer policy trained on the paper's multi-task real
  robot dataset, with image and language inputs and discretized action tokens.
- Does not establish: that language alone supplies geometric safety or that the
  reported policy transfers unchanged to arbitrary embodiments.

### RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (2023)

- [Paper](https://arxiv.org/abs/2307.15818)
- Anthony Brohan et al.
- Establishes: co-fine-tuning vision-language models on robot trajectories and
  web-scale visual-language data in the reported manipulation evaluations.
- Study caution: inspect which models, weights, robot data, and evaluation
  interfaces are available before calling a result reproducible.

### Open X-Embodiment and RT-X (2023)

- [Paper](https://arxiv.org/abs/2310.08864)
- [Official dataset organization](https://github.com/google-deepmind/open_x_embodiment)
- Open X-Embodiment Collaboration et al.
- Establishes: a large cross-institution collection of robot datasets and
  cross-embodiment policy experiments under the paper's unified format.
- Study caution: common serialization does not make action semantics, camera
  geometry, control rates, or data quality identical.

### Octo (2024)

- [Paper](https://arxiv.org/abs/2405.12213)
- [Official code](https://github.com/octo-models/octo)
- Establishes: an open generalist policy pretrained on a large multi-robot
  demonstration collection and fine-tuned in the evaluated settings.

### OpenVLA (2024)

- [Paper](https://arxiv.org/abs/2406.09246)
- [Official code](https://github.com/openvla/openvla)
- Establishes: an open 7B vision-language-action model, training recipe, and
  evaluated fine-tuning/serving results.

### $\pi_0$ (2024)

- [Paper](https://arxiv.org/abs/2410.24164)
- Kevin Black et al.
- Establishes: a VLM plus flow-matching action model evaluated on the stated
  multi-platform manipulation data and tasks.
- Availability caution: verify which weights, data, and code required for a
  claimed result are public before planning a reproduction.

### $\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization (2025)

- [Paper](https://arxiv.org/abs/2504.16054)
- [Official `openpi` code](https://github.com/Physical-Intelligence/openpi)
- Karl Pertsch et al.
- Establishes: the reported co-training and hierarchical action-generation
  approach for long-horizon manipulation evaluations.
- Study caution: distinguish the paper's full internal data recipe from the
  checkpoints and fine-tuning paths actually released in `openpi`.

### GR00T N1 (2025)

- [Paper](https://arxiv.org/abs/2503.14734)
- [Official code](https://github.com/NVIDIA/Isaac-GR00T)
- NVIDIA et al.
- Establishes: a dual-system vision-language/action architecture and released
  tooling for the humanoid manipulation settings described by the project.
- Study caution: check robot support, dataset license, inference hardware, and
  checkpoint scope at the pinned revision.

### SmolVLA (2025)

- [Paper](https://arxiv.org/abs/2506.01844)
- [Official LeRobot code](https://github.com/huggingface/lerobot)
- Hugging Face et al.
- Establishes: a compact open VLA recipe evaluated with community robot data
  and asynchronous inference in the stated tasks.
- Study caution: small parameter count is not a timing or safety guarantee on a
  particular robot computer.

### LeRobot

- [Official code](https://github.com/huggingface/lerobot)
- Establishes: an active open ecosystem for robot datasets, models, training,
  and hardware integrations.
- Study caution: the repository evolves quickly; pin a release/commit and read
  each policy's learning setting instead of calling all of it RL.

## Language-grounded planning and skill composition

### Do As I Can, Not As I Say / SayCan (2022)

- [Paper](https://arxiv.org/abs/2204.01691)
- Michael Ahn et al.
- Establishes: combining language-model skill relevance with learned
  affordance/value estimates for the reported mobile-manipulation tasks.
- Does not establish: that unconstrained model text should become motor
  commands; the candidate skill set and local feasibility model are essential.

### Code as Policies (2022)

- [Paper](https://arxiv.org/abs/2209.07753)
- Jacky Liang et al.
- Establishes: generating compositional robot-policy programs from language in
  the demonstrated tabletop and mobile manipulation settings.
- Study caution: generated code still needs constrained APIs, validation,
  timeouts, and a trusted execution boundary.

### VoxPoser (2023)

- [Paper](https://arxiv.org/abs/2307.05973)
- Wenlong Huang et al.
- Establishes: composing language-conditioned 3D value maps with a motion
  planner in the reported manipulation experiments.
- Study caution: the perception and geometric-planning assumptions are part of
  the result and can fail independently of the language model.

## Evaluation and reproducibility

### Deep Reinforcement Learning That Matters (2017)

- [Paper](https://arxiv.org/abs/1709.06560)
- Peter Henderson et al.
- Establishes: empirical evidence that implementation and experimental choices
  materially affect deep-RL comparisons.

### Deep RL at the Edge of the Statistical Precipice / RLiable (2021)

- [Paper](https://arxiv.org/abs/2108.13264)
- [Official code](https://github.com/google-research/rliable)
- Rishabh Agarwal et al.
- Establishes: uncertainty-aware aggregate evaluation methods for the few-run
  deep-RL regime.

## Case-study projects

### Microduck RL

- [Official training repository](https://github.com/pollen-robotics/microduck_rl)
- [Microduck runtime repository](https://github.com/pollen-robotics/microduck)
- Study use: complete small-biped PPO task definitions, BAM actuator modeling,
  domain randomization, ONNX export, and sim-to-real runtime contracts.

### RSL-RL

- [Official code](https://github.com/leggedrobotics/rsl_rl)
- Study use: compact robot-focused runners, PPO, actor/critic models, storage,
  and student-teacher mechanisms.

### Isaac Lab

- [Official code](https://github.com/isaac-sim/IsaacLab)
- Study use: multi-modal simulator workflows, actuator/sensor models, domain
  randomization, RL and imitation integrations.

### Genesis

- [Official code](https://github.com/Genesis-Embodied-AI/Genesis)
- Study use: a modern general robotics physics platform and an alternative
  vectorized simulation stack.
- Availability caution: distinguish released open simulator features from
  project-roadmap/generative-system descriptions.

## Maintaining this index

When adding a source:

1. link the exact paper/version and official repository;
2. summarize the narrow evidence in original words;
3. add one “does not establish” or study-caution boundary;
4. record the review date if a project is fast-moving; and
5. never update a benchmark superlative without rechecking the protocol and
   newer primary results.
