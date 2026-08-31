# Primary Sources and Open-Source Study Index

Last reviewed: 2026-09-01.

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

### Applied Dynamic Programming (1962)

- [RAND publication record](https://www.rand.org/pubs/reports/R352.html)
- Richard E. Bellman and Stuart E. Dreyfus.
- Establishes: an early primary account of dynamic programming applied to
  sequential optimization, trajectories, feedback control, and related
  problems.
- Study caution: the historical formulation assumes models/representations
  unlike modern sampled deep robot learning.

### Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems (1983)

- [Publisher record](https://doi.org/10.1109/TSMC.1983.6313077)
- Andrew G. Barto, Richard S. Sutton, Charles W. Anderson.
- Establishes: an early actor-critic-style adaptive controller evaluated on a
  pole-balancing problem.
- Study caution: historical influence does not make its network or experiment
  a present-day robot baseline.

### Q-learning (1992)

- [Publisher record](https://link.springer.com/article/10.1007/BF00992698)
- Christopher J. C. H. Watkins and Peter Dayan.
- Establishes: the tabular off-policy Q-learning update and convergence result
  under the paper's discrete, repeatedly sampled conditions.
- Does not establish: convergence with nonlinear neural approximation and
  arbitrary replay distributions.

### REINFORCE (1992)

- [Publisher record](https://link.springer.com/article/10.1007/BF00992696)
- Ronald J. Williams.
- Establishes: stochastic connectionist update rules aligned with gradients of
  expected reinforcement in the paper's settings.
- Study caution: the basic estimator can have high variance; actor-critic,
  advantage, and trust-region machinery address later practical pressures.

### Policy invariance under reward transformations (1999)

- [Author-hosted paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Andrew Y. Ng, Daishi Harada, Stuart Russell.
- Establishes: conditions under which potential-based reward shaping preserves
  optimal policies.
- Study caution: terminal handling, discounting, and implementation must match
  the theorem; arbitrary dense bonuses are not covered.

### A Natural Policy Gradient (2001)

- [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)
- Sham Kakade.
- Establishes: policy-gradient steps adjusted for local policy-distribution
  geometry, a conceptual predecessor to trust-region methods.

### Deterministic Policy Gradient and DDPG (2014/2015)

- [Deterministic Policy Gradient paper](https://proceedings.mlr.press/v32/silver14.html)
- [DDPG paper](https://arxiv.org/abs/1509.02971)
- David Silver et al.; Timothy Lillicrap et al.
- Establishes: deterministic actor gradients and an influential deep
  continuous-control actor-critic implementation with replay/targets.
- Study caution: TD3 was motivated by value-error and brittleness observed in
  this family.

### Playing Atari with Deep Reinforcement Learning (2013)

- [Paper](https://arxiv.org/abs/1312.5602)
- Volodymyr Mnih et al.
- Establishes: an influential deep Q-learning result from pixel observations
  in discrete Atari action spaces.
- Does not establish: suitability of DQN for high-dimensional continuous robot
  commands.

### Double DQN and Rainbow (2015/2017)

- [Double DQN paper](https://arxiv.org/abs/1509.06461)
- [Rainbow paper](https://arxiv.org/abs/1710.02298)
- Hado van Hasselt, Arthur Guez, David Silver; Matteo Hessel et al.
- Establishes: decoupled action selection/evaluation reduces DQN
  overestimation in the evaluated setting, and several value-learning
  extensions can be combined with measured interactions on Atari.
- Study caution: these are discrete-action results and not direct replacements
  for continuous robot actors.

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

### Generative Adversarial Imitation Learning (2016)

- [Paper](https://arxiv.org/abs/1606.03476)
- Jonathan Ho and Stefano Ermon.
- Establishes: matching an expert's occupancy distribution through an
  adversarial discriminator and a policy-optimization loop in the paper's
  control tasks.
- Study caution: the method needs interactive policy rollouts and can inherit
  discriminator instability; it is not ordinary supervised behavior cloning.

### Decision Transformer (2021)

- [Paper](https://arxiv.org/abs/2106.01345)
- [Official code](https://github.com/kzl/decision-transformer)
- Lili Chen et al.
- Establishes: casting offline control as return-conditioned sequence modeling
  on the paper's Atari, OpenAI Gym, and Key-to-Door settings.
- Does not establish: that asking for an arbitrarily high return creates
  behavior absent from the dataset, or that a return token supplies physical
  safety.

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

### Cal-QL: Calibrated Offline RL Pre-Training (2023)

- [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html)
- [Official code](https://github.com/nakamotoo/Cal-QL)
- Mitsuhiko Nakamoto et al.
- Establishes: calibrating a conservative value initialization to reduce the
  early "unlearning" problem during offline-to-online fine-tuning on the
  paper's benchmark suite.
- Study caution: safe online improvement still requires a hardware envelope,
  replay/data-mixing choices, reset protocols, and measured real trials.

### Efficient Online Reinforcement Learning with Offline Data / RLPD (2023)

- [PMLR paper](https://proceedings.mlr.press/v202/ball23a.html)
- [Official code](https://github.com/ikostrikov/rlpd)
- Philip J. Ball et al.
- Establishes: a simple off-policy recipe that mixes offline and newly
  collected data, with ensemble and high update-to-data-ratio choices, in the
  paper's online fine-tuning experiments.
- Does not establish: that arbitrary logged robot data are safe, correctly
  labeled, or sufficiently covering for autonomous exploration.

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

### VQ-BeT: Behavior Generation with Latent Actions (2024)

- [Paper](https://arxiv.org/abs/2403.03181)
- [Official code](https://github.com/jayLEE0301/vq_bet_official)
- Seungjae Lee et al.
- Establishes: a residual vector-quantized action representation plus a
  transformer policy for multimodal behavior generation in the evaluated
  imitation settings.
- Study caution: codebook collapse, action-unit conventions, and temporal
  chunk execution must be checked on a new embodiment.

### 3D Diffusion Policy / DP3 (2024)

- [Paper](https://arxiv.org/abs/2403.03954)
- [Official code](https://github.com/YanjieZe/3D-Diffusion-Policy)
- Yanjie Ze, Gu Zhang et al.
- Establishes: compact point-cloud conditioning for diffusion-policy learning
  on the paper's simulated and physical manipulation tasks.
- Study caution: depth quality, cropping, calibration, point sampling, and
  camera placement are part of the method rather than interchangeable inputs.

### BAKU: An Efficient Transformer for Multi-Task Policy Learning (2024)

- [Paper](https://arxiv.org/abs/2406.07539)
- [Official code](https://github.com/siddhanthaldar/BAKU)
- Siddhant Haldar, Zhuoran Peng, Lerrel Pinto.
- Establishes: a modular transformer policy and action-head comparison across
  the paper's multi-task simulation and real-robot imitation settings.
- Study caution: compare data, observation encoders, action heads, and task
  sampling together; "transformer" alone is not the causal intervention.

### DROID robot-manipulation dataset (2024)

- [Paper](https://arxiv.org/abs/2403.12945)
- [Official dataset code](https://github.com/droid-dataset/droid)
- DROID collaboration.
- Establishes: a large, distributed real-robot manipulation dataset and data
  collection/evaluation pipeline across the participating sites.
- Study caution: dataset size does not erase site, operator, camera, success
  label, embodiment, or action-schema heterogeneity.

### robomimic

- [Official code](https://github.com/ARISE-Initiative/robomimic)
- Study use: a modular imitation-learning framework with behavior-cloning,
  recurrent, transformer, and generative baselines plus robot datasets.
- Study caution: pin the exact dataset, observation modalities, low-dimensional
  keys, environment version, and evaluation protocol before comparing scores.

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

### DeepMimic (2018)

- [Paper](https://arxiv.org/abs/1804.02717)
- Xue Bin Peng et al.
- Establishes: reinforcement learning of physics-based character skills from
  reference motions using phase-conditioned imitation and task rewards in the
  paper's simulated settings.
- Study caution: a human or character reference trajectory is not directly a
  feasible robot trajectory; morphology, contacts, torque limits, and
  retargeting must be modeled.

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

### Adversarial Motion Priors / AMP (2021)

- [Paper](https://arxiv.org/abs/2104.02180)
- Xue Bin Peng et al.
- Establishes: an adversarially learned motion-style reward that can be
  combined with task rewards without requiring phase alignment to a single
  reference trajectory in the evaluated simulated characters.
- Study caution: a discriminator rewards similarity to its training motion
  distribution, not robot feasibility or task completion by itself.

### Rapid Motor Adaptation (2021)

- [Paper](https://arxiv.org/abs/2107.04034)
- [Official project page](https://ashish-kmr.github.io/rma-legged-robots/)
- [Author-linked training code](https://github.com/antonilo/rl_locomotion)
- Ashish Kumar et al.
- Establishes: a base policy plus history-based adaptation module for the
  evaluated quadruped conditions.

### Adversarial Skill Embeddings / ASE (2022)

- [Paper](https://arxiv.org/abs/2205.01906)
- [Official code](https://github.com/nv-tlabs/ASE)
- Xue Bin Peng et al.
- Establishes: a latent-conditioned low-level controller that learns a reusable
  motion-skill space from motion data and can be composed for downstream tasks
  in the paper's simulated humanoid setting.
- Study caution: latent diversity and imitation quality do not prove that a
  downstream task can safely or efficiently search the learned skill space.

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

### MaskedMimic and ProtoMotions (2024)

- [Paper](https://arxiv.org/abs/2409.14393)
- [Official code](https://github.com/NVlabs/ProtoMotions)
- Establishes: a masked, partially specified motion-imitation controller and an
  open training framework for the reported physics-based humanoid tasks.
- Study caution: broad motion in a simulated humanoid does not imply actuator,
  contact, timing, or safety compatibility with a physical robot.

### ASAP: Aligning Simulation and Real-World Physics for Humanoid Robot (2025)

- [Paper](https://arxiv.org/abs/2502.01143)
- [Official code](https://github.com/LeCAR-Lab/ASAP)
- Establishes: a two-stage motion imitation and real-world-dynamics alignment
  approach evaluated on the paper's humanoid motion-tracking tasks.
- Study caution: the learned residual captures discrepancies inside the
  measured operating region; it is not a certificate outside that support.

### BeyondMimic (2025 preprint)

- [Paper](https://arxiv.org/abs/2508.08241)
- [Official code](https://github.com/HybridRobotics/whole_body_tracking)
- Establishes: the paper's whole-body humanoid tracking recipe, including
  motion retargeting and deployable policy components, on its stated robots and
  motion suite.
- Study caution: this is a recent preprint; verify revisions, released assets,
  and the exact real-robot trial protocol before treating comparisons as final.

### Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking (2026 preprint)

- [Paper](https://arxiv.org/abs/2604.17335)
- Establishes: a recent reported approach to terrain-aware whole-body humanoid
  motion generation in the paper's specified data and evaluation setting.
- Study caution: this is a 2026 preprint. Recheck its current version,
  publication status, code availability, and independent reproduction before
  making a design commitment.

### FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control (2025)

- [Paper](https://arxiv.org/abs/2505.22642)
- [Official code](https://github.com/younggyoseo/FastTD3)
- Establishes: a tuned, massively parallel Twin Delayed Deep Deterministic
  Policy Gradient recipe with rapid learning on the paper's humanoid-control
  benchmarks.
- Study caution: the result is a coupled algorithm, implementation, batching,
  replay, update-ratio, simulator, and hardware recipe—not evidence that the
  TD3 name alone causes the speed.

### Learning Sim-to-Real Humanoid Locomotion in 15 Minutes (2025 preprint)

- [Paper](https://arxiv.org/abs/2512.01996)
- [Official code](https://github.com/amazon-far/holosoma)
- Younggyo Seo et al.
- Establishes: reported FastSAC/FastTD3 recipes at massive simulation scale on
  the stated GPUs, humanoids, and protocols.
- Study caution: a training-time headline is hardware/configuration dependent.

### FastDSAC: Constrained Exploration for Scalable Humanoid Locomotion (2026 preprint)

- [Paper](https://arxiv.org/abs/2606.31691)
- [Official code](https://github.com/luge66/FastDSAC)
- Establishes: a recent distributional Soft Actor-Critic variant and training
  recipe evaluated on the paper's humanoid-control suite.
- Study caution: this is a new 2026 preprint. Treat the result as provisional
  until revisions, official artifacts, exact budgets, and independent evidence
  have been checked.

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

### FAST action tokenization (2025)

- [Paper](https://arxiv.org/abs/2501.09747)
- [Official implementation in `openpi`](https://github.com/Physical-Intelligence/openpi)
- Establishes: Frequency-space Action Sequence Tokenization (FAST), which uses
  a discrete cosine transform and byte-pair encoding to compress continuous
  robot action chunks for autoregressive vision-language-action training in the
  paper's datasets and evaluations.
- Study caution: token compression changes sequence length and modeling
  convenience; it does not by itself establish closed-loop stability or safe
  low-level control.

### OpenVLA-OFT (2025)

- [Paper](https://arxiv.org/abs/2502.19645)
- [Official code](https://github.com/moojink/openvla-oft)
- Establishes: an optimized fine-tuning recipe for OpenVLA, including parallel
  decoding and continuous action representations, on the reported manipulation
  benchmarks.
- Study caution: compare against the exact OpenVLA checkpoint, data split,
  action normalization, image preprocessing, and control-frequency contract.

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

### Gemini Robotics (2025)

- [Technical report](https://arxiv.org/abs/2503.20020)
- Google DeepMind et al.
- Establishes: reported embodied-reasoning and vision-language-action results
  for the models, robots, data, and evaluation suites described in the report.
- Availability caution: a technical report is not an open reproduction bundle.
  Verify public weights, training data, evaluator code, and access terms before
  selecting it for a reproducible student capstone.

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

### Green-VLA (2026 preprint)

- [Paper](https://arxiv.org/abs/2602.00919)
- Establishes: a staged curriculum from foundational vision-language models
  through multimodal grounding, multi-embodiment pretraining,
  embodiment-specific adaptation, and reinforcement-learning alignment for the
  paper's Green humanoid and benchmark settings.
- Study caution: this is a 2026 preprint. Recheck version, official code and
  weights, data/preprocessing, baselines, real-trial protocol, and independent
  evidence. Here “Green” names the robot/platform; it is not an energy claim.

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
- Maintenance caution: the official repository was archived in October 2025.
  Pin a commit and environment, and treat the equations/paper as the authority
  if a future dependency conflict changes executable behavior.

### Machine Learning Reproducibility Challenge program report (2021)

- [JMLR paper](https://www.jmlr.org/papers/v22/20-303.html)
- Establishes: lessons from a structured, multi-year effort to independently
  reproduce machine-learning papers, including recurring reporting and artifact
  problems.
- Study caution: a reproduction can test a bounded claim and implementation;
  it does not prove that all results or future versions are correct.

### ACM artifact review and badging policy

- [Official ACM policy](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- Study use: precise vocabulary for artifacts that are available, functional,
  reusable, or whose results were reproduced/replicated under a review process.
- Study caution: read the badge definition and review scope. A badge is not a
  blanket endorsement of safety, scientific validity, or every paper claim.

## Open robot-learning ecosystems and benchmarks

These projects are useful substrates, not a leaderboard declaring one universal
winner. Their rapidly changing interfaces make a pinned revision part of every
experiment.

### Gymnasium

- [Official code](https://github.com/Farama-Foundation/Gymnasium)
- Study use: a standard environment/application programming interface and
  reference environments for reinforcement-learning experiments.
- Study caution: distinguish `terminated` from `truncated`; incorrect bootstrap
  handling changes the target even when the program still runs.

### Stable-Baselines3 and CleanRL

- [Stable-Baselines3 official code](https://github.com/DLR-RM/stable-baselines3)
- [CleanRL official code](https://github.com/vwxyzjn/cleanrl)
- Study use: tested library-style baselines and compact single-file
  implementations, respectively, for checking algorithm mechanics.
- Study caution: neither substitutes for matching environment wrappers,
  budgets, network sizes, evaluation, or robot-specific safety contracts.

### robosuite, ManiSkill, RLBench, and LIBERO

- [robosuite official code](https://github.com/ARISE-Initiative/robosuite)
- [ManiSkill official code](https://github.com/haosulab/ManiSkill)
- [RLBench official code](https://github.com/stepjam/RLBench)
- [LIBERO official code](https://github.com/Lifelong-Robot-Learning/LIBERO)
- Study use: complementary simulation and benchmark ecosystems for robot
  manipulation, demonstrations, multi-task learning, and lifelong learning.
- Study caution: scores across suites are not directly comparable. Record the
  task revision, assets, control mode, observations, demonstrations, reset and
  success logic, and evaluation seeds.

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
