# Runnable Learning Examples

These programs are intentionally small and use only the Python standard
library. They are not production RL implementations. Their purpose is to make
one idea visible enough that you can calculate expected behavior, alter it, and
explain the output.

Run from the book directory:

```bash
python examples/bandit_incremental_mean.py
python examples/returns_and_occupancy.py
python examples/gaussian_policy_math.py
python examples/gae_walkthrough.py
python examples/cem_mpc_point_mass.py
python examples/pd_joint_and_kalman.py
python examples/evaluation_statistics.py
python examples/realtime_data_age.py
python examples/behavior_cloning_shift.py
python examples/camera_latency_and_stopping.py
python examples/gridworld_value_iteration.py
python examples/tabular_q_learning.py
python examples/ppo_clip_demo.py
```

Recommended workflow:

1. read the file;
2. predict one printed value or qualitative outcome;
3. run it unchanged;
4. alter one constant;
5. explain the difference in a lab note; and
6. restore the baseline before making the next change.

| Program | Main concept | First modification |
| --- | --- | --- |
| `bandit_incremental_mean.py` | exploration and sample-average value | compare `epsilon=0`, `0.1`, and `0.5` |
| `returns_and_occupancy.py` | discounted return and discounted state occupancy | change the discount factor and explain both shifts |
| `gaussian_policy_math.py` | Gaussian log probability, likelihood ratios, and entropy | move the new mean farther from the sampled action |
| `gae_walkthrough.py` | temporal-difference residuals and Generalized Advantage Estimation | compare lambda equal to zero and one |
| `cem_mpc_point_mass.py` | sampling-based Model Predictive Control with the cross-entropy method | shorten the planning horizon or increase control cost |
| `pd_joint_and_kalman.py` | proportional-derivative joint control and scalar Kalman filtering | change sensor noise and controller damping separately |
| `gridworld_value_iteration.py` | known-model Bellman optimality backup | change discount factor or step reward |
| `tabular_q_learning.py` | off-policy TD control from experience | implement SARSA and compare cliff entries |
| `ppo_clip_demo.py` | PPO probability ratio and clipping | change clip epsilon and advantage sign |
| `evaluation_statistics.py` | seed-level summaries, bootstrap, and Wilson intervals | add an outlier and compare mean, median, and IQM |
| `realtime_data_age.py` | feedback queue semantics and age of information | add another missed controller release |
| `behavior_cloning_shift.py` | multimodal action averaging and compounding drift | change the horizon and plot cumulative error by hand |
| `camera_latency_and_stopping.py` | pixel geometry and uncertainty-aware stopping | change map age and depth uncertainty |

Each file fixes a random seed so the first run is reproducible. A real research
result must use multiple seeds rather than treating one deterministic teaching
run as a performance distribution.
