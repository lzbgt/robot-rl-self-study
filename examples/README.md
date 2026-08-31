# Runnable Learning Examples

These programs are intentionally small and use only the Python standard
library. They are not production RL implementations. Their purpose is to make
one idea visible enough that you can calculate expected behavior, alter it, and
explain the output.

Run from the book directory:

```bash
python examples/bandit_incremental_mean.py
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
| `gridworld_value_iteration.py` | known-model Bellman optimality backup | change discount factor or step reward |
| `tabular_q_learning.py` | off-policy TD control from experience | implement SARSA and compare cliff entries |
| `ppo_clip_demo.py` | PPO probability ratio and clipping | change clip epsilon and advantage sign |

Each file fixes a random seed so the first run is reproducible. A real research
result must use multiple seeds rather than treating one deterministic teaching
run as a performance distribution.

