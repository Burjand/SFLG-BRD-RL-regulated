# Experiments

This file describes the experiments of the paper and how to run them with this code. The original experiment scripts are not in the repository: they were long, specific to my machine, and mostly handled resuming, parallelism and plotting.

Every experiment is built from the same pieces:

- a hyperparameter dictionary;
- a `Simulation` built with a training seed and run for the training budget;
- a `Simulation` built with each test seed and run for 500 test episodes.

The helper functions below do exactly that. Every code block in this file was run against the repository before release.

## Common protocol (Section 5.1)

- **Graphs.**
  - A fresh connected graph is drawn for every episode, with mean degree about 6:
    - Erdős–Rényi with p = 6/(m−1);
    - Watts–Strogatz with k = 6 and rewiring 0.3;
    - Barabási–Albert with 3 edges per new node.
  - Disconnected samples are redrawn. Edges have unit length and demand is uniform.
  - σ_min = 0.1, and β = −ln(σ_min)/diam(G) is computed per graph.
- **Episodes.**
  - Players start on distinct, uniformly sampled nodes.
  - Every step activates one player, who relocates only on strict improvement.
  - The episode ends when every player has been checked since the last relocation, or after 500 steps.
- **Metrics.**
  - The settling time T is `NE_t`, the step of the last relocation.
  - An episode did not converge if `t == max_iter`. Its T is the step of its last relocation, not 500.
  - Relocations are R = `sim_iters − plateau_counts`, and waste is W = T − R.
- **Seeds.**
  - Training seeds: 209, 580, 319.
  - Test seeds: 307, 79, 729, 397, 948, 865, 795.
  - Experiments with five test seeds use the first five.
  - The training seed seeds the training `Simulation`. Each test seed seeds a separate test `Simulation`, so training and test graphs come from different streams. All training seeds share the same test episodes.
- **Test size.** 500 episodes per (rule, configuration, training seed, test seed).
- **Training budget.** 100·n² episodes (150·n² in the feature study).
- **Uncertainty.** 95% bootstrap intervals over episodes.

The configuration grid (Table 3): m ∈ {15, 25, 35}, n = ⌊m·d⌋ with d ∈ {1/5, 1/4, 1/3, 1/2}, which gives 11 distinct cells:

```
(15,3) (15,5) (15,7) (25,5) (25,6) (25,8) (25,12) (35,7) (35,8) (35,11) (35,17)
```

## Configuration per experiment (Table 4)

| Experiment | Section | Seed pairs | Training episodes | α | H* | n_TD | Target | n_plan |
|---|---|---|---|---|---|---|---|---|
| Feature ablation (FA-1) | 5.4 | 3 × 7 | 150n² | 0.15 | 0.4 | 6 | Tree Backup | off |
| Feature scaling (FA-2) | 5.4 | 3 × 5 | 150n² | 0.15 | 0.4 | 6 | Tree Backup | off |
| Hyperparameter sweep | 5.6 | 3 × 5 | 100n² | 0.15 | 0.4 | 6 | both | 2 |
| Algorithmic ablation | 5.3, 5.5 | 3 × 5 | 100n² | 0.10 | 1.5 | 6 | standard | 5 |
| Scalability and timing | 5.2 | 3 × 7 | 100n² | 0.10 | 1.5 | 6 | standard | 5 |
| Topology transfer | 5.7 | 3 × 7 | 100n² | 0.10 | 1.5 | 1 | standard | 2 |
| Policy extraction | 6 | 3 × 7 | 100n² | 0.10 | 1.5 | 1 | standard | 2 |

Settings shared by every experiment:

- γ = 0.2 and Q₀ = 1;
- τ₀ = 1.0, η_τ = 10⁻³, τ_min = 0.01;
- τ_base = 0.01, τ_boost = 1.5, K = 3;
- ε from 1.0 to 0.05 with decay 0.999 per episode (only for the ε-greedy variants).

`config.json5` holds the scalability row (Table 5).

## Helper functions

Put this in a script at the repository root.

```python
import os, time
import numpy as np
from collections import defaultdict

from simulation import Simulation

# Table 5: the regulator of the scalability experiment, on (m, n) = (25, 6)
HP = {
    "seed": 209, "regulated": True, "fixed_graph": False,
    "network_topology": "erdos-renyi", "n_nodes": 25, "n_players": 6,
    "demand_distribution": ("uniform", 1, 1), "weight_distribution": ("uniform", 1, 1),
    "graph_density": 0.25, "rationality_epsilon": 0.0, "max_iter": 500,
    "agent_type": "RL", "learn_from_scratch": True, "action_selection": "boltzmann",
    "alpha": 0.10, "gamma": 0.2, "optimistic_value": 1,
    "n_steps_bootstrap": 6, "n_step_method": "standard",
    "reward_function_name": "potential_function_difference",
    "plan_enabled": True, "planning_time": 5,
    "epsilon": 1.0, "epsilon_decay_rate": 0.999, "epsilon_min": 0.05,
    "boltzmann_tau": 1.0, "boltzmann_tau_lr": 0.001, "target_entropy": 1.5,
}


def train(hp, train_seed, out_dir, n_episodes=None):
    hp = dict(hp, seed=train_seed, learn_from_scratch=True)
    sim = Simulation(hp)
    agent = sim.regulatory_agent
    agent.q_values_file_path = os.path.join(out_dir, "q_values.pkl.gz")
    if hasattr(agent, "model_file_path"):
        agent.model_file_path = os.path.join(out_dir, "model.pkl.gz")
    n_episodes = n_episodes or 100 * hp["n_players"] ** 2  # training budget of the article
    sim.run_simulations(n_episodes)                          # saves the tables at the end
    return dict(agent.q_table)


def make_regulator(hp, q_table, test_seed):
    hp = dict(hp, seed=test_seed, learn_from_scratch=True, epsilon=0.0, boltzmann_tau=0.01)
    sim = Simulation(hp)
    sim.learn_from_scratch = False           # no learning during the test episodes
    agent = sim.regulatory_agent
    agent.learn_from_scratch = False         # tau controller off, tau boost on
    agent.epsilon = 0.0
    if hasattr(agent, "tau"):
        agent.tau = 0.01
    agent.q_table = defaultdict(lambda: np.full(hp["n_players"], hp["optimistic_value"]), q_table)
    return sim


def make_baseline(hp, rule, test_seed):
    hp = dict(hp, seed=test_seed, learn_from_scratch=False)
    if rule == "Random (filtered)":
        hp["regulated"] = False
    else:
        hp["agent_type"] = {"Greedy": "Greedy", "First-improvement": "FI", "Random": "Random"}[rule]
    return Simulation(hp)


def evaluate(sim, n_episodes=500):
    out = {"iters_to_NE": [], "sim_iters": [], "plateau_counts": []}
    for _ in range(n_episodes):
        sim.BRD.create_players()                       # new initial profile (and a new graph inside the episode)
        NE_t, t, pot_dev, player_dev, _ = sim.run_FLG_BRD_simulation()
        out["iters_to_NE"].append(NE_t)                # settling time T
        out["sim_iters"].append(t)                     # steps, including the final check of every player
        out["plateau_counts"].append(sum(player_dev[i] == player_dev[i - 1] for i in range(1, len(player_dev))))
    return out
```

Example: one training seed and one test seed of the regulator, against the four baselines.

```python
q = train(HP, 209, "data/agent_data/my_run/seed_209")
res = {"Regulator": evaluate(make_regulator(HP, q, 307))}
for rule in ("Greedy", "First-improvement", "Random (filtered)", "Random"):
    res[rule] = evaluate(make_baseline(HP, rule, 307))

for rule, r in res.items():
    T = np.array(r["iters_to_NE"])
    failed = np.array(r["sim_iters"]) >= HP["max_iter"]
    print(f"{rule:18s} E[T] = {T.mean():.2f}   non-convergence = {failed.mean():.2%}")
```

Build every `Simulation` right before its episodes run. The constructor reseeds NumPy and `random`, so the order matters if you want runs to repeat exactly.

Regulator variants (`QL_Simple`, ε-greedy and so on) go through `make_regulator` too. Change `agent_type` or `action_selection` in `hp` for both training and testing.

## Scalability and total computational cost (Section 5.2)

**Settings.** Use `HP` as it is, on the 11 cells of Table 3, with 3 training seeds × 7 test seeds. The rules are the regulator, greedy, first-improvement, random (filtered) and random.

**Timing.** Measure only the decision phase:

- for the regulator, `state.observe()` + `choose_action()`;
- for every other rule, `choose_action()` alone.

The paper used 100 episodes per (rule, cell), single-threaded, on one graph per cell. Players are re-placed every episode.

```python
def timed_episode(sim, is_regulator):
    state, agent = sim.state, sim.regulatory_agent
    b = np.ones(sim.n_players, dtype=np.bool_)
    St = state.observe(b)
    prev, repeats, step_us = None, 0, []
    for _ in range(sim.max_iter):
        if sim.regulated and sim.agent_type != "FI":
            valid = tuple(range(sim.n_players))
        else:
            valid = tuple(i for i, x in enumerate(b) if x)
        repeats = repeats + 1 if tuple(St) == prev else 0
        prev = tuple(St)

        t0 = time.perf_counter()
        if is_regulator:
            St = state.observe(b)
        action = agent.choose_action(tuple(St), valid, state_unchanged_count=repeats)
        step_us.append((time.perf_counter() - t0) * 1e6)

        player = state.canonical_to_player(action) if is_regulator else action
        if sim.BRD.compute_player_best_response(player):
            b[:] = True
        b[player] = False
        if not b.any():
            break
        St = state.observe(b)
    return np.array(step_us)
```

Before importing NumPy, set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to 1, so that every rule is timed on a single thread.

**Total cost.**

- Every rule also pays the best-response update of the activated player. The code runs that update for every rule, greedy included.
- The update costs one candidate evaluation: u = d_Greedy / n, the greedy decision time divided by n.
- The per-step cost is c(π) = d_π + u, and the total cost is C(π) = E[T] · c(π).
- The paper's timings come from an Intel Core i7-13700HX. Absolute times differ between machines; the ratios are the comparable part.

## Reliability under state aliasing and learning mechanisms (Sections 5.3 and 5.5)

**Settings.** `HP` on the 11 cells, with 3 training seeds × 5 test seeds. The variants are changes to `HP`:

| Variant | Paper | Change to `HP` |
|---|---|---|
| E | full regulator | none |
| E_no_D | without Dyna | `plan_enabled: False` |
| E_no_N | without n-step returns | `n_steps_bootstrap: 1` |
| E_TB | Tree Backup target | `n_step_method: "tree_backup"` |
| E_no_B | ε-greedy instead of Boltzmann | `action_selection: "epsilon-greedy"` |
| S_B | Q-learning with Boltzmann only | `plan_enabled: False, n_steps_bootstrap: 1` |
| S | plain Q-learning | `agent_type: "QL_Simple"` |

The ε-greedy variants act greedily at test time, because `make_regulator` sets ε = 0.

Section 5.3 also re-evaluates the Boltzmann-trained tables with argmax selection (random tie-breaking, no τ boost). To do that, replace `choose_action` on the test agent:

```python
import types

def argmax_choose_action(self, state, valid_actions, state_unchanged_count=0):
    q_values = self.q_table[tuple(state)]
    valid_qs = np.array([q_values[i] for i in valid_actions], dtype=np.float64)
    best = np.flatnonzero(valid_qs == valid_qs.max())
    choice = int(np.random.choice(best)) if best.size > 1 else int(best[0])
    return int(valid_actions[choice])

sim = make_regulator(HP, q, 307)
sim.regulatory_agent.choose_action = types.MethodType(argmax_choose_action, sim.regulatory_agent)
```

## Feature ablation (Section 5.4)

**Settings.** Start from the development configuration:

```python
HP_DEV = dict(HP, alpha=0.15, target_entropy=0.4, n_steps_bootstrap=6,
              n_step_method="tree_backup", plan_enabled=False)
```

Use m = 25 and a training budget of 150·n²:

- FA-1 uses n = 6 and 7 test seeds. It is also tested out of distribution on Erdős–Rényi with m = 50 and m = 70, and on Barabási–Albert and Watts–Strogatz with m = 30.
- FA-2 uses n ∈ {4, 6, 9, 13} and 5 test seeds.

**Variants.** The variants change the observation, so they are edits to `State.observe` in `state_observation.py`:

- Remove one flag (b, w, z or h): drop its block from the concatenated state and from the profile id.
- Threat radius: the deployed map uses the closed 2-hop ball (`d <= 2` in `State.__init__`). Radius 1 is `d <= 1`. Co-location (radius 0) counts only players on the same node.
- Encodings: replace the median with the mean for w. Replace closeness centrality at the 60th percentile with eigenvector centrality at the 55th, 60th or 65th percentile, or with degree above 1.5, 1.6 or 1.7 times the mean degree, for z.

If a variant changes the number of blocks, `Extracted_policy` and `Decision_tree_agent` also need to change. They expect four blocks of n entries.

In the original result files, the variant called `BASELINE` is the co-location threat flag, not the deployed 2-hop map. The 2-hop map is `H_K2`.

## Hyperparameter sweep and discount factor (Section 5.6)

**Settings.**

- Development configuration with Dyna: `dict(HP_DEV, plan_enabled=True, planning_time=2)`.
- Run it under both targets (`n_step_method` `"standard"` and `"tree_backup"`).
- Cells with m ∈ {15, 25}, 3 training seeds × 5 test seeds.

**Sweep design.** One parameter changes at a time from the centre point (α = 0.15, γ = 0.2, n_TD = 6, n_plan = 2, H* = 0.4):

| Parameter | Levels |
|---|---|
| α | 0.05, 0.10, 0.15, 0.30, 0.50 |
| γ | 0.00, 0.10, 0.20, 0.50, 0.90 |
| n_TD | 1, 3, 6, 10, 20 |
| n_plan | 0 (planning off), 1, 2, 5, 10 |
| H* | 0.10, 0.25, 0.40, 0.80, 1.50 |

τ_boost is swept only at test time, over 0.01, 0.5, 1.5, 3.0 and 6.0: set `sim.regulatory_agent.tau_boost` after `make_regulator`.

## Topology transfer (Section 5.7)

**Settings.** `dict(HP, n_steps_bootstrap=1, planning_time=2)`.

**Design.**

- Two cells: (25, 6) and (35, 17).
- Train on each graph family (`network_topology` set to `"erdos-renyi"`, `"barabasi-albert"` or `"watts-strogatz"`), then test every table on all three families. Change only `network_topology` in the hp passed to `make_regulator`.
- 3 training seeds × 7 test seeds. Test seeds 865 and 795 are replaced by 612 and 421.

## Policy extraction (Section 6)

**Settings.** `dict(HP, n_steps_bootstrap=1, planning_time=2)` at (25, 6) on Erdős–Rényi, with 3 training seeds × 7 test seeds.

The table in `data/agent_data/Q_Learning/` is the one trained with seed 209 for this experiment.

**Distilling a tree and testing it as a regulator.**

```python
import gzip, pickle
from src.tools.policy_extraction import Extracted_policy
from src.regulatory_agent.convergence_time.decision_tree_agent import Decision_tree_agent

q_path = "data/agent_data/Q_Learning/q_values.pkl.gz"
distiller = Extracted_policy(q_table_file_path=q_path)
distiller.prepare_dataset()                  # one row per (observation, rank); label: rank is the argmax
distiller.bayes_optimal_diagnostic()         # fidelity ceilings (Table 10)
distiller.cross_validate(k=10, max_depth=3, min_samples_leaf=50)
distiller.train(max_depth=3, min_samples_leaf=50)

sim = make_baseline(HP, "Random", 307)       # any rule; the agent is replaced below
sim.regulatory_agent = Decision_tree_agent(prebuilt_distiller=distiller)
sim.agent_type = "DecisionTree"              # the loop translates ranks to players
tree_results = evaluate(sim)

with gzip.open(q_path, "rb") as f:           # the regulator itself, for comparison
    regulator_results = evaluate(make_regulator(HP, pickle.load(f), 307))
```

The paper's trees use depths 3 and 5, with `min_samples_leaf=50`. On the stored table, 10-fold CV of the depth-3 tree gives:

- balanced accuracy 69.94 ± 4.85%;
- F1 43.76 ± 5.48%.

**Exhaustive check (Table 11).** No tree selects a player with b = 0. To check it, apply the tree to every canonical observation:

```python
from itertools import combinations_with_replacement

def all_observations(n):
    # Every canonical observation: a multiset of n profile ids 8b + 4w + 2z + h, in ascending order
    for ids in combinations_with_replacement(range(16), n):
        b = [(p >> 3) & 1 for p in ids]
        w = [(p >> 2) & 1 for p in ids]
        z = [(p >> 1) & 1 for p in ids]
        h = [p & 1 for p in ids]
        yield tuple(b + w + z + h), b

tree = Decision_tree_agent(prebuilt_distiller=distiller)
n = 6
total = with_b = picks_b0 = 0
for obs, b in all_observations(n):
    total += 1
    if any(b):
        with_b += 1
        rank = tree.choose_action(obs, tuple(range(n)))
        picks_b0 += (b[rank] == 0)
print(total, with_b, picks_b0)   # 54264 52548 0, for both depths
```

## Cost of a full re-run

Training is single-threaded per seed, and the original scripts ran 2 seeds in parallel. The policy-extraction training (3600 episodes at (25, 6)) took about 35 s per seed. The training budget grows with n², and each step costs more as m and n grow, so the largest cells take much longer. Test episodes are cheap next to training.
