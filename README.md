# Learned regulators for best-response dynamics

Code for the paper *Explainable Reinforcement Learning Regulators for Best-Response Dynamics in Potential Games*, by Andrés B. Torres-Reyes and Rolando Menchaca-Méndez (submitted to *Mathematics*, MDPI, 2026).

The Shapley facility location game (SFLG) was introduced by [Ben-Porat and Tennenholtz (2017)](https://arxiv.org/abs/1709.10278). In it, players place facilities, and users split their demand among the facilities by a Shapley-value allocation. We use a discrete version of the game on a graph, where players take turns playing best responses until they reach a pure Nash equilibrium. The order in which they are activated decides how long that takes. A regulator chooses who moves next, seeing only four binary flags per player. In the paper we learn that regulator with tabular reinforcement learning and compare it with the usual activation rules. The comparison is by total computational cost, not only by the number of activations.

This repository contains:

- the game;
- the asynchronous best-response dynamics;
- the activation rules from the paper: greedy, first-improvement, random and random (filtered);
- the learned regulator (n-step Q-learning with Dyna planning and Boltzmann selection);
- the plain Q-learning baseline;
- the distillation of the learned policy into decision trees.

The experiment scripts are not included. [EXPERIMENTS.md](EXPERIMENTS.md) describes every experiment of the paper and how to run it with this code.

## Contents

```
main.py                   runs episodes with the settings in config.json5
simulation.py             Simulation: agent setup and the episode loop (flags, rewards, termination)
config.json5              settings read by main.py
config_manager.py         loads config.json5
src/
  environment/
    flg_graph.py               graph generation (Erdős–Rényi, Watts–Strogatz, Barabási–Albert), distances, demand
    users_behavior.py          similarities and Shapley attractions of users to facilities
    game_behavior.py           payoffs, payoffs at every candidate node (Algorithm 1), potential function
    best_response_dynamics.py  players and the best-response update
  regulatory_agent/
    drunk_agent.py             random (filtered)
    convergence_time/
      state_observation.py       the observation: flags b, w, z, h and the canonical sort
      reward.py                  reward functions
      Q_learning_tabular_agent/  the learned regulator
      simple_q_learning_agent.py plain Q-learning
      greedy_agent_time.py       greedy selection
      first_improvement_agent.py first-improvement
      pure_random_agent.py       random
      decision_tree_agent.py     a distilled decision tree used as the regulator
  tools/
    policy_extraction.py       relational dataset, CART trees, cross-validation, fidelity ceiling
    results_manager.py         saves the runs of main.py
    algorithm_tools.py         shortest paths and diameter
data/agent_data/Q_Learning/   a trained regulator (m = 25, n = 6, Erdős–Rényi)
```

## Installation

You need Python 3.10 or newer.

```
git clone https://github.com/Burjand/SFLG-BRD-RL-regulated.git
cd SFLG-BRD-RL-regulated
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Tested with Python 3.13, numpy 2.5, networkx 3.6, pandas 3.0, scikit-learn 1.9, matplotlib 3.11 and pyjson5 2.0.

Run everything from the repository root. The agents read and write their tables under `data/`, relative to the working directory.

## Quick start

```
python main.py
```

With the default config this does three things:

1. Loads the trained regulator stored in `data/agent_data/Q_Learning/`.
2. Runs 500 episodes, each on a fresh Erdős–Rényi graph with m = 25 nodes and n = 6 players.
3. Prints the mean settling time and plots it per episode.

Expect about 5.7 activations. For reference, greedy selection needs about 4.7 and uniform random about 14.7.

To try another activation rule, change `agent_type` in `config.json5`:

| `agent_type` | Rule in the paper | Chooses among |
|---|---|---|
| `RL` | learned regulator | all n ranks of the canonical order |
| `QL_Simple` | plain Q-learning | all n ranks |
| `Greedy` | greedy (maximum improvement) | all players |
| `FI` | first-improvement | players with b = 1 |
| `Random` | random | all players |
| `DecisionTree` | depth-5 tree distilled from the stored table when the run starts | all n ranks |
| any, with `regulated: false` | random (filtered) | players with b = 1 |

With `n_simulations: 1`, `main.py` runs a single episode and plots the potential, the positions and the utilities over time.

## Training a regulator

1. In `config.json5`, set `learn_from_scratch: true`.
2. Set `n_simulations` to the training budget. The paper uses 100·n² episodes, which is 3600 for n = 6.
3. Run `python main.py`.

During training the Boltzmann temperature is driven by the entropy controller. At the end, the Q-table and the Dyna model are written to `data/agent_data/Q_Learning/`, replacing the stored ones. `git checkout -- data/agent_data/Q_Learning` brings the stored ones back.

To evaluate the new table, set `learn_from_scratch: false` again.

A table is tied to the number of players it was trained with, because an observation has 4n entries. The same table can be used on other graph orders and graph families.

With `save_results: true`, each run's histories, config and figure are saved in `data/experiments/`.

## Using the code from Python

Every experiment in the paper builds a `Simulation` from a dictionary of hyperparameters and drives it directly:

```python
from simulation import Simulation

sim = Simulation(hp)                     # hp: see EXPERIMENTS.md for the full dictionary
histories = sim.run_simulations(3600)    # training if hp["learn_from_scratch"] is True
NE_t, t, potential, players, reward = sim.run_FLG_BRD_simulation()   # one episode
```

[EXPERIMENTS.md](EXPERIMENTS.md) gives the complete hyperparameter dictionary, plus tested helper functions to train, evaluate, time the rules and distill trees.

## Code and paper notation

| Paper | Code |
|---|---|
| m, n | `n_nodes`, `n_players` |
| β, similarity decay | `alpha` in `Users_behavior._compute_similarities` |
| σ_min | `min_similarity` |
| b, best-response flag | `players_find_best_response` (`BR_flags` in `State.observe`) |
| w, h, z flags | `wealth_flags`, `threat_flags`, `topology_flags` |
| Ranks 1, …, n | ranks 0, …, n−1 |
| Settling time T | `NE_t` (stored as `iters_to_NE` in the experiments) |
| Steps until the episode ends | `t` (`sim_iters`). An episode did not converge if `t == max_iter`. |
| Q₀, n_TD, n_plan | `optimistic_value`, `n_steps_bootstrap`, `planning_time` |
| τ₀, η_τ, H* | `boltzmann_tau`, `boltzmann_tau_lr`, `target_entropy` |
| τ_base, τ_boost, K | 0.01 (set at inference), `tau_boost=1.5`, `unchanged_threshold=3` (constructor defaults) |
| Regulator | `Q_learning_agent` (`agent_type: "RL"`) |
| Random (filtered) | `Drunk_agent` (`regulated: false`) |

In `policy_extraction.py`, the features are named `Is_Stable`, `Is_Poor`, `At_Hub`, `At_Crowded` (b, w, z, h), plus `Rank`. `Is_Stable` holds the raw b flag, so 1 means "not checked yet". The name is older than its current meaning. I kept it because the stored trees use these column names.

Ranks in a tree are 0-based too. For example, the split `Rank <= 3.5` separates ranks 1–4 of the paper from ranks 5–6.

## Implementation details

These are in the code but not spelled out in the paper:

- **Reward.**
  - A step that does not end the episode gets 10·(Φ(x_t) − Φ(x_{t−1})).
  - The final step (every flag cleared, or `max_iter` reached) gets Φ(x_T), unscaled, instead of the difference.
  - The factor 10 sets the scale on which Q₀ = 1 and the temperatures act.
- **Move tolerance.**
  - A player relocates only if its best gain exceeds 1e-5·max(1, |u|), where u is its current payoff. Payoffs are float32, and the tolerance keeps rounding noise from producing moves.
  - Greedy selection uses the same threshold when it looks for the best player.
  - First-improvement accepts a player whose gain over its stored payoff exceeds 1e-9. It can occasionally pick a player who then does not move.
- **Ties.**
  - A best response is the lowest-index node among equal payoffs.
  - Greedy breaks ties between players uniformly at random.
  - In the canonical sort, players with the same profile id stay in ascending player-ID order.
  - The decision tree picks the lowest rank among equal scores.
- **Randomness.**
  - Each episode samples a new graph (unless `fixed_graph: true`), seeded from a separate generator.
  - The agents and the dynamics use the global NumPy and `random` state, which `Simulation` seeds when it is built.
  - With the same seed and the same library versions, a run repeats exactly. `compute_player_best_response` draws from NumPy's generator on every call even though `rationality_epsilon` is 0. Removing that draw would shift every random stream after it.
- **Unused setting.** `graph_density` has no effect: the generators fix the mean degree at 6.

## The stored regulator

`data/agent_data/Q_Learning/` holds the regulator trained for the policy-extraction experiment (Section 6), with training seed 209:

- Erdős–Rényi graphs, m = 25, n = 6;
- 3600 training episodes;
- α = 0.10, γ = 0.2, H* = 1.5, n_TD = 1, n_plan = 2 (Table 4).

It has 315 visited observations. The seed-209 trees shown in Section 6 (depths 3 and 5) were distilled from it.

```
python -m src.tools.policy_extraction
```

This command distills it again (depth 5) and prints the fidelity ceilings of Table 10: 83.60% accuracy, 74.98% balanced accuracy and 50.12% F1.

## Citation

If you use this code, please cite the paper:

```
A. B. Torres-Reyes and R. Menchaca-Méndez, "Explainable Reinforcement Learning Regulators
for Best-Response Dynamics in Potential Games," submitted to Mathematics (MDPI), 2026.
```

The SFLG model comes from:

```
O. Ben-Porat and M. Tennenholtz, "Shapley Facility Location Games," arXiv:1709.10278, 2017.
https://doi.org/10.48550/arXiv.1709.10278
```

## License

MIT. See [LICENSE](LICENSE).
