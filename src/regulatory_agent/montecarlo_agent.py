import numpy as np
import gzip, pickle as pk, os
from collections import defaultdict
from typing import Iterable, Tuple, Any

# IMplemented with ChatGPT for fast testing
class Montecarlo_agent:
    """
    On-policy First-Visit Monte-Carlo (episodic) agent with epsilon-soft policy (epsilon-greedy).
    - Updates Q(s,a) by incremental averaging of first-visit returns at episode end.
    - Does not maintain an explicit transition model (plan() is a no-op).
    - Saves/loads learned Q and counts to disk.
    """

    def __init__(self,
                 max_possible_actions: int,
                 gamma: float = 0.99,
                 epsilon: float = 0.5,
                 epsilon_decay_rate: float = 0.995,
                 epsilon_min: float = 0.01,
                 optimistic_value: float = 0.0,
                 learn_from_scratch: bool = True,
                 seed: int | None = None,
                 q_values_file_path: str = 'data/agent_data/MC/q_values.pkl.gz',
                 counts_file_path: str = 'data/agent_data/MC/counts.pkl.gz'):
        self.max_possible_actions = int(max_possible_actions)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay_rate = float(epsilon_decay_rate)
        self.epsilon_min = float(epsilon_min)
        self.optimistic_value = float(optimistic_value)
        self.learn_from_scratch = bool(learn_from_scratch)

        # RNG for reproducibility
        self.rng = np.random.default_rng(seed)

        # persistence paths
        self.q_values_file_path = q_values_file_path
        self.counts_file_path = counts_file_path

        # Q-table: defaultdict(state_key -> numpy array of length max_possible_actions)
        q_factory = lambda: np.full(self.max_possible_actions, self.optimistic_value, dtype=np.float64)
        self.q_table = defaultdict(lambda: q_factory())

        # counts: number of times we've seen (s,a) updated (for incremental average)
        counts_factory = lambda: np.zeros(self.max_possible_actions, dtype=np.int64)
        self.returns_count = defaultdict(lambda: counts_factory())

        # per-episode buffers
        self._episode_states: list[Tuple] = []    # list of state-keys (tuple)
        self._episode_actions: list[int] = []
        self._episode_rewards: list[float] = []

        # initialize (optionally load existing)
        if not learn_from_scratch:
            self._load_from_disk()

    # -------------------------
    # Public interface methods
    # -------------------------

    def choose_action(self, state: Iterable, valid_actions: Iterable[int]):
        """
        Epsilon-greedy selection. `state` can be any iterable (Simulation passes np.array).
        """
        state_key = tuple(state)  # convert to hashable key
        q_values = self.q_table[state_key]
        valid_actions = tuple(valid_actions)

        if len(valid_actions) == 1:
            return int(valid_actions[0])

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))
        else:
            # greedy among valid_actions with random tie-breaking
            vals = np.array([q_values[a] for a in valid_actions], dtype=np.float64)
            maxval = vals.max()
            tie_mask = np.isclose(vals, maxval)
            candidates = np.asarray(valid_actions)[tie_mask]
            return int(self.rng.choice(candidates))

    def learn(self, St, a: int, r: float, St_plus_1, done: bool, t: int, T: int | None = None):
        """
        Record the transition. On episode end (done=True) perform First-Visit MC updates.
        Simulation calls this each step; the MC agent accumulates until done.
        """
        s = tuple(St)
        s_next = tuple(St_plus_1)

        # If buffers empty, start episode by recording current state
        if not self._episode_states:
            self._episode_states.append(s)

        # append transition (reward leads to s_next)
        self._episode_actions.append(int(a))
        self._episode_rewards.append(float(r))
        self._episode_states.append(s_next)

        if done:
            self._finish_episode_and_update()
            # After updating, optionally decay epsilon here or leave it to Simulation.reduce_exploration_rate
            # self.reduce_exploration_rate()

    def plan(self, planning_time: int, St, a: int, r: float, St_plus_1):
        """
        No explicit planning for this model-free Monte-Carlo agent.
        Kept for interface compatibility. If desired, a lightweight model-based replay could be added.
        """
        return

    def reduce_exploration_rate(self):
        """Decay epsilon but enforce the minimum."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

    def memorize_knowledge(self):
        """Persist Q-table and counts to disk."""
        # ensure folder exists
        for path in (self.q_values_file_path, self.counts_file_path):
            parent = os.path.dirname(path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

        # Convert defaultdicts to plain dicts suitable for pickle (numpy arrays are picklable)
        q_dict = {k: np.asarray(v) for k, v in self.q_table.items()}
        with gzip.open(self.q_values_file_path, 'wb') as f:
            pk.dump(q_dict, f, protocol=pk.HIGHEST_PROTOCOL)

        counts_dict = {k: np.asarray(v) for k, v in self.returns_count.items()}
        with gzip.open(self.counts_file_path, 'wb') as f:
            pk.dump(counts_dict, f, protocol=pk.HIGHEST_PROTOCOL)

    # -------------------------
    # Optional helpers for parallel experiments
    # -------------------------
    def export_state(self) -> dict:
        """Return a picklable snapshot of q_table and counts."""
        return {
            "q_table": {k: np.asarray(v) for k, v in self.q_table.items()},
            "counts": {k: np.asarray(v) for k, v in self.returns_count.items()}
        }

    def import_state(self, state_dict: dict):
        """
        Merge another exported state into this agent.
        Simple merge: if a state-action pair exists in both, use counts-weighted average of Q.
        """
        incoming_q = state_dict.get("q_table", {})
        incoming_counts = state_dict.get("counts", {})

        for s_k, q_vals in incoming_q.items():
            q_vals = np.asarray(q_vals, dtype=np.float64)
            inc_counts = np.asarray(incoming_counts.get(s_k, np.zeros_like(q_vals)), dtype=np.int64)

            if s_k not in self.q_table:
                self.q_table[s_k] = q_vals.copy()
                self.returns_count[s_k] = inc_counts.copy()
            else:
                # weighted average: (q1*c1 + q2*c2) / (c1 + c2) for each action where c1+c2>0
                current_q = self.q_table[s_k]
                current_c = self.returns_count[s_k]
                total_c = current_c + inc_counts
                # avoid division by zero
                nonzero = total_c > 0
                merged_q = current_q.copy()
                merged_q[nonzero] = (current_q[nonzero] * current_c[nonzero] + q_vals[nonzero] * inc_counts[nonzero]) / total_c[nonzero]
                self.q_table[s_k] = merged_q
                self.returns_count[s_k] = total_c

    # -------------------------
    # Internal helpers
    # -------------------------
    def _finish_episode_and_update(self):
        """
        Perform First-Visit Monte-Carlo updates using recorded episode:
            states: [s0, s1, ..., sN]
            actions: [a0, a1, ..., a_{N-1}]
            rewards: [r0, r1, ..., r_{N-1}]  where r_i is reward after taking a_i leading to s_{i+1}
        We compute returns G_t = sum_{k=t}^{N-1} gamma^{k-t} r_k
        Then for each first-visit (s_t, a_t) we update:
            N <- N + 1
            Q <- Q + (G - Q) / N
        """
        states = self._episode_states
        actions = self._episode_actions
        rewards = self._episode_rewards
        T = len(actions)  # number of steps in episode

        # compute returns G_t for all t by reverse accumulation
        G = 0.0
        returns = [0.0] * T
        for t in range(T - 1, -1, -1):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # track first-visit occurrences
        seen_state_action = set()
        for t in range(T):
            s_t = states[t]
            a_t = actions[t]
            key = (s_t, a_t)
            if key in seen_state_action:
                continue
            seen_state_action.add(key)

            # incremental average update
            self.returns_count[s_t][a_t] += 1
            n = self.returns_count[s_t][a_t]
            current_q = self.q_table[s_t][a_t]
            G_t = returns[t]
            # Q <- Q + (G - Q) / N
            self.q_table[s_t][a_t] = current_q + (G_t - current_q) / n

        # clear buffers
        self._episode_states.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()

    def _load_from_disk(self):
        """Load Q and counts if files exist."""
        try:
            if os.path.exists(self.q_values_file_path):
                with gzip.open(self.q_values_file_path, 'rb') as f:
                    q_dict = pk.load(f)
                for k, v in q_dict.items():
                    self.q_table[k] = np.asarray(v, dtype=np.float64)

            if os.path.exists(self.counts_file_path):
                with gzip.open(self.counts_file_path, 'rb') as f:
                    counts_dict = pk.load(f)
                for k, v in counts_dict.items():
                    self.returns_count[k] = np.asarray(v, dtype=np.int64)

            print("MonteCarloAgent: loaded saved knowledge.")
        except Exception as e:
            print("MonteCarloAgent: failed to load saved files:", e)
            # leave defaults (fresh agent)