"""
Plain tabular Q-learning ("plain Q-learning" baseline in the article): one-step
TD update, epsilon-greedy, no Dyna planning and no tau boost. Same canonical
state, rank-space actions and Q-table layout as the full agent.
"""
import numpy as np
import pickle as pk, gzip, os
from collections import defaultdict


class Simple_Q_learning_agent():

    def __init__(self, max_possible_actions, alpha=0.1, gamma=0.9, epsilon=0.9,
                 epsilon_decay_rate=0.995, epsilon_min=0.001, optimistic_value=0,
                 learn_from_scratch=True):
        """
        Initialize the simple Q-learning baseline.

        Args:
            max_possible_actions (int): Size of the (rank-space) action vector.
            alpha (float in (0, 1]): Step size.
            gamma (float in (0, 1]): Discount factor.
            epsilon (float in [0, 1]): Initial exploration rate (epsilon-greedy).
            epsilon_decay_rate (float in (0, 1]): Multiplicative decay applied
                once per episode by reduce_exploration_rate().
            epsilon_min (float): Lower floor for epsilon under decay.
            optimistic_value (float): Initial Q-value for unseen state/action.
            learn_from_scratch (bool): If True, start with an empty Q-table.
                If False, load it from disk.
        """
        self.learn_from_scratch = learn_from_scratch
        self.max_possible_actions = max_possible_actions
        self.q_values_file_path = 'data/agent_data/Q_Learning_Simple/q_values.pkl.gz'

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.optimistic_value = optimistic_value

        self.initialize_agent()


    def initialize_agent(self):
        """
        Initialize the Q-table. If learn_from_scratch == True, create an empty
        table with optimistic initial values; otherwise load it from disk.
        """
        if self.learn_from_scratch:
            self.q_table = defaultdict(
                lambda: np.full(self.max_possible_actions, self.optimistic_value)
            )
        else:
            if os.path.exists(self.q_values_file_path):
                with gzip.open(self.q_values_file_path, 'rb') as f:
                    loaded_dict = pk.load(f)
                self.q_table = defaultdict(
                    lambda: np.full(self.max_possible_actions, self.optimistic_value),
                    loaded_dict
                )
            else:
                print(f"{self.q_values_file_path} does not exist. Starting from an empty Q-table.")
                self.q_table = defaultdict(
                    lambda: np.full(self.max_possible_actions, self.optimistic_value)
                )


    def choose_action(self, state, valid_actions, state_unchanged_count=0):
        """
        Plain epsilon-greedy over the valid actions (state_unchanged_count is ignored).
        """
        state_key = tuple(state)
        q_values = self.q_table[state_key]

        if np.random.random() < self.epsilon:
            # Explore uniformly over valid actions
            return int(np.random.choice(valid_actions))

        # Exploit: pick the valid action with the highest Q-value, breaking
        # ties uniformly at random.
        valid_qs = np.array([q_values[i] for i in valid_actions])
        max_q = valid_qs.max()
        best_actions = [a for a, q in zip(valid_actions, valid_qs) if q == max_q]
        return int(np.random.choice(best_actions))


    def learn(self, St, a, r, St_plus_1, done, t, T=None):
        """
        One-step tabular Q-learning update:
            Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
        On terminal transitions the bootstrap term is dropped.
        """
        St = tuple(St)
        St_plus_1 = tuple(St_plus_1)

        # Float arrays, as in the full agent
        self.q_table[St] = np.asarray(self.q_table[St], dtype=np.float64)
        self.q_table[St_plus_1] = np.asarray(self.q_table[St_plus_1], dtype=np.float64)

        a = int(a)
        target = float(r)
        if not done:
            target += self.gamma * float(np.max(self.q_table[St_plus_1]))

        self.q_table[St][a] += self.alpha * (target - self.q_table[St][a])


    def plan(self, planning_time, St, a, r, St_plus_1):
        """
        No planning in this baseline.
        """
        pass


    def memorize_knowledge(self):
        """
        Save the learned Q-table to disk (no Dyna model in this agent).
        """
        os.makedirs(os.path.dirname(self.q_values_file_path), exist_ok=True)
        learned_q_values = dict(self.q_table)
        with gzip.open(self.q_values_file_path, 'wb') as f:
            pk.dump(learned_q_values, f, protocol=pk.HIGHEST_PROTOCOL)


    def reduce_exploration_rate(self):
        """
        Exponential decay with floor, same schedule as the full agent.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_rate
