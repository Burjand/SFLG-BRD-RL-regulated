import sys
import os

import numpy as np
from Q_learning_agent import Q_learning_agent
from montecarlo_agent import Montecarlo_agent
from Q_learning_eligibility_agent import Q_lambda_agent
from collections import defaultdict

def simple_bandit_test():
    # Two actions: action 0 -> reward +1 (good), action 1 -> reward 0 (bad)
    """agent = Q_learning_agent(max_possible_actions=2, alpha=0.2, gamma=0.9,
                             epsilon=0.5, epsilon_decay_rate=0.995, epsilon_min=0.01,
                             optimistic_value=0, learn_from_scratch=True, n=1)"""
    """agent = Montecarlo_agent(max_possible_actions=2, gamma=0.9,
                             epsilon=0.5, epsilon_decay_rate=0.995, epsilon_min=0.01,
                             optimistic_value=0, learn_from_scratch=True)"""
    agent = Q_lambda_agent(max_possible_actions=2, alpha=0.2, gamma=0.9,
                             epsilon=0.5, epsilon_decay_rate=0.995, epsilon_min=0.01,
                             optimistic_value=0, learn_from_scratch=True, n=1)

    n_episodes = 2000
    choose_counts = [0,0]
    for ep in range(n_episodes):
        state = (0,)  # single-state
        # choose action
        a = agent.choose_action(state, valid_actions=(0,1))
        # deterministic reward
        r = 1.0 if a == 0 else 0.0
        # single-step episode: s->a->r->s (done)
        agent.learn(state, a, r, state, done=True, t=0, T=1)
        agent.reduce_exploration_rate()
        choose_counts[a] += 1

    print("action counts (0 good, 1 bad):", choose_counts)
    print("q_table sample:", agent.q_table[state])

if __name__ == "__main__":
    simple_bandit_test()