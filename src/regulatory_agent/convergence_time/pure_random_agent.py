import random


class PureRandom_agent():
    """
    Uniform random choice over ALL players ("Random" in the article).
    Used with regulated=True, so simulation.py passes the full player list.
    Drunk_agent is the filtered version ("Random (filtered)").
    """

    def __init__(self):
        self.epsilon = 0

    def choose_action(self, state, valid_actions, state_unchanged_count=0):
        return random.choice(valid_actions)

    def learn(self, St, a, r, St_plus_1, done, t, T=None):
        pass

    def plan(self, planning_time, St, a, r, St_plus_1):
        pass

    def memorize_knowledge(self):
        pass

    def reduce_exploration_rate(self):
        pass
