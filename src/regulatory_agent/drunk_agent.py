import numpy as np

class Drunk_agent():

    def __init__(self):
        """
        Initialize the Drunk Agent. This agent chooses at random among the players
        that have not been checked yet ("Random (filtered)" in the article). It is
        used when regulated=False.
        """

    def choose_action(self, state, valid_actions, state_unchanged_count=0):

        return np.random.choice(valid_actions)

    def learn(self, St, a, r, St_plus_1, done, t, T=None):

        pass

    def plan(self, planning_time, St, a, r, St_plus_1):

        pass

    def memorize_knowledge(self):

        pass

    def reduce_exploration_rate(self):

        pass
