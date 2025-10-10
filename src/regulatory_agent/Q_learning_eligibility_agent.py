import numpy as np
import pickle as pk, gzip, os
from collections import defaultdict

from src.regulatory_agent.Q_learning_agent import Q_learning_agent

class Q_lambda_agent(Q_learning_agent):
    """
    A Q-learning agent that uses Watkins' Q(λ) with eligibility traces.
    Inherits from the base Q_learning_agent and overrides the learning logic.
    """
    def __init__(self, max_possible_actions, alpha=0.1, gamma=0.9, epsilon=0.9, 
                 lambda_=0.9, epsilon_decay_rate=0.995, epsilon_min=0.001, 
                 optimistic_value=10, n=1, learn_from_scratch=True):
        
        # Call the parent's __init__ but we don't need 'n' anymore
        super().__init__(max_possible_actions, alpha, gamma, epsilon, 
                         epsilon_decay_rate, epsilon_min, optimistic_value, 
                         n=1, learn_from_scratch=learn_from_scratch)
        
        self.lambda_ = lambda_
        self.initialize_agent() # Re-initialize to add eligibility traces

    def initialize_agent(self):
        """
        Initializes the Q-table and adds the eligibility trace table.
        """

        super().initialize_agent()
        self.eligibility_traces = defaultdict(lambda: np.zeros(self.max_possible_actions))

    def learn(self, St, a, r, St_plus_1, done, t, T):
        """
        Updates the Q-table using the Watkins' Q(λ) algorithm.
        This method completely replaces the n-step logic from the parent class.

        Args:
            St (tuple): The current state.
            a (int): The action taken.
            r (float): The reward received.
            St_plus_1 (tuple): The next state.
            done (bool): True if the episode has terminated.
        """
        St = tuple(St)
        St_plus_1 = tuple(St_plus_1)
        
        # 1. Calculate the TD Error (δ)
        q_current = self.q_table[St][a]
        q_next_max = np.max(self.q_table[St_plus_1]) if not done else 0.0
        delta = r + self.gamma * q_next_max - q_current

        # 2. Increment the trace for the current state-action pair
        self.eligibility_traces[St][a] += 1

        # 3. Update all Q-values and decay all traces
        for s_key in self.q_table:
            # Update Q-values based on the TD error and trace value
            self.q_table[s_key] += self.alpha * delta * self.eligibility_traces[s_key]
            
            # Decay the traces
            self.eligibility_traces[s_key] *= self.gamma * self.lambda_

        # 4. Off-policy correction and episode reset
        best_action = np.argmax(self.q_table[St])
        if done or a != best_action:
            # Reset all traces
            self.eligibility_traces.clear()
