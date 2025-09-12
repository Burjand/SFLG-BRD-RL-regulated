import numpy as np

class Drunk_agent():

    def __init__(self, max_possible_actions, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay_rate=0.995, epsilon_min=0.001, optimistic_value=100, n=4, learn_from_scratch=True):
        """
        Initialize the RL Agent. This is a Q-learning agent.

        Args:
            max_possible_actions (int): The maximum number of action the agent will handle,
            alpha (float 0<alpha<=1): The step size the agent uses to consider the information provided by new experiences,
            gamma (float 0<alpha<=1),
            epsilon=0.1,
            epsilon_min,
            optimistic_value=100,
            n (positive int): n-steps for bootstrapping,
            learn_from_scratch=True
        """
        self.learn_from_scratch = learn_from_scratch
        self.max_possible_actions = max_possible_actions
        self.q_values_file_path = 'data/agent_data/Q_Learning/q_values.pkl.gz'
        self.model_file_path = 'data/agent_data/Q_Learning/model.pkl.gz'
        
        # Agent's hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.n = n
        self.optimistic_value = optimistic_value

    def choose_action(self, state, valid_actions):
        
        state_key = tuple(state)

        return np.random.choice(valid_actions)

    def learn(self, St, a, r, St_plus_1, done, t, T=None):
        
        pass

    def plan(self, planning_time, St, a, r, St_plus_1):
        
        pass

    def memorize_knowledge(self):
        
        pass
    
    def reduce_exploration_rate(self):

        pass