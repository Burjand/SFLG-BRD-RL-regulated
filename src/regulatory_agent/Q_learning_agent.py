import numpy as np
import pickle as pk, gzip, os
from collections import defaultdict


class Q_learning_agent():

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

        self.states, self.actions, self.rewards = [], [], []
        self.initialize_agent()

    
    def initialize_agent(self):
        """
        Initialize the agent's base information. If learn_from_scratch == True creates empty q_table and model, if not, takes it from stored data
        """
        if self.learn_from_scratch:

            self.q_table = defaultdict(lambda: np.full(self.max_possible_actions, self.optimistic_value)) # Initialize values optimistically to promote exploration at the beginning
            # Model maps (state) -> (action) -> ( (reward, next_state) : count )
            self.model = defaultdict(
                    lambda: defaultdict(  # For each state, create a dict for actions
                    lambda: defaultdict(int) # For each action, create a dict for (reward, next_state): counts
                )
            )

        else:

            if os.path.exists(self.q_values_file_path):
                with gzip.open(self.q_values_file_path, 'rb') as f:
                    loaded_dict = pk.load(f)
                
                # Convert back to defaultdict
                self.q_table = defaultdict(
                    lambda: np.full(self.max_possible_actions, self.optimistic_value),
                    loaded_dict
                )

                with gzip.open(self.model_file_path, 'rb') as f:
                    model_loaded_dict = pk.load(f)
                    self.model = defaultdict(
                            lambda: defaultdict(
                                lambda: defaultdict(int)
                            ), 
                            {k_s: defaultdict(lambda: defaultdict(int), 
                                {k_a: defaultdict(int, v_a) for k_a, v_a in v_s.items()}) 
                                for k_s, v_s in model_loaded_dict.items()}
                        )

            else:
                print(f"{self.q_values_file_path} does not exist.")


    def choose_action(self, state, valid_actions):
        """
        Choose an action based on the current observation.

        Args:
            state (list): The current state of the environment.
            valid_actions (list): The actions that are valid for ther agent to take (considering burning cells and out-of-bound cells)

        Returns:
            int: The chosen action (index).
        """
        # Get Q-values for this state (automatically initialized if new)
        state_key = tuple(state)
        q_values = self.q_table[state_key]

        # Epsilon-Greedy agent
        # OPTIMIZE THIS PART OF THE CODE
        greedy_action = self.greedy_action(q_values, valid_actions)
        exploratory_action = np.random.choice(valid_actions)
        chosen_action = np.random.choice(a=[greedy_action, exploratory_action], p=[1-self.epsilon, self.epsilon])

        return chosen_action
    
    
    def greedy_action(self, q_values, valid_actions):

        """
        Check the q_table for the action with the best q_value
        """

        valid_q = [q_values[i] for i in valid_actions] # Select the q_values that correspond to the valid actions
        max_q = max(valid_q) # Select the maximum q_value
        best_actions = [i for i in valid_actions if q_values[i] == max_q] # Select all the actions that have the best value
        return np.random.choice(best_actions) # Choose randomly one of those actions


    def compute_n_step_return(self, tau, T=None):
        """
        Compute the n-step return G_{tau}^{(n)} starting at time tau,
        using self.states and self.rewards, without indexing past the end.
        """
        n = self.n
        gamma = self.gamma
        G = 0.0

        R = len(self.rewards)  # number of rewards currently in buffer
        if tau < 0 or tau >= max(1, len(self.states)):  # defensive
            return 0.0

        # If there are no rewards available from tau, return 0
        if tau >= R:
            return 0.0

        # number of actual rewards we can sum starting at tau
        k = min(n, R - tau)  # k >= 1 when tau < R

        # sum the k actual rewards: r_{tau}, r_{tau+1}, ..., r_{tau+k-1}
        for j in range(k):
            G += (gamma ** j) * self.rewards[tau + j]

        # bootstrap if we have at least n rewards available after tau
        # i.e., if tau + n < R  (there exists rewards[tau + n])
        if tau + n < R:
            # bootstrap from state s_{tau + n}
            s_boot = self.states[tau + n]
            G += (gamma ** n) * np.max(self.q_table[s_boot])

        return G
    

    def learn(self, St, a, r, St_plus_1, done, t, T=None):
        alpha = self.alpha

        St = tuple(St)
        St_plus_1 = tuple(St_plus_1)

        # ensure float arrays
        self.q_table[St] = np.asarray(self.q_table[St], dtype=np.float64)
        self.q_table[St_plus_1] = np.asarray(self.q_table[St_plus_1], dtype=np.float64)

        if not self.states:
            self.states.append(St)

        self.states.append(St_plus_1)
        self.actions.append(int(a))
        self.rewards.append(float(r))

        tau = t - self.n + 1

        if tau >= 0:
            G = self.compute_n_step_return(tau)
            s_tau = self.states[tau]
            a_tau = self.actions[tau]
            self.q_table[s_tau][a_tau] += alpha * (G - self.q_table[s_tau][a_tau])

        if done:
            last_tau = len(self.rewards) - 1
            start_tau = max(0, tau + 1)
            for new_tau in range(start_tau, last_tau + 1):
                G = self.compute_n_step_return(new_tau)
                s_new = self.states[new_tau]
                a_new = self.actions[new_tau]
                self.q_table[s_new][a_new] += alpha * (G - self.q_table[s_new][a_new])

            self.states, self.actions, self.rewards = [], [], []



    def plan(self, planning_time, St, a, r, St_plus_1):
        """
        Updates the internal model and performs planning steps.

        Args:
            planning_time (int): Number of planning steps to perform.
            St (tuple/hashable): The current state from the real environment.
            a (int): The action taken in the real environment.
            r (float): The reward received from the real environment.
            St_plus_1 (tuple/hashable): The next state from the real environment.
        """

        if not self.model: # Check if the top-level dictionary is empty
            return

        alpha = self.alpha
        gamma = self.gamma
        St = tuple(St)
        St_plus_1 = tuple(St_plus_1)
        
        self.model[St][a][r, St_plus_1] += 1
        for _ in range(planning_time):
            s = np.random.choice(tuple(self.model.keys()))
            if not self.model[s]:
                continue # Skip this planning step if no actions for this state

            a = np.random.choice(tuple(self.model[s].keys()))
            if not self.model[s][a]:
                continue # Skip if no outcomes for this (s,a)

            total_count = np.sum(self.model[s][a].values())
            if total_count == 0:
                continue

            r, s_next = np.random.choice(a=tuple(self.model[s][a].keys()), p=[count/total_count for count in self.model[s][a].values()])
            max_q = np.max(self.q_table[s_next])
            self.q_table[s][a] += alpha*(r + gamma*max_q - self.q_table[s][a])



    def memorize_knowledge(self):
        """
        Save the information learned so far by storing the q_table
        """
        
        learned_q_values = dict(self.q_table)

        with gzip.open(self.q_values_file_path, 'wb') as f:
            pk.dump(learned_q_values, f, protocol=pk.HIGHEST_PROTOCOL)

        learned_model_for_saving = {}
        for s_key, actions_dict in self.model.items():
            learned_model_for_saving[s_key] = {}
            for a_key, outcome_counts_defaultdict in actions_dict.items():
                learned_model_for_saving[s_key][a_key] = dict(outcome_counts_defaultdict) # Convert inner defaultdict to dict

        with gzip.open(self.model_file_path, 'wb') as f:
            pk.dump(learned_model_for_saving, f, protocol=pk.HIGHEST_PROTOCOL)

    
    def reduce_exploration_rate(self):

        """
        Reduce exploration rate by inducing decay in epsilon
        """

        if self.epsilon > self.epsilon_min:
            
            self.epsilon *= self.epsilon_decay_rate