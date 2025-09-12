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

                print("Using memorized knowledge.")

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


    def compute_n_step_return(self, tau, T):
        """
        Compute the n-step return G_{tau}^{(n)} starting at time tau,
        using self.states and self.rewards, without indexing past the end.
        """
        n = self.n
        gamma = self.gamma
        G = 0.0

        up_to = min(tau + n, T) # Determine the last time index for which we will include actual rewards.
        for i in range(tau+1, up_to): # Loop over reward indices r_{tau+1}, r_{tau+2}, ..., r_{up_to}.
            G += (gamma**(i - (tau+1))) * self.rewards[i] 

        # If the episode does NOT terminate within the n steps (i.e., tau + n < T), then we bootstrap from the value estimate at state s_{tau+n}.
        if tau + n < T:

            G += (gamma**n) * np.max(self.q_table[self.states[tau + n]])

        return G
    

    def learn(self, St, a, r, St_plus_1, done, t, T):
        """
        Updates the agent's internal model or policy based on the feedback
        received from the environment after taking an action.

        Args:
            St (np.array): The state before the action.
            a (int): The action taken.
            r (float): The reward received.
            St_plus_1 (np.array): The state after the action.
            done (bool): True if the episode has ended, False otherwise.
            t (int): The current time step
            T (int): Indicator of terminal state
        """
        n = self.n
        alpha = self.alpha
        gamma = self.gamma

        # Ensure states are hashable keys
        St = tuple(St)
        St_plus_1 = tuple(St_plus_1)

        # If this is the very first call in the episode, the states buffer may be empty.
        if self.states == []:
            self.states.append(St)
        
        # Append the next state, the action taken, and the reward observed.
        self.states.append(St_plus_1)
        self.actions.append(a)
        self.rewards.append(r)
        
        tau = t - n + 1 # Determine the time index whose n-step return is now available for update

        # If tau >= 0, we have collected enough subsequent rewards (or reached a terminal) to compute the n-step return for time tau and update Q(s_tau, a_tau)
        if tau >= 0:

            G = self.compute_n_step_return(tau, T)            
            self.q_table[self.states[tau]][self.actions[tau]] += alpha*(G - self.q_table[self.states[tau]][self.actions[tau]])

        if done:
            for new_tau_idx in range(tau + 1, len(self.actions)): # Iterate up to the last action's index

                G = self.compute_n_step_return(new_tau_idx, T)

                current_q_value = self.q_table[self.states[new_tau_idx]][self.actions[new_tau_idx]]
                self.q_table[self.states[new_tau_idx]][self.actions[new_tau_idx]] = current_q_value + alpha * (G - current_q_value)

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