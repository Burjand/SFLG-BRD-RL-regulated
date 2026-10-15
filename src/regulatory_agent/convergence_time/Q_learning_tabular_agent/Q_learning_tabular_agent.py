import numpy as np
import pickle as pk, gzip, os
from collections import defaultdict


class Q_learning_agent():

    def __init__(self, max_possible_actions, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay_rate=0.995, epsilon_min=0.001, optimistic_value=0, n=4, learn_from_scratch=True, action_selection="boltzmann", boltzmann_tau=1.0, boltzmann_tau_lr=0.001, target_entropy=0.4, tau_boost=1.5, unchanged_threshold=3, n_step_method="standard"):
        """
        Initialize the RL Agent. This is a tabular n-step Q-learning agent with Dyna planning.

        Args:
            max_possible_actions (int): The maximum number of actions the agent will handle (n_players, one per rank),
            alpha (float 0<alpha<=1): The step size the agent uses to consider the information provided by new experiences,
            gamma (float 0<gamma<=1): Discount factor,
            epsilon, epsilon_decay_rate, epsilon_min: epsilon-greedy schedule (only used with action_selection="epsilon-greedy"),
            optimistic_value (float): Initial Q-value of unvisited state-action pairs,
            n (positive int): n-steps for bootstrapping,
            learn_from_scratch (bool): True: empty tables and training mode. False: load the stored tables, inference mode,
            action_selection (str): "boltzmann" or "epsilon-greedy",
            boltzmann_tau (float): Initial temperature (training). At inference it is set to 0.01,
            boltzmann_tau_lr (float): Step size of the target-entropy temperature controller,
            target_entropy (float): Target entropy of the controller,
            tau_boost (float): Temperature used at inference when the observation keeps repeating,
            unchanged_threshold (int): Repetitions of the same observation before tau_boost is used,
            n_step_method (str): "standard" (greedy bootstrap) or "tree_backup" (Sutton & Barto 7.5).
        """
        self.learn_from_scratch = learn_from_scratch
        self.action_selection = action_selection
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
        self.tau = boltzmann_tau
        self.tau_lr = boltzmann_tau_lr
        self.target_entropy = target_entropy
        self.tau_boost = tau_boost
        self.unchanged_threshold = unchanged_threshold
        self.n_step_method = n_step_method

        self.states, self.actions, self.rewards = [], [], []
        self.initialize_agent()


    def _empty_tables(self):
        self.q_table = defaultdict(lambda: np.full(self.max_possible_actions, self.optimistic_value)) # Initialize values optimistically to promote exploration at the beginning
        # Model maps (state) -> (action) -> ( (reward, next_state) : count )
        self.model = defaultdict(
                lambda: defaultdict(  # For each state, create a dict for actions
                lambda: defaultdict(int) # For each action, create a dict for (reward, next_state): counts
            )
        )


    def initialize_agent(self):
        """
        Initialize the agent's base information. If learn_from_scratch == True creates empty q_table and model, if not, takes it from stored data
        """
        if self.learn_from_scratch:

            self._empty_tables()

        else:

            if os.path.exists(self.q_values_file_path):
                with gzip.open(self.q_values_file_path, 'rb') as f:
                    loaded_dict = pk.load(f)

                # Convert back to defaultdict
                self.q_table = defaultdict(
                    lambda: np.full(self.max_possible_actions, self.optimistic_value),
                    loaded_dict
                )

                if os.path.exists(self.model_file_path):
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
                    self.model = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

            else:
                print(f"{self.q_values_file_path} does not exist. Starting from an empty Q-table.")
                self._empty_tables()

        if not self.learn_from_scratch and self.action_selection == "boltzmann":

            self.tau = 0.01



    def choose_action(self, state, valid_actions, state_unchanged_count=0):
        """
        Choose an action based on the current observation.

        Args:
            state (list): The current (canonical) observation.
            valid_actions (list): The ranks the agent may choose.
            state_unchanged_count (int): Consecutive previous steps with the same observation. At inference, tau is boosted when it exceeds unchanged_threshold.

        Returns:
            int: The chosen action (rank).
        """
        # Get Q-values for this state (automatically initialized if new)
        state_key = tuple(state)
        q_values = self.q_table[state_key]
        if self.action_selection == 'epsilon-greedy':
            chosen_action = self._epsilon_greedy_action(q_values, valid_actions)
        else:
            chosen_action = self._boltzmann_action(q_values, valid_actions, state, state_unchanged_count)

        return chosen_action


    def _update_tau(self, probabilities, state):
        # When learning we use the entropy target technique
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        self.tau = self.tau - self.tau_lr * (entropy - self.target_entropy)
        self.tau = max(0.01, self.tau)


    def _boltzmann_action(self, q_values, valid_actions, state, state_unchanged_count=0):
        """
        Select an action using a Boltzmann distribution. Tau value will be managed taking into consideration
        if we are in inference or training mode.
        """
        valid_qs = np.array([q_values[i] for i in valid_actions]) # Select the q_values that correspond to the valid actions
        qs_shifted = valid_qs - np.max(valid_qs) # Avoid overflow in exp() (Because Softmax/Boltzmann is invariant if I shift with a constant)

        # During inference, boost tau if the observation has been repeating for too long (escapes aliased-state loops)
        if (not self.learn_from_scratch) and state_unchanged_count > self.unchanged_threshold:
            effective_tau = self.tau_boost
        else:
            effective_tau = self.tau

        exp_values = np.exp(qs_shifted / effective_tau)
        probabilities = exp_values / np.sum(exp_values)
        if self.learn_from_scratch:
            self._update_tau(probabilities, state)# Calculate tau

        return np.random.choice(valid_actions, p=probabilities) # Boltzmann selection


    def _epsilon_greedy_action(self, q_values, valid_actions):

        """
        Check the q_table for the action with the best q_value
        """
        valid_qs = [q_values[i] for i in valid_actions] # Select the q_values that correspond to the valid actions
        max_q = max(valid_qs) # Select the maximum q_value
        best_actions = [i for i in valid_actions if q_values[i] == max_q] # Select all the actions that have the best value
        greedy_action = np.random.choice(best_actions) # Choose randomly one of those actions
        exploratory_action = np.random.choice(valid_actions)
        return np.random.choice(a=[greedy_action, exploratory_action], p=[1-self.epsilon, self.epsilon])


    def compute_n_step_return(self, tau, bootstrap=True, T=None):
        """
        n-step return at index tau, with the method chosen by n_step_method:
          "standard"    : sum of up to n rewards + greedy bootstrap max_a Q(S_{tau+n}, a)
          "tree_backup" : n-step Tree Backup return (Sutton & Barto 7.5), target policy = Boltzmann at the current tau
        """
        if self.n_step_method == "tree_backup":
            return self._compute_tree_backup_return(tau, bootstrap, T)
        return self._compute_standard_n_step_return(tau, bootstrap, T)


    def _compute_standard_n_step_return(self, tau, bootstrap=True, T=None):
        """
        n-step return G_tau starting at time tau, using self.states and self.rewards
        without indexing past the end of the buffers.
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

        # bootstrap from state s_{tau + n} if it exists in the buffer and
        # the caller hasn't disabled bootstrapping (e.g., terminal state).
        if bootstrap and tau + n < len(self.states):
            s_boot = self.states[tau + n]
            G += (gamma ** n) * np.max(self.q_table[s_boot])

        return G


    def _soft_target_policy(self, state_key):
        """
        Boltzmann target policy pi(a|s) at the current tau over all ranks
        (the regulator always receives every rank as a valid action).
        """
        q = np.asarray(self.q_table[state_key], dtype=np.float64)
        q_shifted = q - np.max(q)
        tau_eff = max(self.tau, 0.01)
        exp_vals = np.exp(q_shifted / tau_eff)
        return exp_vals / np.sum(exp_vals)


    def _compute_tree_backup_return(self, tau, bootstrap=True, T=None):
        """
        n-step Tree Backup return (Sutton & Barto, Sec. 7.5):

            G_{t:h} = R_{t+1}
                   + gamma * sum_{a != A_{t+1}} pi(a|S_{t+1}) Q(S_{t+1}, a)
                   + gamma * pi(A_{t+1}|S_{t+1}) * G_{t+1:h}

        At the horizon: R_h + gamma * sum_a pi(a|S_h) Q(S_h, a), or R_h alone if terminal.
        Buffers: self.rewards[i] = R_{i+1}, self.states[i] = S_i, self.actions[i] = A_i.
        """
        n = self.n
        gamma = self.gamma
        R = len(self.rewards)

        if tau < 0 or tau >= max(1, len(self.states)):
            return 0.0
        if tau >= R:
            return 0.0

        k = min(n, R - tau)  # number of actual rewards available in the window

        # ---- LEAF / horizon term (j = k-1) ----
        leaf_reward = self.rewards[tau + k - 1]                  # = R_{tau+k}
        if bootstrap and tau + k < len(self.states):
            s_leaf = self.states[tau + k]                        # = S_{tau+k}
            pi_leaf = self._soft_target_policy(s_leaf)
            q_leaf = np.asarray(self.q_table[s_leaf], dtype=np.float64)
            G = leaf_reward + gamma * float(np.dot(pi_leaf, q_leaf))
        else:
            # Terminal at horizon, or no further state buffered: no bootstrap.
            G = leaf_reward

        # ---- Fold inward j = k-2 down to 0 ----
        for j in range(k - 2, -1, -1):
            r_j = self.rewards[tau + j]                          # = R_{tau+j+1}
            s_next = self.states[tau + j + 1]                    # = S_{tau+j+1}
            a_next = int(self.actions[tau + j + 1])              # = A_{tau+j+1}
            pi_next = self._soft_target_policy(s_next)
            q_next = np.asarray(self.q_table[s_next], dtype=np.float64)
            # Off-path expectation: sum_a pi(a) Q(s,a) - pi(a_next) Q(s, a_next)
            full_expect = float(np.dot(pi_next, q_next))
            off_path = full_expect - pi_next[a_next] * q_next[a_next]
            G = r_j + gamma * (off_path + pi_next[a_next] * G)

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

        tau = t - self.n

        if tau >= 0:
            G = self.compute_n_step_return(tau, bootstrap=not done)
            s_tau = self.states[tau]
            a_tau = self.actions[tau]
            self.q_table[s_tau][a_tau] += alpha * (G - self.q_table[s_tau][a_tau])

        if done:
            last_tau = len(self.rewards) - 1
            start_tau = max(0, tau + 1)
            for new_tau in range(start_tau, last_tau + 1):
                G = self.compute_n_step_return(new_tau, bootstrap=False)
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
        St = tuple(St)
        St_plus_1 = tuple(St_plus_1)

        self.model[St][a][r, St_plus_1] += 1

        alpha = self.alpha
        gamma = self.gamma

        for _ in range(planning_time):
            model_keys = list(self.model.keys())
            s = model_keys[np.random.randint(len(model_keys))]
            if not self.model[s]:
                continue

            action_keys = list(self.model[s].keys())
            a = action_keys[np.random.randint(len(action_keys))]
            if not self.model[s][a]:
                continue

            outcome_keys = list(self.model[s][a].keys())
            counts = np.array([self.model[s][a][k] for k in outcome_keys])
            probs = counts / counts.sum()
            r, s_next = outcome_keys[np.random.choice(len(outcome_keys), p=probs)]

            max_q = np.max(self.q_table[s_next])
            self.q_table[s][a] += alpha * (r + gamma * max_q - self.q_table[s][a])



    def memorize_knowledge(self):
        """
        Save the information learned so far by storing the q_table and the Dyna model
        """
        os.makedirs(os.path.dirname(self.q_values_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)

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
