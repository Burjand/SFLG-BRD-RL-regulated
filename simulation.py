import numpy as np
import random
import matplotlib.pyplot as plt

from src.environment.flg_graph import FLG_graph
from src.environment.best_response_dynamics import BRD as BRD_setup
from src.regulatory_agent.convergence_time.state_observation import State
from src.regulatory_agent.convergence_time.reward import Reward
from src.regulatory_agent.convergence_time.Q_learning_tabular_agent.Q_learning_tabular_agent import Q_learning_agent
from src.regulatory_agent.convergence_time.simple_q_learning_agent import Simple_Q_learning_agent
from src.regulatory_agent.drunk_agent import Drunk_agent
from src.regulatory_agent.convergence_time.greedy_agent_time import Greedy_agent_time
from src.regulatory_agent.convergence_time.first_improvement_agent import First_improvement_agent
from src.regulatory_agent.convergence_time.decision_tree_agent import Decision_tree_agent
from src.regulatory_agent.convergence_time.pure_random_agent import PureRandom_agent

class Simulation():

    def __init__(self, simulation_hyperparameters):

        # Simulation hyperparameters
        self.seed = simulation_hyperparameters['seed']
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.regulated = simulation_hyperparameters['regulated']
        # True: one graph for all episodes. False: a fresh graph per episode (used in the article)
        self.fixed_graph = simulation_hyperparameters.get('fixed_graph', False)
        # Separate RNG for the per-episode graph seeds, so graph sampling does not interact with the agents' randomness
        self._graph_rng = np.random.default_rng(seed=self.seed)

        # Environment hyperparameters
        self.network_topology = simulation_hyperparameters['network_topology']
        self.n_nodes = simulation_hyperparameters['n_nodes']
        self.demand_distribution = simulation_hyperparameters['demand_distribution']
        self.weight_distribution = simulation_hyperparameters['weight_distribution']
        self.graph_density = simulation_hyperparameters['graph_density']

        # BRD Hyperparameters
        self.rationality_epsilon = simulation_hyperparameters['rationality_epsilon']
        self.n_players = simulation_hyperparameters['n_players']
        self.max_iter = simulation_hyperparameters['max_iter']

        # Regulatory agent hyperparameters
        self.agent_type = simulation_hyperparameters['agent_type']
        self.learn_from_scratch = simulation_hyperparameters['learn_from_scratch']
        self.action_selection = simulation_hyperparameters['action_selection']
        self.alpha = simulation_hyperparameters['alpha']
        self.gamma = simulation_hyperparameters['gamma']
        self.optimistic_value = simulation_hyperparameters['optimistic_value']
        self.n_steps_bootstrap = simulation_hyperparameters['n_steps_bootstrap']
        # "standard" (greedy bootstrap) or "tree_backup" (Sutton & Barto 7.5)
        self.n_step_method = simulation_hyperparameters.get('n_step_method', 'standard')
        self.reward_function_name = simulation_hyperparameters['reward_function_name']
        self.plan_enabled = simulation_hyperparameters['plan_enabled']
        self.planning_time = simulation_hyperparameters['planning_time']
        # Regulatory agent action
        # epsilon-greedy
        self.epsilon_min = simulation_hyperparameters['epsilon_min']
        self.epsilon = simulation_hyperparameters['epsilon']
        self.epsilon_decay_rate = simulation_hyperparameters['epsilon_decay_rate']
        # Boltzmann
        self.boltzmann_tau = simulation_hyperparameters['boltzmann_tau']
        self.boltzmann_tau_lr = simulation_hyperparameters['boltzmann_tau_lr']
        self.target_entropy = simulation_hyperparameters['target_entropy']

        self.setup_simulation()


    def setup_simulation(self):

        """ Initializes the whole simulation environment """

        # Setup environment
        # Generate the FLG environment
        flg_graph = FLG_graph(network_topology=self.network_topology, n_nodes=self.n_nodes, seed=self.seed, demand_distribution=self.demand_distribution, weight_distribution=self.weight_distribution, graph_density=self.graph_density)

        # BRD players
        self.BRD = BRD_setup(self.n_players, flg_graph, seed = self.seed, rationality_epsilon=self.rationality_epsilon)

        # Regulatory agent
        self.state = State(flg_graph, self.BRD)

        self.reward_giver = Reward(self.reward_function_name)

        if self.regulated:
            if self.agent_type=="RL":
                self.regulatory_agent = Q_learning_agent(
                                        max_possible_actions = self.n_players,
                                        alpha = self.alpha,
                                        gamma = self.gamma,
                                        epsilon = self.epsilon,
                                        epsilon_decay_rate = self.epsilon_decay_rate,
                                        epsilon_min = self.epsilon_min,
                                        optimistic_value = self.optimistic_value,
                                        n = self.n_steps_bootstrap,
                                        learn_from_scratch = self.learn_from_scratch,
                                        action_selection=self.action_selection,
                                        boltzmann_tau = self.boltzmann_tau,
                                        boltzmann_tau_lr = self.boltzmann_tau_lr,
                                        target_entropy = self.target_entropy,
                                        n_step_method = self.n_step_method
                                        )
            elif self.agent_type=="QL_Simple":
                self.regulatory_agent = Simple_Q_learning_agent(
                                        max_possible_actions = self.n_players,
                                        alpha = self.alpha,
                                        gamma = self.gamma,
                                        epsilon = self.epsilon,
                                        epsilon_decay_rate = self.epsilon_decay_rate,
                                        epsilon_min = self.epsilon_min,
                                        optimistic_value = self.optimistic_value,
                                        learn_from_scratch = self.learn_from_scratch
                                        )
            elif self.agent_type=="Greedy":
                self.regulatory_agent = Greedy_agent_time(self.n_players, self.BRD)
            elif self.agent_type=="FI":
                self.regulatory_agent = First_improvement_agent(self.n_players, self.BRD)
            elif self.agent_type=="Random":
                self.regulatory_agent = PureRandom_agent()
            elif self.agent_type=="DecisionTree":
                self.regulatory_agent = Decision_tree_agent()
            else:
                raise KeyError("Unrecognized agent type")
        else:
            self.regulatory_agent = Drunk_agent()


    def run_FLG_BRD_simulation(self):

        """
        Handle the simulation process (one episode)

        Returns:
            NE_t (int): Settling time, the step of the last relocation
            t (int): Steps the episode took (t == max_iter means it did not converge)
            potential_function_development (list): How the potential function changes for each iteration
            players_development_over_time (list): Players' facility assignments and utilities for each iteration
            episode_reward (float): Reward of the final step
        """

        if not self.fixed_graph:
            # Sample a fresh graph for this episode
            new_seed = int(self._graph_rng.integers(1, 2**31))
            flg_graph = FLG_graph(network_topology=self.network_topology, n_nodes=self.n_nodes, seed=new_seed, demand_distribution=self.demand_distribution, weight_distribution=self.weight_distribution, graph_density=self.graph_density)

            # BRD players
            self.BRD = BRD_setup(self.n_players, flg_graph, seed=new_seed, rationality_epsilon=self.rationality_epsilon)

            # Greedy and FI hold a reference to the BRD instance, so they are rebuilt with it
            self.state = State(flg_graph, self.BRD)
            if self.agent_type == "Greedy":
                self.regulatory_agent = Greedy_agent_time(self.n_players, self.BRD)
            elif self.agent_type == "FI":
                self.regulatory_agent = First_improvement_agent(self.n_players, self.BRD)
        # With fixed_graph the players are still re-placed every episode by BRD.create_players() in run_simulations()

        # Local bindings
        RA = self.regulatory_agent
        state = self.state

        ##### Best Response Dynamics process #####
        players_find_best_response = np.ones(self.n_players, dtype=np.bool_) # b flags: 1 = not checked since the last relocation
        done = False
        prev_pfv = self.BRD.game_behavior.calculate_potential_function(self.BRD.players)
        current_pfv = prev_pfv
        St = state.observe(players_find_best_response)

        T = self.max_iter

        # Variables for simulation study
        t = 0 # Iterations
        NE_t = 0
        potential_function_development = [] # Track how the potential function changes over time
        players_development_over_time = [] # Track players' assignments and utilities over time
        episode_reward = 0

        # Consecutive steps with the same observation. At inference it triggers the agent's tau boost
        prev_observed_state = None
        state_unchanged_count = 0

        # Main simulation loop
        while not done:

            t += 1

            # Regulated agents (except FI) choose among all players. FI and Random (filtered) only among
            # players flagged as 'needing a check', to avoid the coupon collector's problem in random selection
            if self.regulated and self.agent_type != "FI":
                current_valid_actions = [i for i in range(self.n_players)]
            else:
                current_valid_actions = [i for i, x in enumerate(players_find_best_response) if x]

            # Update compressed-state recurrence counter
            St_tuple = tuple(St)
            if prev_observed_state is not None and St_tuple == prev_observed_state:
                state_unchanged_count += 1
            else:
                state_unchanged_count = 0
            prev_observed_state = St_tuple

            # Regulatory agent chooses action
            # RL, QL_Simple and DecisionTree return a rank in the canonical order; the others return a player ID
            canonical_action = RA.choose_action(St_tuple, tuple(current_valid_actions), state_unchanged_count=state_unchanged_count)
            action = canonical_action  # learn() stores the rank-space action

            if self.regulated and self.agent_type in ("RL", "DecisionTree", "QL_Simple"):
                player_in_turn = state.canonical_to_player(canonical_action)
            else:
                player_in_turn = canonical_action

            updated = self.BRD.compute_player_best_response(player_in_turn)
            if updated:
                players_find_best_response[:] = True
                players_find_best_response[player_in_turn] = False
                NE_t = t
            else:
                players_find_best_response[player_in_turn] = False

            done = False if any(players_find_best_response) and t < self.max_iter else True # Check if simulation should end
            current_pfv = self.BRD.game_behavior.calculate_potential_function(self.BRD.players)
            St_plus_1 = state.observe(players_find_best_response)

            # Learning and planning
            if done: # Reward at the end of the episode: the final potential
                r = current_pfv
                episode_reward = r
                if self.learn_from_scratch: RA.reduce_exploration_rate()
            else:
                r = 10*self.reward_giver.reward_function(prev_pfv, current_pfv, t)

            St_prepared = tuple(St)
            St_plus_1_prepared = tuple(St_plus_1)

            if self.regulated and self.learn_from_scratch:
                RA.learn(St_prepared, action, r, St_plus_1_prepared, done, t, T)
                if self.plan_enabled:
                    RA.plan(self.planning_time, St_prepared, action, r, St_plus_1_prepared)

            # Prepare for next iteration
            St = St_plus_1
            prev_pfv = current_pfv

            # Simulation development study
            potential_function_development.append(current_pfv)
            snapshot = [
                (self.BRD.players[pid]['facility_position'], self.BRD.players[pid]['Utility'])
                for pid in range(self.n_players)
            ]
            players_development_over_time.append(snapshot)

        # The loop ends when no player could find a better response (pure NE) or at max_iter
        return NE_t, t, potential_function_development, players_development_over_time, episode_reward


    def run_simulations(self, n_simulations):

        """
        Handle multiple simulations
        """

        # Variables for simulation study
        history_NE_iterations = []
        history_simulation_iterations = []
        potential_function_developments_per_iteration = []
        players_developments_per_iteration = []
        episode_count = 0
        history_rewards = []
        history_epsilon = []

        for i in range(n_simulations):

            self.BRD.create_players() #Reset the players' positions
            if i%100 == 0:
                print(f"Running simulation {i}/{n_simulations}...")
            NE_iterations, simulation_iterations, potential_function_development, players_development_over_time, episode_reward = self.run_FLG_BRD_simulation()
            episode_count += 1

            history_NE_iterations.append(NE_iterations) # (List)
            history_simulation_iterations.append(simulation_iterations) # (List)
            history_rewards.append(episode_reward) # (List)
            potential_function_developments_per_iteration.append(potential_function_development) # (List of lists)
            players_developments_per_iteration.append(players_development_over_time) # (List of lists)

            if self.regulated and self.learn_from_scratch:
                history_epsilon.append(self.regulatory_agent.epsilon)
            else:
                history_epsilon.append(0)

        if self.learn_from_scratch:
            self.regulatory_agent.memorize_knowledge()

        return history_NE_iterations, history_simulation_iterations, history_rewards, potential_function_developments_per_iteration, players_developments_per_iteration, history_epsilon


    def show_simulation_results(self, NE_iterations, simulation_iterations, potential_function_development, players_development_over_time, episode_reward, plot_results=False):

        # Show the results of the simulation
        print(f"NE achieved in {NE_iterations} iterations")
        print(f"Simulation completed in {simulation_iterations} iterations")
        print("Final players' positions:")

        last_snapshot = players_development_over_time[-1]
        for player_id, (pos, util) in enumerate(last_snapshot):
            print(f"Player {player_id}: Facility Position: {pos}, Utility: {util}")

        print(f"Final Potential Function value: {potential_function_development[-1]}")
        print(f"Episode reward: {episode_reward}")

        if plot_results:

            # Plotting simulation development
            plt.figure(figsize=(10, 12))

            # Plot development of the Potential Function Over Time
            plt.subplot(3, 1, 1)  # 3 rows, 1 column, position 1
            plt.plot(potential_function_development, marker='o')
            plt.title("Potential Function Over Time")
            plt.xlabel("Iteration")
            plt.ylabel("Potential Function Value")
            plt.grid(True)

            # Data for the other two plots
            player_facility_positions = {pid: [] for pid in range(self.n_players)}
            player_utilities = {pid: [] for pid in range(self.n_players)}

            for snapshot in players_development_over_time:
                for pid, (pos, util) in enumerate(snapshot):
                    player_facility_positions[pid].append(pos)
                    player_utilities[pid].append(util)

            # Plot development of the Facility Positions Over Time
            plt.subplot(3, 1, 2)  # Position 2
            for pid, positions in player_facility_positions.items():
                plt.plot(positions, label=f'Player {pid}', marker='o')
            plt.title("Facility Positions Over Time")
            plt.xlabel("Iteration")
            plt.ylabel("Facility ID")
            plt.legend()
            plt.grid(True)

            # Plot development of the Utility Over Time
            plt.subplot(3, 1, 3)  # Position 3
            for pid, utils in player_utilities.items():
                plt.plot(utils, label=f'Player {pid}', marker='x')
            plt.title("Utility Over Time")
            plt.xlabel("Iteration")
            plt.ylabel("Utility")
            plt.legend()
            plt.grid(True)


            plt.tight_layout()
            plt.show()


    def show_multiple_simulations_results(self, history_NE_iterations, history_simulation_iterations, history_rewards, potential_function_developments_per_iteration, players_developments_per_iteration, history_epsilon, graph_smoothness=100):

        # --- Helper for Smoothing (EMA) ---
        def smooth_ema(data, alpha):
            if not data: return data
            smoothed = []
            last = data[0]
            for point in data:
                # alpha ~ 2 / (window_size + 1). So alpha=0.04 is roughly a window of 50.
                new_val = last * (1 - alpha) + point * alpha
                smoothed.append(new_val)
                last = new_val
            return smoothed

        # Determine alpha based on user's smoothness preference
        # If smoothness is 50, alpha is approx 0.04
        ema_alpha = 2 / (graph_smoothness + 1)

        # --- Create Figure with 2 Subplots ---
        # sharex=True aligns the Episode Numbers perfectly
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # --- Print Statistics ---
        print(f"Average amount of iterations per simulation: {np.mean(history_simulation_iterations):.2f}")
        print(f"Average amount of iterations to achieve NE: {np.mean(history_NE_iterations):.2f}")
        print(f"Standard deviation: {np.std(history_NE_iterations):.2f}")

        # --- PANEL 1: LEARNING CURVE (Performance) ---
        # 1. Plot Raw Data (Gray, transparent) to show the noise/variance
        ax1.plot(history_NE_iterations, color='gray', alpha=0.2, label='Raw Data (Noise)')

        # 2. Plot Smoothed Trend (Blue)
        smoothed_ne = smooth_ema(history_NE_iterations, alpha=ema_alpha)
        ax1.plot(smoothed_ne, color='blue', linewidth=2, label='Learning Trend')

        ax1.set_ylabel("Iterations to Reach NE")
        ax1.set_title(f"Performance Improvement (Avg: {np.mean(history_NE_iterations):.2f})")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # --- PANEL 2: EXPLORATION STRATEGY (Epsilon) ---
        ax2.plot(history_epsilon, color='orange', linewidth=2, label='Epsilon (Exploration Rate)')
        ax2.set_ylabel("Epsilon Value")
        ax2.set_xlabel("Simulation Episode Number")
        ax2.set_title("Exploration Decay")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        plt.tight_layout()

        # plt.show() is called in main.py, after saving
        return fig
