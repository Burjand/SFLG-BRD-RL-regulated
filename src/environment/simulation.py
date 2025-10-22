import copy
import numpy as np
import matplotlib.pyplot as plt

from src.environment.facility_location_game import FLG_environment
from src.environment.tools.algorithm_tools import a_tools
from src.environment.tools.general_tools import flatten, pack
from src.environment.best_response_dynamics import BRD as BRD_setup
from src.regulatory_agent.regulatory_agent import Regulatory_Agent as RA_setup
from src.environment.rewards import Reward
from src.environment.observation import update_movement_flags

class Simulation():

    def __init__(self, n_nodes, n_potential_facilities, n_brd_players, max_iterations, seed, demand_distribution, weight_distribution, compact_states, num_boxes, distribution, k_movement_memory, learn_from_scratch, alpha, gamma, epsilon, epsilon_decay_rate, epsilon_min, optimistic_value, n_steps_bootstrap, plan, planning_time, reward_function_name, eligibility_lambda):

        # Environment
        self.n_nodes = n_nodes
        self.n_potential_facilities = n_potential_facilities
        self.n_brd_players = n_brd_players
        self.max_iterations = max_iterations
        self.seed = seed
        self.main_rng = np.random.default_rng(seed=self.seed)
        self.demand_distribution = demand_distribution
        self.weight_distribution = weight_distribution

        # State observation
        self.compact_states = compact_states
        self.num_boxes = num_boxes
        self.distribution = distribution
        self.k_movement_memory = k_movement_memory

        # Regulatory agent
        self.learn_from_scratch = learn_from_scratch
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon if learn_from_scratch else epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.optimistic_value = optimistic_value
        self.n = n_steps_bootstrap
        self.plan = plan
        self.planning_time = planning_time
        self.reward_function_name = reward_function_name
        self.eligibility_lambda = eligibility_lambda

        self.setup_simulation()


    def setup_simulation(self):

        """
        Initializes the whole simulation environment
        """

        # Setup environment
        # Generate the FLG environment
        self.FLG_env = FLG_environment(self.n_nodes,
                                       self.n_potential_facilities,
                                       seed = self.seed,
                                       demand_distribution = self.demand_distribution,
                                       weight_distribution = self.weight_distribution)
        assert all(isinstance(node, int) for node in self.FLG_env.graph.nodes()) # Check that nodes were correctly generated (Just in case)

        # Precompute flattened adjacency and demands as numpy arrays (static parts of the state)
        adj_flat = flatten(self.FLG_env.adj_matrix)
        self._adj_array = np.asarray(adj_flat, dtype=np.float64)
        demands = [int(d) for d in self.FLG_env.node_demand.values()]
        if self.compact_states:
            demands = pack(demands, self.num_boxes, self.distribution)
        self._demand_array = np.asarray(demands, dtype=np.float64)

        # Calculate all distances between nodes using Dijkstra's algorithm for computational efficiency
        self.distances = a_tools().calculate_distance_matrix(self.FLG_env.graph)

        # BRD players
        self.BRD = BRD_setup(self.n_brd_players,
                             self.distances,
                             self.FLG_env,
                             seed = self.seed)
        
        # Reward system
        self.reward_giver = Reward(self.reward_function_name)
        print(f"Using reward type '{self.reward_function_name}'")

        # Setup Regulatory Agent
        if not self.learn_from_scratch: self.epsilon = self.epsilon_min #When not learning use minimum epsilon
        self.regulatory_agent = RA_setup(max_possible_actions = self.n_brd_players,
                                         alpha = self.alpha,
                                         gamma = self.gamma,
                                         epsilon = self.epsilon,
                                         epsilon_decay_rate = self.epsilon_decay_rate,
                                         epsilon_min = self.epsilon_min,
                                         optimistic_value = self.optimistic_value,
                                         n = self.n,
                                         learn_from_scratch = self.learn_from_scratch)
        
        self._static_state = np.concatenate([self._adj_array, self._demand_array]).tolist()
        

    def run_FLG_BRD_simulation(self):

        """
        Handle the simulation process

        Returns:
            iterations (int): Shows how many iterations the simulation took
            potential_function_development (list): How the potential function changes for each iteration  
            players_development_over_time (list): Players' facility assignments for each iteration
        """

        # Local bindings to avoid attribute lookup overhead
        BRD = self.BRD
        RA = self.regulatory_agent
        reward_giver = self.reward_giver
        max_iter = self.max_iterations
        n_players = self.n_brd_players
        plan_enabled = self.plan
        planning_time = self.planning_time
        T = max_iter
        static_state = self._static_state

        # Best Response Dynamics process
        players_find_best_response = np.ones(n_players, dtype=np.bool_)
        players_positions = [BRD.players[p]['facility_position'] for p in range(BRD.n_players)]
        players_utilities = [BRD.players[p].get('Utility', 0.0) for p in range(n_players)]
        movement_flags = [0] * n_players
        movement_flags_memory = [0] * n_players

        St = static_state + players_positions + movement_flags #players_positions movement_flags [BRD.calculate_potential_function()] players_utilities
 
        len_players_pos = len(players_positions)
        len_players_ut = len(players_utilities)

        valid_actions = tuple(range(n_players))
        done = False

        t = 0 # Iterations
        potential_function_development = [] # Track how the potential function changes over time        
        players_development_over_time = [] # Track players' assignments over time
        episode_reward = 0
        r = 0

        # Main simulation loop        
        while t < max_iter and players_find_best_response.any():    

            t += 1    
        
            # Regulatory agent chooses action
            player_in_turn = RA.choose_action(tuple(St), valid_actions) # The action is the turn of a player
            action = player_in_turn

            # Environment simulation step
            updated = BRD.find_best_response(player_in_turn)
            if updated:
                players_find_best_response[:] = True  # Force recheck all players
            else:
                players_find_best_response[player_in_turn] = False

            done = False if any(players_find_best_response) and t < self.max_iterations else True # Check if simulation should end

            players_positions[:] = [BRD.players[p]['facility_position'] for p in range(n_players)]
            players_utilities[:] = [BRD.players[p].get('utility', 0.0) for p in range(n_players)]
            potential_function_current_value = BRD.calculate_potential_function()

            St_plus_1 = copy.deepcopy(St)
            movement_flags, movement_flags_memory = update_movement_flags(player_in_turn, movement_flags, movement_flags_memory, k=self.k_movement_memory)
            St_plus_1[-(len_players_pos + len_players_ut):] = players_positions + movement_flags #players_positions [potential_function_current_value] players_utilities movement_flags

            # Learning and planning
            done = not players_find_best_response.any() or t >= max_iter
            if done: # Reward at the end of the episode
                r = reward_giver.reward_function(t)
                episode_reward = r
                RA.reduce_exploration_rate()
                ### DEBUG ###
                #print(RA.epsilon)
            
            St_prepared = tuple(St)
            St_plus_1_prepared = tuple(St_plus_1)

            if self.learn_from_scratch:
                RA.learn(St_prepared, action, r, St_plus_1_prepared, done, t, T)
                if plan_enabled:
                    RA.plan(planning_time, St_prepared, action, r, St_plus_1_prepared)

            # Prepare for next iteration
            St = St_plus_1

            # Simulation development study            
            potential_function_development.append(potential_function_current_value)
            snapshot = {pid: copy.deepcopy(data) for pid, data in self.BRD.players.items()}
            players_development_over_time.append(snapshot)                

        # When the while loop ends it means that all player were not capable of finding a best response
        return t, potential_function_development, players_development_over_time, episode_reward

    
    def run_simulations(self, n_simulations):

        """
        Handle multiple simulations
        """
        
        # Variables for simulation study
        history_iterations = []
        potential_function_developments_per_iteration = []
        players_developments_per_iteration = []
        episode_count = 0
        history_rewards = []

        for i in range(n_simulations):
            
            if i%50 == 0:
                print(f"Running simulation {i+1}/{n_simulations}...")
                print(f"Current epsilon: {self.regulatory_agent.epsilon}")
            iterations, potential_function_development, players_development_over_time, episode_reward = self.run_FLG_BRD_simulation()
            episode_count += 1

            history_iterations.append(iterations) # (List)
            history_rewards.append(episode_reward) # (List)
            potential_function_developments_per_iteration.append(potential_function_development) # (List of lists)
            players_developments_per_iteration.append(players_development_over_time) # (List of lists)

        #avg_iterations = stats.mean(history_iterations)
        if self.learn_from_scratch:
            self.regulatory_agent.memorize_knowledge()

        return history_iterations, history_rewards, potential_function_developments_per_iteration, players_developments_per_iteration
    

    def show_simulation_results(self, iterations, potential_function_development, players_development_over_time, episode_reward, plot_results=False):

        # Show the results of the simulation
        print(f"Simulation completed in {iterations} iterations")
        print("Final players' positions:")
        for player_id, player_data in players_development_over_time[-1].items():
            print(f"Player {player_id}: Facility Position: {player_data['facility_position']}, Utility: {player_data['Utility']}")
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
            player_facility_positions = {pid: [] for pid in range(self.n_brd_players)}
            player_utilities = {pid: [] for pid in range(self.n_brd_players)}

            for snapshot in players_development_over_time:
                for pid, data in snapshot.items():
                    player_facility_positions[pid].append(data['facility_position'])
                    player_utilities[pid].append(data['Utility'])

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
            

    def show_multiple_simulations_results(self, history_iterations, history_rewards, potential_function_developments_per_iteration, players_developments_per_iteration, graph_smoothnes=25):

        # Show the results of the simulation
        # Plotting simulation development
            plt.figure(1, figsize=(10, 12))

            window = 25
            sum_iter = 0
            history_iterations_smoothed = []
            for i in range(len(history_iterations)):
                sum_iter += history_iterations[i]
                if (i+1) % window == 0:
                    history_iterations_smoothed.append(sum_iter/window)
                    sum_iter = 0

            x = [i+1 for i in range(len(history_iterations_smoothed))]
            print(f"Average amount of iterations: {np.mean(history_iterations)}")
            print(f"Standard deviation: {np.std(history_iterations)}")

            plt.title("Iterations necessary to achieve equilibrium (smoothed)")
            plt.xlabel(f"Simulation number (Averaged {window})")
            plt.ylabel("Average necessary iterations")
            plt.grid(True)
            plt.plot(x, history_iterations_smoothed)
            plt.tight_layout()

            """plt.figure(1, figsize=(10, 12))           

            window = graph_smoothnes
            sum_iter = 0
            history_rewards_smoothed = []
            for i in range(len(history_rewards)):
                sum_iter += history_rewards[i]
                if (i+1) % window == 0:
                    history_rewards_smoothed.append(sum_iter/window)
                    sum_iter = 0

            x = [i+1 for i in range(len(history_rewards_smoothed))]

            plt.title("Reward through each episode (smoothed)")
            plt.xlabel(f"Simulation number (Averaged {window})")
            plt.ylabel("Average reward")
            plt.grid(True)
            plt.plot(x, history_rewards_smoothed)
            plt.tight_layout()"""

            plt.show()
        

