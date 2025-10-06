import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.simulation import Simulation
import src.environment.tools.general_tools as general_tools

if __name__ == "__main__":

    # --- CONFIGURATION ---
    CONFIGURATION = general_tools.extract_json_data("config.json5")

    # Simulation hyperparameters
    n_simulations = CONFIGURATION['n_simulations'] # Number of simulations to run
    seed = CONFIGURATION['seed']

    # Environment hyperparameters
    n_nodes = CONFIGURATION['n_nodes']    # Number of nodes in the graph
    n_potential_facilities = CONFIGURATION['n_potential_facilities'] # Number of potential facilities
    n_brd_players = CONFIGURATION['n_brd_players'] # Number of players in the BRD process
    max_iterations = CONFIGURATION['max_iterations'] # Max number of iterations before stopping an episode       
    convergence_threshold = CONFIGURATION['convergence_threshold'] # Threshold used to determine convergence of the potential function (Not used in this version of the code)
    demand_distribution = tuple(CONFIGURATION['demand_distribution']) # The distribution of the graph's demand (node weights)
    cost_distribution = tuple(CONFIGURATION['weight_distribution']) # The distribution of the graph's costs (edge weights)

    # Regulatory Agent hyperparameters
    learn_from_scratch = CONFIGURATION['learn_from_scratch']
    alpha = CONFIGURATION['alpha']
    gamma = CONFIGURATION['gamma']
    epsilon = CONFIGURATION['epsilon']
    epsilon_decay_rate = CONFIGURATION['epsilon_decay_rate']
    epsilon_min = CONFIGURATION['epsilon_min']
    optimistic_value = CONFIGURATION['optimistic_value']
    n_step_bootstrap = CONFIGURATION['n'] # n-steps bootstrap
    reward_function_name =  CONFIGURATION['reward_function_name']
    plan = CONFIGURATION['plan']
    planning_time = CONFIGURATION['planning_time']

    # --- SIMULATION ---
    # Setup simulation
    simulation = Simulation(n_nodes,
                            n_potential_facilities,
                            n_brd_players,
                            max_iterations,
                            seed,
                            demand_distribution,
                            cost_distribution,
                            learn_from_scratch,
                            alpha,
                            gamma,
                            epsilon,
                            epsilon_decay_rate,
                            epsilon_min,
                            optimistic_value,
                            n_step_bootstrap,
                            plan,
                            planning_time,
                            reward_function_name
                            )
    
    # Run simulations
    if n_simulations == 1:
        iterations, potential_function_development, players_development_over_time, episode_reward = simulation.run_FLG_BRD_simulation()
        simulation.show_simulation_results(iterations, potential_function_development, players_development_over_time, episode_reward, True)

    else:

        assert (isinstance(n_simulations, int) and n_simulations > 1)
        history_iterations, history_rewards, potential_function_developments_per_iteration, players_developments_per_iteration = simulation.run_simulations(n_simulations)
        simulation.show_multiple_simulations_results(history_iterations, history_rewards, potential_function_developments_per_iteration, players_developments_per_iteration, graph_smoothnes=5)
        





    

