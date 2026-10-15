import os
import matplotlib.pyplot as plt

from simulation import Simulation
from config_manager import global_config
from src.tools.results_manager import ResultsManager

if __name__ == "__main__":

    # Simulation hyperparameters
    n_simulations = global_config.get('n_simulations')
    save_results = global_config.get('save_results')

    simulation_hyperparameters = {
        # Simulation hyperparameters
        "seed": global_config.get('seed'),
        "regulated": global_config.get('regulated'),
        "fixed_graph": global_config.get('fixed_graph', False),

        # Environment hyperparameters
        "network_topology": global_config.get('network_topology'),
        "n_nodes": global_config.get('n_nodes'),
        "demand_distribution": tuple(global_config.get('demand_distribution')),
        "weight_distribution": tuple(global_config.get('weight_distribution')),
        "graph_density": global_config.get('graph_density'),

        # BRD Hyperparameters
        "rationality_epsilon": global_config.get('rationality_epsilon'),
        "n_players": global_config.get('n_players'),
        "max_iter": global_config.get('max_iter'),

        # Regulatory agent hyperparameters
        "agent_type": global_config.get("agent_type"),
        "learn_from_scratch": global_config.get('learn_from_scratch'),
        "action_selection": global_config.get("action_selection"),
        "alpha": global_config.get('alpha'),
        "gamma": global_config.get('gamma'),
        "optimistic_value": global_config.get('optimistic_value'),
        "n_steps_bootstrap": global_config.get('n_steps_bootstrap'),
        "n_step_method": global_config.get('n_step_method', 'standard'),
        "reward_function_name": global_config.get('reward_function_name'),
        "plan_enabled": global_config.get('plan'),
        "planning_time": global_config.get('planning_time'),
        # Regulatory agent action
        # epsilon-greedy
        "epsilon_min": global_config.get('epsilon_min'),
        "epsilon": global_config.get('epsilon') if global_config.get('learn_from_scratch') else 0,
        "epsilon_decay_rate": global_config.get('epsilon_decay_rate'),
        # Boltzmann
        "boltzmann_tau": global_config.get('boltzmann_tau'),
        "boltzmann_tau_lr": global_config.get('boltzmann_tau_lr'),
        "target_entropy": global_config.get('target_entropy'),
    }

    simulation = Simulation(simulation_hyperparameters)

    # Run simulations
    if n_simulations == 1:
        NE_iterations, simulation_iterations, potential_function_development, players_development_over_time, episode_reward = simulation.run_FLG_BRD_simulation()
        simulation.show_simulation_results(NE_iterations, simulation_iterations, potential_function_development, players_development_over_time, episode_reward, True)

    else:
        assert (isinstance(n_simulations, int) and n_simulations > 1)
        histories = simulation.run_simulations(n_simulations)
        (history_NE, history_sim, history_rew, pot_devs, player_devs, history_epsilon) = histories

        # Generate the Figure
        fig = simulation.show_multiple_simulations_results(
            history_NE, history_sim, history_rew, pot_devs, player_devs, history_epsilon
        )

        if save_results:
            print("\n[System] Saving results to disk...")

            # Prepare Data Packet
            results_payload = {
                'history_NE_iterations': history_NE,
                'history_simulation_iterations': history_sim,
                'history_rewards': history_rew,
                'history_epsilon': history_epsilon,
                'potential_function_developments': pot_devs,
                'players_developments': player_devs
            }

            # Initialize Manager
            res_manager = ResultsManager()

            try:
                # List all folders in the experiment directory
                existing_folders = [f for f in os.listdir(res_manager.base_dir) if os.path.isdir(os.path.join(res_manager.base_dir, f))]

                # Extract IDs (folders look like "1_Random_...")
                ids = []
                for folder in existing_folders:
                    try:
                        # Split by underscore and take the first part
                        ids.append(int(folder.split('_')[0]))
                    except (ValueError, IndexError):
                        pass # Ignore folders that don't follow the naming convention

                # Calculate next ID
                next_id = max(ids) + 1 if ids else 1

            except Exception as e:
                print(f"Warning: Could not determine next ID, defaulting to 1. Error: {e}")
                next_id = 1

            # Determine Agent Type Name
            if simulation.regulated:
                agent_type = f"Regulated_{simulation.reward_function_name}"
            else:
                agent_type = "Random"

            # Save with the dynamic ID
            res_manager.save_experiment(
                experiment_id=next_id,
                agent_type=agent_type,
                training=simulation.learn_from_scratch,
                config=global_config,
                results_data=results_payload,
                figure=fig
            )
        else:
            print("\n[System] Skipping save (save_results=False).")

        plt.show()
