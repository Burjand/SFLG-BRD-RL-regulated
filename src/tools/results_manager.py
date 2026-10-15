import os
import json
import gzip
import pickle
import datetime
import matplotlib.pyplot as plt

class ResultsManager:
    def __init__(self, base_dir="data/experiments"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_experiment(self, experiment_id, agent_type, training, config, results_data, figure=None):
        """
        Saves experiment data, config, AND the plot image to a unique folder.

        Args:
            experiment_id (int/str): Identifier.
            agent_type (str): Name of the agent.
            config (dict): The configuration used.
            results_data (dict): The history lists.
            figure (matplotlib.figure.Figure, optional): The plot to save.
        """
        # 1. Create unique folder name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_type != 'Random':
            if training:
                folder_name = f"{experiment_id}_{agent_type}_train_{timestamp}"
            else:
                folder_name = f"{experiment_id}_{agent_type}_test_{timestamp}"
        else:
            folder_name = f"{experiment_id}_{agent_type}_{timestamp}"
        folder_path = os.path.join(self.base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # 2. Save Configuration
        config_path = os.path.join(folder_path, "config.txt")
        try:
            config_dict = config if isinstance(config, dict) else config.__dict__
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
        except Exception:
            with open(config_path, 'w') as f:
                f.write(str(config))

        # 3. Save Results Data
        data_path = os.path.join(folder_path, "results.pkl.gz")
        with gzip.open(data_path, 'wb') as f:
            pickle.dump(results_data, f)

        # 4. Save Figure Image
        if figure:
            image_path = os.path.join(folder_path, "iterations_graph.png")
            figure.savefig(image_path, dpi=300, bbox_inches='tight')
            print(f"[ResultsManager] Graph image saved to: {image_path}")

        print(f"[ResultsManager] Experiment saved successfully at: {folder_path}")
        return folder_path

    def load_experiment(self, folder_path):
        """Loads the results dictionary from a specific experiment folder."""
        data_path = os.path.join(folder_path, "results.pkl.gz")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No results file found at {data_path}")

        with gzip.open(data_path, 'rb') as f:
            data = pickle.load(f)

        return data

    def plot_comparison(self, folder_paths, labels=None, smoothness=25):
        """
        Loads multiple experiments and plots their NE Iterations on the same graph.

        Args:
            folder_paths (list): List of directory paths to compare.
            labels (list): List of legend labels (optional).
            smoothness (int): Rolling average window.
        """
        plt.figure(figsize=(12, 6))

        for i, folder in enumerate(folder_paths):
            try:
                data = self.load_experiment(folder)
                ne_history = data['history_NE_iterations']

                # Smooth data
                smoothed = []
                window_sum = 0
                for j, val in enumerate(ne_history):
                    window_sum += val
                    if (j + 1) % smoothness == 0:
                        smoothed.append(window_sum / smoothness)
                        window_sum = 0

                # Plot
                label = labels[i] if labels else os.path.basename(folder)
                plt.plot(smoothed, label=label, linewidth=2)

            except Exception as e:
                print(f"Failed to load {folder}: {e}")

        plt.title(f"Convergence Comparison (Smoothed window={smoothness})")
        plt.xlabel(f"Simulation Batches (x{smoothness})")
        plt.ylabel("Iterations to NE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
