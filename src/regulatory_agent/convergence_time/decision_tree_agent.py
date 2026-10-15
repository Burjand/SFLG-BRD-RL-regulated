"""
Decision Tree agent trained with the Q-table of the RL agent
"""
from src.tools.policy_extraction import Extracted_policy
import pandas as pd
import numpy as np

class Decision_tree_agent():

    def __init__(self, max_depth=5, k=10, min_samples_leaf=50,
                 q_table_path=None, prebuilt_distiller=None,
                 run_bayes_diagnostic=True, run_cv=True):
        """
        Without arguments: loads the main Q-table (data/agent_data/Q_Learning/),
        runs the Bayes-rule diagnostic and k-fold CV, and trains a tree of depth max_depth.

          q_table_path        : use another Q-table file.
          prebuilt_distiller  : an already trained Extracted_policy; skips every other step.
          run_bayes_diagnostic: run the fidelity-ceiling diagnostic.
          run_cv              : run k-fold cross-validation.
        """
        if prebuilt_distiller is not None:
            self.distiller = prebuilt_distiller
        else:
            path = q_table_path or "data/agent_data/Q_Learning/q_values.pkl.gz"
            self.distiller = Extracted_policy(q_table_file_path=path)
            self.distiller.prepare_dataset()
            if run_bayes_diagnostic:
                self.distiller.bayes_optimal_diagnostic()
            if run_cv:
                self.distiller.cross_validate(k=k, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            self.distiller.train(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            print(self.distiller.get_feature_importance())
        self.epsilon = 0

    def choose_action(self, state, valid_actions, state_unchanged_count=0):

        if self.distiller.model is None:
            raise ValueError("Tree must be trained before use as a regulator.")

        n = len(valid_actions)
        num_base = len(self.distiller.feature_names)

        # Extract per-rank chunks of the canonical state (same as prepare_dataset)
        chunks = [state[i*n:(i+1)*n] for i in range(num_base)]

        # Build one row per rank with the same column structure as the training data
        rows = []
        for rank in range(n):
            profile = [int(chunks[i][rank]) for i in range(num_base)]
            profile.append(rank)
            rows.append(profile)

        X = pd.DataFrame(rows, columns=list(self.distiller.dataset_X.columns))
        proba = self.distiller.model.predict_proba(X)

        # proba[:, 1] is P(SELECT). Pick the rank with the highest SELECT probability (lowest rank on ties).
        return int(np.argmax(proba[:, 1]))

    def learn(self, St, a, r, St_plus_1, done, t, T=None):

        pass

    def plan(self, planning_time, St, a, r, St_plus_1):

        pass

    def memorize_knowledge(self):

        pass

    def reduce_exploration_rate(self):

        pass
