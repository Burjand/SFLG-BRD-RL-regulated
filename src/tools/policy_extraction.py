import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import pickle as pk
import gzip
import os

class Extracted_policy:
    """
    Clase para destilar una Tabla Q de RL en un Árbol de Decisión Interpretable usando
    un enfoque relacional.
    """

    def __init__(self, q_table_file_path=None, q_table_dict=None):
        """
        Puede inicializarse con una ruta de archivo O con un diccionario en memoria.
        """
        # Flags b, w, z, h of the article, in this order. Is_Stable holds the raw b flag (1 = not checked yet)
        self.feature_names = ["Is_Stable", "Is_Poor", "At_Hub", "At_Crowded"]
        self.model = None
        self.dataset_X = None
        self.dataset_y = None
        self.q_table = {}

        self.seed = 43

        # Carga de datos
        if q_table_dict:
            self.q_table = q_table_dict
        elif q_table_file_path:
            self.q_table = self.load_raw_q_table(q_table_file_path)
        else:
            print("Warning: DistilledPolicy initialized empty. Load data manually.")

    def load_raw_q_table(self, file_path):
        if os.path.exists(file_path):
            try:
                with gzip.open(file_path, 'rb') as f:
                    loaded_dict = pk.load(f)
                print(f"Q-Table loaded successfully from {file_path}")
                print(f"Total states: {len(loaded_dict)}")
                return loaded_dict
            except Exception as e:
                print(f"Error loading file: {e}")
                return {}
        else:
            print(f"Error: File {file_path} does not exist.")
            return {}

    def prepare_dataset(self):
        """
        Transforma la Q-Table en un Dataset Relacional: una fila por (estado, rank),
        etiqueta 1 si ese rank es el argmax de la Q-Table.
        """
        if not self.q_table:
            raise ValueError("No Q-Table loaded.")

        relational_X = []
        relational_y = []

        num_features = len(self.feature_names)

        # Iterar sobre la tabla Q
        for state_tuple, q_values in self.q_table.items():

            # Filtro: Ignorar estados no visitados (todo 0)
            if np.sum(np.abs(q_values)) == 0:
                continue

            n_features_total = len(state_tuple)

            # La longitud de la tupla debe ser un múltiplo exacto de las features
            if n_features_total % num_features != 0:
                continue

            n_players = n_features_total // num_features

            # Acción óptima (rank elegido por el agente)
            chosen_player_id = np.argmax(q_values)

            # Separar el estado en bloques [b_0..b_n, w_0..w_n, z_0..z_n, h_0..h_n]
            chunks = []
            for i in range(num_features):
                start = i * n_players
                end = (i + 1) * n_players
                chunks.append(state_tuple[start:end])

            # Una fila por cada rank
            for pid in range(n_players):
                player_profile = [chunk[pid] for chunk in chunks]

                # The rank is a 5th feature: players with the same 4 flags can be treated differently by rank
                player_profile.append(pid)

                # Etiqueta: 1 si este rank fue el elegido, 0 si no
                is_selected = 1 if pid == chosen_player_id else 0

                relational_X.append(player_profile)
                relational_y.append(is_selected)

        # Guardar como DataFrame interno (columns: base features + Rank)
        extended_feature_names = self.feature_names + ["Rank"]
        self.dataset_X = pd.DataFrame(relational_X, columns=extended_feature_names)
        self.dataset_y = pd.Series(relational_y)

        print(f"Relational dataset generated: {len(self.dataset_X)} instances.")
        print(f"Selection distribution (balance): {self.dataset_y.mean():.2%}")

    def train(self, max_depth=4, min_samples_leaf=50):
        """
        Entrena el Árbol de Decisión.
        """
        if self.dataset_X is None:
            self.prepare_dataset()

        # class_weight='balanced' porque hay muchos más casos de "No Elegir"
        self.model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=self.seed
        )

        self.model.fit(self.dataset_X, self.dataset_y)

        # Calcular Accuracy
        acc = self.model.score(self.dataset_X, self.dataset_y)
        print(f"Model trained. Fidelity (training accuracy): {acc:.2%}")

    def cross_validate(self, k=5, max_depth=4, min_samples_leaf=50):
        """
        Stratified K-Fold cross-validation over the relational dataset, with the
        same tree hyperparameters as train(). Reports accuracy, balanced accuracy
        and F1 of the SELECT class (mean +/- std over the folds). Does not modify self.model.
        """
        if self.dataset_X is None:
            self.prepare_dataset()

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)

        accuracies = []
        balanced_accuracies = []
        f1_scores = []

        for train_idx, test_idx in skf.split(self.dataset_X, self.dataset_y):
            X_train = self.dataset_X.iloc[train_idx]
            y_train = self.dataset_y.iloc[train_idx]
            X_test  = self.dataset_X.iloc[test_idx]
            y_test  = self.dataset_y.iloc[test_idx]

            fold_model = DecisionTreeClassifier(
                criterion='gini',
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=self.seed
            )
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
            balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        print(f"\nStratified K-Fold cross-validation (K={k}):")
        print(f"  Accuracy:               {np.mean(accuracies):.2%} +/- {np.std(accuracies):.2%}")
        print(f"  Balanced accuracy:      {np.mean(balanced_accuracies):.2%} +/- {np.std(balanced_accuracies):.2%}")
        print(f"  F1 (SELECT class):      {np.mean(f1_scores):.2%} +/- {np.std(f1_scores):.2%}")

        return {
            'accuracy': accuracies,
            'balanced_accuracy': balanced_accuracies,
            'f1': f1_scores,
        }

    def plot(self, save_path="distilled_tree.png", show=True):
        """
        Genera y guarda la visualización del árbol.
        """
        if self.model is None:
            print("Error: You must call .train() before plotting.")
            return

        plt.figure(figsize=(12, 8))
        plot_tree(self.model,
                  feature_names=list(self.dataset_X.columns),
                  class_names=["IGNORE", "SELECT"],
                  filled=True,
                  rounded=True,
                  fontsize=10)

        plt.title("Player selection heuristic (Extracted Policy)", fontsize=14)

        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved at {save_path}")

        if show:
            plt.show()

    def get_feature_importance(self):
        """
        Retorna un DataFrame con la importancia de cada variable.
        """
        if self.model is None:
            return None

        imp = pd.DataFrame({
            'Feature': list(self.dataset_X.columns),
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        return imp

    def bayes_optimal_diagnostic(self):
        """
        Ceiling on the in-sample accuracy, balanced accuracy and F1 that any
        deterministic classifier on these features can reach (rows with identical
        features must get the same prediction). Section 6.2 of the article.

          - Accuracy: per-group majority label.
          - Balanced accuracy: predict 1 iff the group's positive rate exceeds the overall rate.
          - F1: best prefix of the groups sorted by positive rate (Lipton, Elkan, Naryanaswamy 2014).
        """
        if self.dataset_X is None:
            self.prepare_dataset()

        feature_columns = list(self.dataset_X.columns)
        df = self.dataset_X.copy()
        df['_label'] = self.dataset_y.values

        # Per-group statistics: size n_g and positive count p_g.
        group_stats = df.groupby(feature_columns)['_label'].agg(['size', 'sum'])
        group_stats = group_stats.rename(columns={'size': 'n', 'sum': 'p'})
        n_unique = len(group_stats)
        n_total = len(df)

        y_true = self.dataset_y.values
        pos_rate_global = float(y_true.mean())
        total_p = int(group_stats['p'].sum())

        # Per-group local positive rate, broadcast back to row level
        group_pos_rate = df.groupby(feature_columns)['_label'].transform('mean').values

        # Accuracy ceiling: per-group majority rule.
        y_pred_acc = (group_pos_rate > 0.5).astype(int)
        bayes_acc = accuracy_score(y_true, y_pred_acc)

        # Balanced accuracy ceiling: per-group threshold at the population positive rate.
        y_pred_bal = (group_pos_rate > pos_rate_global).astype(int)
        bayes_bal_acc = balanced_accuracy_score(y_true, y_pred_bal)

        # F1 ceiling: predict 1 on the top-k groups by positive rate, F1 = 2 * cum_TP / (cum_pred_pos + total_p), best k
        sorted_groups = group_stats.assign(rate=group_stats['p'] / group_stats['n']) \
                                   .sort_values('rate', ascending=False)
        cum_tp = sorted_groups['p'].cumsum().values
        cum_pred_pos = sorted_groups['n'].cumsum().values
        denom = cum_pred_pos + total_p
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_curve = np.where(denom > 0, 2.0 * cum_tp / denom, 0.0)
        bayes_f1 = float(np.max(f1_curve)) if f1_curve.size > 0 else 0.0

        print(f"\nBayes-rule ceiling ({n_unique} unique feature vectors over {n_total} rows):")
        print(f"  Mean rows per unique feature vector: {n_total / n_unique:.2f}")
        print(f"  Population SELECT rate:              {pos_rate_global:.2%}")
        print(f"  Accuracy ceiling:                    {bayes_acc:.2%}")
        print(f"  Balanced accuracy ceiling:           {bayes_bal_acc:.2%}")
        print(f"  F1 ceiling (top-K rule):             {bayes_f1:.2%}")

        return {
            'bayes_accuracy': bayes_acc,
            'bayes_balanced_accuracy': bayes_bal_acc,
            'bayes_f1': bayes_f1,
            'population_pos_rate': pos_rate_global,
            'n_unique_feature_vectors': n_unique,
            'n_total_rows': n_total,
        }


if __name__ == "__main__":
    max_depth = 5
    distiller = Extracted_policy(q_table_file_path="data/agent_data/Q_Learning/q_values.pkl.gz")
    distiller.prepare_dataset()
    distiller.bayes_optimal_diagnostic()
    distiller.cross_validate(k=10, max_depth=max_depth, min_samples_leaf=50)
    distiller.train(max_depth=max_depth, min_samples_leaf=50)
    print(distiller.get_feature_importance())
    distiller.plot(save_path="data/experiments/distilled_policy/distilled_policy.png")
