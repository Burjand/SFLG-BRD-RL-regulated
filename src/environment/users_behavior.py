import numpy as np

from config_manager import global_config


class Users_behavior():

    def __init__(self, FLG_graph, n_players):

        self.n_users = FLG_graph.n_nodes
        self.distances = FLG_graph.distances
        self.D = FLG_graph.D
        self.n_players = n_players
        self.min_similarity = global_config.get('min_similarity')

        self._compute_similarities()


    def _compute_similarities(self):

        """ Calculates the similarities of all the users in the game for all possible facilities """
        alpha = -np.log(self.min_similarity) / self.D # beta in the article. Furthest node has a controllable similarity

        self.similarities_table = np.empty((self.n_users, self.n_users), dtype=np.float32)

        for user in range(self.n_users):
            dist = np.array([self.distances[user][i] for i in range(self.n_users)])
            self.similarities_table[user, :] = np.exp(-alpha * dist)

        assert np.allclose(self.similarities_table.diagonal(), 1, atol=1e-6) # Make sure every user considers a facility on its own node the best


    def compute_attractions(self, players):

        """ Calculate all attractions (Shapley shares mu of every user to every player) """
        # Extract positions into a simple array ONCE
        strategy_profile = [int(players[i]['facility_position']) for i in range(self.n_players)]

        # Pre-allocate the result matrix (every cell is overwritten by put_along_axis below)
        mu_table = np.empty((self.n_users, self.n_players), dtype=np.float32)

        # Pre-fetch the whole similarity block we need (n_users, n_players)
        relevant_similarities = self.similarities_table[:, strategy_profile]

        # Vectorize the sorting
        sorted_indices = np.argsort(relevant_similarities, axis=1)
        sorted_sims = np.take_along_axis(relevant_similarities, sorted_indices, axis=1)

        # Calculate Diffs: sigma_j - sigma_{j-1} (prepend a virtual sigma_0 = 0)
        diffs = np.diff(sorted_sims, axis=1, prepend=0)

        # Clamp negatives (floating point errors)
        diffs = np.maximum(diffs, 0.0)

        # Denominators: [N, N-1, ..., 1]
        denoms = np.arange(self.n_players, 0, -1, dtype=np.float32)

        # Increments
        incs = diffs / denoms[None, :] # Broadcast division

        # Prefix Sums (The Mu values in sorted order)
        sorted_mus = np.cumsum(incs, axis=1)

        # UNSORT (Map back to original players)
        np.put_along_axis(mu_table, sorted_indices, sorted_mus, axis=1)

        return mu_table
