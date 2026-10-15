import numpy as np

from src.environment.users_behavior import Users_behavior

class Game_behavior():

    def __init__(self, flg_graph, n_players):

        self.ub = Users_behavior(flg_graph, n_players)
        self.nodes_demands = np.array(
            [flg_graph.nodes_demands_raw[i] for i in range(len(flg_graph.nodes_demands_raw))],
            dtype=np.float32
        ) # Demand in absolute values


    def calculate_facilities_utilities(self, players):

        mu_table = self.ub.compute_attractions(players)
        captured_demands = self.nodes_demands @ mu_table

        return captured_demands


    def calculate_single_player_utility(self, players, target_player_id):

        mu_table = self.ub.compute_attractions(players)
        target_player_shares = np.array(mu_table)[:, target_player_id]

        return np.sum(target_player_shares * self.nodes_demands)


    def calculate_player_utility_for_all_candidates(self, players, target_player_id):

        """
        Utility of `target_player_id` at every candidate node, with the other
        players fixed (Algorithm 1 in the article). Returns a length-N array.
        """
        M = self.ub.n_players
        N = self.ub.n_users

        # Profile of the fixed other players
        profile_others = [
            int(players[i]['facility_position'])
            for i in range(M)
            if i != target_player_id
        ]

        # Sort the others' similarities once per sweep (ascending)
        sigma_others = self.ub.similarities_table[:, profile_others]            # (N, M-1)
        sigma_others_sorted = np.sort(sigma_others, axis=1)                     # (N, M-1)

        # E_full[u, j]: Shapley share the deviator gets from the j least similar others
        #   E_full[u, j] = sum_{i=1..j} (sigma_sorted[i-1] - sigma_sorted[i-2]) / (M - i + 1),  sigma_sorted[-1] = 0
        if M > 1:
            diffs_others = np.diff(sigma_others_sorted, axis=1, prepend=0)      # (N, M-1)
            denoms_E = np.arange(M, 1, -1, dtype=np.float32)                    # (M-1,) = [M, M-1, ..., 2]
            contributions = diffs_others / denoms_E
            E_full = np.empty((N, M), dtype=np.float32)
            E_full[:, 0] = 0.0
            E_full[:, 1:] = np.cumsum(contributions, axis=1)
            # Previous similarity below rank k (0 for k = 1)
            sigma_padded = np.empty((N, M), dtype=np.float32)
            sigma_padded[:, 0] = 0.0
            sigma_padded[:, 1:] = sigma_others_sorted
        else:
            # Single-player game: deviator gets the entire similarity at every candidate.
            E_full = np.zeros((N, 1), dtype=np.float32)
            sigma_padded = np.zeros((N, 1), dtype=np.float32)

        # Deviator similarities for every candidate position
        s_all = self.ub.similarities_table                                      # (N, N)

        # Rank of each candidate among the others' similarities: k = 1 + count(sigma_others < s)
        if M > 1:
            rank_below = (sigma_others_sorted[:, :, None] < s_all[:, None, :]).sum(axis=1)  # (N, N)
        else:
            rank_below = np.zeros((N, N), dtype=np.int64)
        k_all = rank_below + 1                                                  # (N, N), in [1, M]

        # Fancy-index into the precomputed tables
        rows = np.arange(N)[:, None]                                            # (N, 1)
        E_at_k    = E_full[rows, k_all - 1]                                     # (N, N)
        prev_at_k = sigma_padded[rows, k_all - 1]                               # (N, N)
        denom_at_k = (M - k_all + 1).astype(np.float32)                         # (N, N)

        # Deviator's mu-share for each (user, candidate)
        mu_p = E_at_k + (s_all - prev_at_k) / denom_at_k                        # (N, N)
        mu_p = np.maximum(mu_p, 0.0)  # safety against float drift at ties

        # Utility per candidate
        return self.nodes_demands @ mu_p  # shape: (N_candidates,)


    def calculate_potential_function(self, players):

        """
        Calculates the exact Potential Function Phi(x) for the current profile.
        Formula: Sum_u ( Demand_u * [ Sum_intervals (Length * Harmonic_Sum(Count)) ] )
        """
        # Get similarities for current profile
        strategy_profile = [int(players[i]['facility_position']) for i in range(len(players))]
        profile_similarities = self.ub.similarities_table[:, strategy_profile]

        # Sort every row ascending to determine intervals
        sorted_sims = np.sort(profile_similarities, axis=1)

        # Calculate Interval Lengths
        diffs = np.diff(sorted_sims, axis=1, prepend=0)

        # Create Harmonic Weights
        m = len(players)
        k_values = np.arange(1, m + 1)
        harmonics = np.cumsum(1.0 / k_values)
        harmonic_weights = harmonics[::-1]

        # Calculate Potential Contribution per User
        user_potentials = np.sum(diffs * harmonic_weights, axis=1)

        # Weighted Sum by Node Demand
        total_potential = np.sum(user_potentials * self.nodes_demands)

        return total_potential


    def calculate_social_welfare(self, players):

        # Closed-form social welfare: Sum_u demand_u * max_p sigma(u, x_p).

        strategy_profile = [int(players[i]['facility_position']) for i in range(len(players))]
        profile_similarities = self.ub.similarities_table[:, strategy_profile]
        max_sim_per_user = np.max(profile_similarities, axis=1)
        return float(np.sum(max_sim_per_user * self.nodes_demands))
