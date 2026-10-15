import numpy as np

from src.environment.game_behavior import Game_behavior


class BRD():

    def __init__(self, n_players, FLG_graph, seed=42, rationality_epsilon=0.05):

        self.flg_graph = FLG_graph
        self.n_nodes = self.flg_graph.n_nodes
        self.n_players = n_players
        self.seed = seed
        self.rationality_epsilon = rationality_epsilon
        self.rng = np.random.default_rng(seed=self.seed)

        self.game_behavior = Game_behavior(self.flg_graph, n_players)
        self.create_players()


    def create_players(self):

        """
        Creates the players for the game, with initialization of facility positions at random.

        Returns:
            players (dict): A dictionary containing as key the player id (int) and as items their facility position (int) and their utility (float).
        """

        # Check amount of players is less than the number of available facilities
        if self.n_players > self.n_nodes:
            raise ValueError("More players than available facilities.")

        # Potential facilities (nodes): 1 if a player was placed there, 0 if not
        self.facilities = {i: 0 for i in range(self.n_nodes)}

        # Create players and assign them to random, distinct facilities
        self.players = {}
        for i in range(self.n_players):
            selected_random_facility = int(self.rng.choice(tuple(k for k, v in self.facilities.items() if v == 0)))
            self.players[i] = {'facility_position': selected_random_facility, 'Utility': None}
            self.facilities[selected_random_facility] = 1  # Mark the facility as occupied by a player

        # Update the utilities of all players
        facilities_utilities = self.game_behavior.calculate_facilities_utilities(self.players)
        for i in range(self.n_players):
            self.players[i]['Utility'] = facilities_utilities[i]


    def compute_player_best_response(self, player_id):

        # Store the current best facility and utility for the player

        action_type = np.random.choice(a=["rational_action", "irrational_action"], p=[1-self.rationality_epsilon, self.rationality_epsilon])
        facility_options = [i for i in range(self.n_nodes)] # Consider all nodes (Allows sharing facilities)
        change_flag = False # A flag to know if the player changed position (AKA changed strategy)
        current_facility = self.players[player_id]['facility_position']

        if action_type == "rational_action":

            best_option = current_facility

            # Utility of player_id at every candidate node, the other players fixed (Algorithm 1 in the article)
            candidate_utilities = self.game_behavior.calculate_player_utility_for_all_candidates(self.players, player_id)
            # Current utility from the same function (not the stored one) so float32 rounding does not create fake gains
            best_utility = float(candidate_utilities[current_facility])
            candidate_utilities[current_facility] = -np.inf # Exclude current position

            best_idx = int(np.argmax(candidate_utilities))
            best_value = float(candidate_utilities[best_idx])

            # Move only on a gain above float32 noise.
            tol = 1e-5 * max(1.0, abs(best_utility))
            if best_value > best_utility + tol:
                best_option = best_idx
                best_utility = best_value
                change_flag = True

        else:

            best_option = np.random.choice(facility_options)
            if best_option != current_facility:
                change_flag = True

        # If it discovered a better option
        if change_flag:
            # Change position
            self.players[player_id]['facility_position'] = best_option

            # And update all players' utilities
            new_utilities = self.game_behavior.calculate_facilities_utilities(self.players)
            for i in range(self.n_players):
                self.players[i]['Utility'] = new_utilities[i]

            return True

        return False
