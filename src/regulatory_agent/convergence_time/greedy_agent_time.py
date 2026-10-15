"""
Greedy agent based on steepest ascent theorem
"""
import numpy as np

class Greedy_agent_time():

    def __init__(self, n_players, brd_instance):
        self.n_players = n_players
        self.BRD = brd_instance
        self.epsilon = 0

    def choose_action(self, state, valid_actions, state_unchanged_count=0):
        best_players = []
        max_potential_gain = -float('inf')

        # Evaluate each player's best-response gain without moving anyone
        for player_id in valid_actions:
            current_facility = self.BRD.players[player_id]['facility_position']

            candidate_utilities = self.BRD.game_behavior.calculate_player_utility_for_all_candidates(
                self.BRD.players, player_id
            )
            # Same-function baseline (not stored Utility) to avoid float32 phantom gains.
            current_utility = float(candidate_utilities[current_facility])
            candidate_utilities[current_facility] = -np.inf
            best_value = float(candidate_utilities.max())
            # Threshold above float32 noise; matches BRD's move tolerance.
            gain_threshold = 1e-5 * max(1.0, abs(current_utility))
            gain = best_value - current_utility
            if gain > gain_threshold:
                if gain > max_potential_gain:
                    max_potential_gain = gain
                    best_players = [player_id]
                elif gain == max_potential_gain:
                    best_players.append(player_id)

        # Return best player (random among ties)
        if not best_players:
            return int(np.random.choice(valid_actions))

        return int(np.random.choice(best_players))

    def learn(self, St, a, r, St_plus_1, done, t, T=None):

        pass

    def plan(self, planning_time, St, a, r, St_plus_1):

        pass

    def memorize_knowledge(self):

        pass

    def reduce_exploration_rate(self):

        pass
