import numpy as np

class First_improvement_agent():

    def __init__(self, n_players, brd_instance):
        """
        Initialize the First Improvement Agent. This agent chooses a player at random, evaluates if it will change position, if it does, it chooses the player, if not, picks another player at random.
        """
        self.n_players = n_players
        self.BRD = brd_instance
        self.epsilon = 0

    def choose_action(self, state, valid_actions, state_unchanged_count=0):

        candidates = list(valid_actions)
        k = len(candidates)

        while k > 0:

            i = np.random.randint(k)
            player_id = candidates[i]

            current_facility = self.BRD.players[player_id]['facility_position']
            current_utility  = self.BRD.players[player_id]['Utility']

            candidate_utilities = self.BRD.game_behavior.calculate_player_utility_for_all_candidates(
                self.BRD.players, player_id
            )
            candidate_utilities[current_facility] = -np.inf
            best_value = float(candidate_utilities.max())
            gain = max(best_value - current_utility, 0.0)

            if gain > 1e-9:
                return player_id

            candidates[i] = candidates[k-1]
            k -= 1

        # No candidate improves: fall back to a random choice
        return np.random.choice(candidates)


    def learn(self, St, a, r, St_plus_1, done, t, T=None):

        pass

    def plan(self, planning_time, St, a, r, St_plus_1):

        pass

    def memorize_knowledge(self):

        pass

    def reduce_exploration_rate(self):

        pass
