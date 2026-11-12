import numpy as np


class _No_learning():

    def learn(self, St, a, r, St_plus_1, done, t, T=None):
        pass

    def plan(self, planning_time, St, a, r, St_plus_1):
        pass

    def memorize_knowledge(self):
        pass

    def reduce_exploration_rate(self):
        pass


class Round_robin_agent(_No_learning):
    """
    Players in the fixed cyclic order 0, 1, ..., n-1, starting from player 0 in every episode ("Round-robin" in the article).
    """

    def __init__(self, n_players):
        self.n_players = n_players
        self.next_player = 0
        self.epsilon = 0

    def choose_action(self, state, valid_actions, state_unchanged_count=0):
        player = self.next_player
        self.next_player = (player + 1) % self.n_players
        assert player in valid_actions, "round-robin reached a checked player"
        return player


class Lowest_payoff_agent(_No_learning):
    """
    The unchecked player with the lowest current payoff, ties to the lowest index ("Lowest payoff first" in the article).
    """

    def __init__(self, brd_instance):
        self.BRD = brd_instance
        self.epsilon = 0

    def choose_action(self, state, valid_actions, state_unchanged_count=0):
        players = self.BRD.players
        return min(valid_actions, key=lambda i: (players[i]['Utility'], i))


class Masked_argmax_agent(_No_learning):
    """
    Deterministic reading of a learned Q-table ("Masked argmax" in the article): the best rank among those whose
    best-response flag is set. Ties, and observations not in the table, go to the highest rank. Returns a rank.
    """

    def __init__(self, n_players, q_table):
        self.n_players = n_players
        self.q_table = q_table
        self.epsilon = 0

    def choose_action(self, state, valid_actions, state_unchanged_count=0):
        ranks = np.flatnonzero(np.asarray(state[:self.n_players]) == 1)  # the first n entries are the b flags, by rank
        row = self.q_table.get(tuple(state))
        if row is None:
            return int(ranks.max())
        values = np.asarray(row)[ranks]
        return int(ranks[values == values.max()].max())


class Top_of_order_agent(_No_learning):
    """
    The player at the top of the canonical order (rank n), which always has its best-response flag set
    ("Top of the order" in the article, Lemma 3(ii)). Returns a rank.
    """

    def __init__(self, n_players):
        self.n_players = n_players
        self.epsilon = 0

    def choose_action(self, state, valid_actions, state_unchanged_count=0):
        assert state[self.n_players - 1] == 1, "the top rank has a cleared flag"
        return self.n_players - 1
