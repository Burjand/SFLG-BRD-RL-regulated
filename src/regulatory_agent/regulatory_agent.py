class Regulatory_Agent():

    def __init__(self, main_rng, n_players):
        
        self.main_rng = main_rng
        self.n_players = n_players


    def choose_action(self):

        action = self.main_rng.choice(tuple(range(self.n_brd_players))) # Randomly select a player to find their best response

        return action

