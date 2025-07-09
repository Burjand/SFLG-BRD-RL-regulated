class Regulatory_Agent():

    def __init__(self, main_rng, n_players):
        
        self.main_rng = main_rng
        self.n_players = n_players


    def choose_action(self, St, valid_actions):

        action = self.main_rng.choice(valid_actions) # Randomly select a player to find their best response

        return action

