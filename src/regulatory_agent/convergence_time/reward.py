class Reward():

    def __init__(self, reward_function_name):

        self.set_reward_function(reward_function_name)


    def p_function_m_iterations(self, potential_function_value, iterations):

        return potential_function_value - iterations

    def p_function(self, potential_function_value_0, potential_function_value_1, iterations):

        return potential_function_value_1

    def p_function_diff_m_iterations(self, potential_function_value_0, potential_function_value_1, iterations):

        return (potential_function_value_1 - potential_function_value_0) - iterations

    def p_function_diff(self, potential_function_value_0, potential_function_value_1, iterations):

        return (potential_function_value_1 - potential_function_value_0)

    def iterations(self, potential_function_value_0, potential_function_value_1, iterations):

        return -1*iterations

    def set_reward_function(self, name):

        """Select reward function by name"""

        reward_functions = {
            'potential_function_minus_iterations': self.p_function_m_iterations,
            'absolute_potential_function': self.p_function,
            'potential_function_difference_minus_iterations': self.p_function_diff_m_iterations,
            'potential_function_difference': self.p_function_diff,
            'iterations': self.iterations,
        }
        if name not in reward_functions:
            raise KeyError(f"Unknown reward function '{name}'. Options: {list(reward_functions)}")
        self.reward_function = reward_functions[name]
