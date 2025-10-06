import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import src.environment.tools.general_tools as general_tools

from src.regulatory_agent.Q_learning_agent import Q_learning_agent
from regulatory_agent.drunk_agent import Drunk_agent

CONFIGURATION = general_tools.extract_json_data("config.json5")
agent_architecture_text = CONFIGURATION['agent_architecture']

if agent_architecture_text == "q_learning_agent":
    agent_architecture = Q_learning_agent

elif agent_architecture_text == "drunk_agent":
    agent_architecture = Drunk_agent

else:
    raise Exception("Agent architecture not recognized")

class Regulatory_Agent(agent_architecture):

    pass

