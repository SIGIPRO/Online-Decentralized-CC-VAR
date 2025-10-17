from .model import BaseModel

""" 

Base policy classes for variable and parameter sharing

"""

class VariablePolicy:
    
    def __init__(self):
        pass

class ParameterPolicy:
    
    def __init__(self):
        pass

"""

Specialized policy classes for variable and parameter sharing

"""


class PartialVariablePolicy(VariablePolicy):
    
    def __init__(self):
        pass

class NoParameterPolicy(ParameterPolicy):
    
    def __init__(self):
        pass

"""

DistributedAgent:

Agent class to hold neighborhood reports, exchange information between agents, and apply distributed optimization.

If one thinks about the control/RL framework, the followings are states and actions, and rewards:

STATES: Neighborhood reports
ACTIONS: Aggregation of local information, apply gradients, send information to others
REWARD: Reward is the error achieved by the Model


"""
class DistributedAgent:
    
    def __init__(self, modelParams):
        self._model = BaseModel(*modelParams)


    def update(self):
        # TODO Add some code here

        self._model.update_gradient()

    

