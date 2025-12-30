import numpy as np
from src.core import BaseAgent
from src.cc_utils import CellularComplexInMemoryData
from src.implementations.protocols import KStepProtocol
from src.implementations.mixing import KGTMixingModel
from src.implementations.models import CCVARModel

# Local time-series per cluster: {name: np.ndarray of shape (features, time)}
local_data = {"cluster_0": np.random.randn(4, 20)}

# Communication neighborhood (outgoing and incoming clusters)
Nout = {"cluster_1": {}}
Nin = {"cluster_1": {}}

# Forwarded directly to ccvar.CCVAR(*ccvar_params)
ccvar_params = (...)

mixing_params = (
    {"tracking": {"self": 0.0}, "correction": 0.0},  # initial_aux_vars
    {"self": 1.0, "cluster_1": 0.5},                 # weights
    {"K": 1.0, "c": 0.01, "s": 1.0},                 # eta hyperparameters
)

agent = BaseAgent(
    model=CCVARModel,
    modelParams=(ccvar_params,),
    Nin=Nin,
    Nout=Nout,
    data=CellularComplexInMemoryData,
    dataParams=(local_data,),
    protocol=KStepProtocol,
    protocolParams=(1, 5),  # send data every step, parameters every 5 steps
    mix=KGTMixingModel,
    mixingParams=mixing_params,
)

for t in range(10):
    if not agent.update(t=t):
        break

forecast = agent.estimate(steps=1)
print(forecast)