# Distributed CC-VAR

Lightweight building blocks for simulating distributed conditional value-at-risk (CC-VAR) estimation over cellular-complex graphs. The library provides agents, protocols, mixing rules, and model wrappers that you can compose into custom experiments.

## Overview
- Core abstractions in `src/core/`: `BaseAgent`, `BaseModel`, `BaseMixingModel`, `BaseProtocol`, and a stub `BaseEnvironment`.
- Reference implementations in `src/implementations/`: K-step synchronous protocol, KGT-style mixing, a CCVAR model wrapper, and helper agents (`DataHolderAgent`, `ZeroPadderAgent`). `label_propagator.py` is currently a stub.
- Data utilities in `src/cc_utils/ccdata.py`: in-memory cellular-complex iterator, MATLAB loader (`from_matlab`), and zero-filled helpers.
- External dependency: the CCVAR wrapper expects `ccvar.CCVAR` to be importable; that package is not bundled here.

## Installation
Requires Python 3.8+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Make sure your environment also has the `ccvar` module (or add it to `PYTHONPATH`), since `src/implementations/models/ccvar.py` imports it directly.

## Quickstart (single-agent sketch)
Adapt the parameters to your data shapes, neighborhood, and CCVAR signature.

```python
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
```

## Project layout
- `src/core/`: base classes for agents, models, protocols, mixing, and environment.
- `src/implementations/`: concrete protocol/mixing/model/agent helpers (K-step, KGT, CCVAR wrapper, data holders).
- `src/cc_utils/ccdata.py`: cellular-complex data loaders/iterators (`from_matlab`, zero padding).
- `examples/`: stubs for dataset-driven demos (e.g., `examples/ocean_dataset_cellular.py`).
- `configs/`: placeholder for experiment configs.
- `pyproject.toml`: packaging metadata and dependency pins (`numpy`, `pyyaml`, `scipy`).

## Notes and gaps
- Environment/router logic is intentionally minimal; implement message routing for multi-agent simulations.
- Example scripts and the label propagator are placeholders; fill these in to match your topology and datasets.
- No automated tests are present yet—add coverage before extending algorithms.

## License
MIT License (metadata in `pyproject.toml`).
