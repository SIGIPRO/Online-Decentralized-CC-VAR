import numpy as np
from src.core import BaseAgent
from src.cc_utils import CCIMPartialData, CellularComplexFakeClustering
from src.implementations.protocols import KStepProtocol
from src.implementations.mixing import KGTMixingModel
from src.implementations.models import CCVARModel
import scipy.io as sio # type: ignore[import-untyped]
from pathlib import Path




if __name__ == "__main__":
        # 1. Setup Paths
    dataset_name = "noaa_coastwatch_cellular"
    current_dir = Path.cwd() 
    root_name = (current_dir / ".." / ".." / "data" / "Input").resolve()
    output_dir = (current_dir / ".." / ".." / "data" / "Output" / dataset_name / "Figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Data
    try:
        m = sio.loadmat(root_name / dataset_name / "data_oriented_mov.mat")
        topology = sio.loadmat(root_name / dataset_name / "adjacencies_oriented.mat")
    except FileNotFoundError:
        print("Error: Data files not found. Check paths.")
        exit()

    signal_node = m['l'].T.astype(float)
    signal_edge = m['s'].T.astype(float)
    signal_poly = m['u'].T.astype(float)

    # Center Data (Essential for NMSE to match)
    signal_node -= np.mean(signal_node)
    signal_edge -= np.mean(signal_edge)
    signal_poly -= np.mean(signal_poly)

    cc_data = { 
        0 : signal_node,
        1 : signal_edge,
        2 : signal_poly

    }

    T_total = signal_edge.shape[1]

    # 4. Setup Complex & Parameters
    cellularComplex = {
        1: topology['B1'].astype(float),
        2: topology['B2'].astype(float)
    }
    clusteringParameters = {
        'd' : 10,
        'dim' : 0,
        'Q-hop' : 5
    }

    clusters = CellularComplexFakeClustering(cellularComplex=cellularComplex, clusteringParameters=clusteringParameters)
    
    mixing_params = (
        {"tracking": {"self": 0.0}, "correction": 0.0},  # initial_aux_vars
        {"self": 1.0, "cluster_1": 0.5},                 # weights
        {"K": 1.0, "c": 0.01, "s": 1.0},                 # eta hyperparameters
    )

    ## TODO: Complete ccvar_params
    algorithmParam = {
        'Tstep': 6,
        'P': 2,
        'K': [2, (2, 2), 2],     
        'mu': [0, (0, 0), 0],    
        'lambda': 0.01,
        'gamma': 0.98,
        'enabler': [True, True, True], 
        'FeatureNormalzn': True,
        'BiasEn': True
    }
    ccvar_params = (algorithmParam, cellularComplex)
    T = None
    agent_list = []
    for cluster_head in clusters.clustered_complexes:
        processed_data = dict()
        global_idx = clusters.global_to_local_idx[cluster_head]
        for dim in cc_data:
            processed_data[dim] = cc_data[dim][global_idx[dim],:]
        
        interface = dict()
        for head_tuple in clusters.interface:
            try: idx_head = head_tuple.index(cluster_head)
            except: continue
            idx_neighbor = 1 - idx_head
            interface[head_tuple[idx_neighbor]] = clusters.interface[head_tuple]

        Nout = dict()
        Nex = dict()

        for head_tuple in clusters.Nout:
            if cluster_head not in head_tuple: continue

            if cluster_head == head_tuple[0]:
                Nout[head_tuple[1]] = clusters.Nout[head_tuple]
                
            else:
                Nex[head_tuple[0]] = clusters.Nout[head_tuple]

        dataParams = (processed_data, 
                      interface,
                      Nout,
                      Nex,
                      global_idx)
        ## TODO: Check BaseProtocol and KStepProtocol
        currAgent = BaseAgent(
            model=CCVARModel,
            modelParams=(ccvar_params,),
            data=CCIMPartialData,
            dataParams=dataParams,
            protocol=KStepProtocol,
            protocolParams=(1, 5),  # send data every step, parameters every 5 steps
            mix=KGTMixingModel,
            mixingParams=mixing_params,
        )
        agent_list.append(currAgent)

        if T is None:
            T = currAgent._data._T_total
        else:
            if currAgent._data._T_total < T:
                T = currAgent._data._T_total


    for t in range(0,T):
        for agent in agent_list:
            agent.update()

            ## TODO: Handle communication




    # print(clusters.clustered_complexes)
    


# Local time-series per cluster: {name: np.ndarray of shape (features, time)}
# local_data = {"cluster_0": np.random.randn(4, 20)}

# # # Communication neighborhood (outgoing and incoming clusters)
# # Nout = {"cluster_1": {}}
# # Nin = {"cluster_1": {}}

# # Forwarded directly to ccvar.CCVAR(*ccvar_params)
# ccvar_params = (...)

# mixing_params = (
#     {"tracking": {"self": 0.0}, "correction": 0.0},  # initial_aux_vars
#     {"self": 1.0, "cluster_1": 0.5},                 # weights
#     {"K": 1.0, "c": 0.01, "s": 1.0},                 # eta hyperparameters
# )

# agent = BaseAgent(
#     model=CCVARModel,
#     modelParams=(ccvar_params,),
#     Nin=Nin,
#     Nout=Nout,
#     data=CellularComplexInMemoryData,
#     dataParams=(local_data,),
#     protocol=KStepProtocol,
#     protocolParams=(1, 5),  # send data every step, parameters every 5 steps
#     mix=KGTMixingModel,
#     mixingParams=mixing_params,
# )

# for t in range(10):
#     if not agent.update(t=t):
#         break

# forecast = agent.estimate(steps=1)
# print(forecast)