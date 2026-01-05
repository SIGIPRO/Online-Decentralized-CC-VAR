import numpy as np
from src.core import BaseAgent
from src.cc_utils import CCIMPartialData 
# CellularComplexFakeClustering
# from src.implementations.protocols import KStepProtocol
# from src.implementations.mixing import KGTMixingModel
# from src.implementations.models import CCVARModel
import scipy.io as sio # type: ignore[import-untyped]
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def load_data(datasetParams):
        # 1. Setup Paths
    # dataset_name = "noaa_coastwatch_cellular"
    dataset_name = datasetParams.dataset_name
    data_name = datasetParams.data_name
    adjacencies_name = datasetParams.adj_name
    current_dir = Path.cwd() 
    root_name = (current_dir / ".." / ".." / "data" / "Input").resolve()


    # 2. Load Data
    try:
        # m = sio.loadmat(root_name / dataset_name / "data_oriented_mov.mat")
        m = sio.loadmat(root_name / dataset_name / data_name)
        # topology = sio.loadmat(root_name / dataset_name / "adjacencies_oriented.mat")
        topology = sio.loadmat(root_name / dataset_name / adjacencies_name)
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

    # T_total = signal_edge.shape[1]

    # 4. Setup Complex & Parameters
    cellularComplex = {
        1: topology['B1'].astype(float),
        2: topology['B2'].astype(float)
    }
    return cc_data, cellularComplex

def get_output_dir(dataset_name):
    current_dir = Path.cwd() 
    output_dir = (current_dir / ".." / ".." / "data" / "Output" / dataset_name / "Figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


@hydra.main(version_base=None, config_path='../conf')
def main(cfg: DictConfig):
    outputDir = get_output_dir(cfg.dataset.dataset_name)
    cc_data, cellularComplex = load_data(cfg.dataset)

    # clusteringParameters = {
    #     'd' : 10,
    #     'dim' : 0,
    #     'Q-hop' : 5
    # }

    # clusters = CellularComplexFakeClustering(cellularComplex=cellularComplex, clusteringParameters=cfg.clustering)
    clusters = instantiate(config=cfg.clustering, cellularComplex=cellularComplex)
    
    # mixing_params = (
    #     {"tracking": {"self": 0.0}, "correction": 0.0},  # initial_aux_vars
    #     {"self": 1.0, "cluster_1": 0.5},                 # weights
    #     {"K": 1.0, "c": 0.01, "s": 1.0},                 # eta hyperparameters
    # )

    # algorithmParam = {
    #     'Tstep': 6,
    #     'P': 2,
    #     'K': [2, (2, 2), 2],     
    #     'mu': [0, (0, 0), 0],    
    #     'lambda': 0.01,
    #     'gamma': 0.98,
    #     'enabler': [True, True, True], 
    #     'FeatureNormalzn': True,
    #     'BiasEn': True
    # }
    # ccvar_params = (algorithmParam, cellularComplex)
    # ccvar_params = (cfg.model.algorithmParam, cellularComplex)
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
        # K_data = getattr(cfg.protocol, "K_data", 1)
        # K_param = getattr(cfg.protocol, "K_param", 1)
        protocol = instantiate(cfg.protocol)
        ccvarmodel = instantiate(cfg.model, cellularComplex = clusters.clustered_complexes[cluster_head]) ## Check the usage of clusters 
        ccdata = CCIMPartialData(*dataParams)
        mixing = instantiate(cfg.mixing)
        imputer = instantiate(cfg.imputer)
        ## TODO 1: Check if mixing is complying with the data, model, protocol. (Codex controlled it)
        ## TODO 2:  Check if imputer is complying with the data, model, protocol and mixing.
        ## TODO 3: Check if agent is complying with the data, model, protocol, mixing and imputer.
        ## TODO 4: Implement metric, results plotter and cellular complex plotter.
        currAgent = BaseAgent(
            cluster_id = cluster_head,
            model = ccvarmodel,
            data = ccdata,
            protocol = protocol,
            mix = mixing,
            imputer = imputer,
            neighbors = clusters.agent_graph[cluster_head] #Check the usage of clusters
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
