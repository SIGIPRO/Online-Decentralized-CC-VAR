import numpy as np
from src.core import BaseAgent
import scipy.io as sio # type: ignore[import-untyped]
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from cellexp_util.metric.metric_utils import MetricManager, general_metric
from cellexp_util.registry.metric_registry import ensure_metrics_registered
from tqdm import tqdm # type: ignore[import-untyped]


def append_same_dim(prediction, groundTruth, dim):
    gt = []; y = []
    for cluster_head in groundTruth:
        curr_gt = groundTruth[cluster_head][dim]
        curr_y = prediction[cluster_head][dim]

        y.append(curr_y)
        gt.append(curr_gt)

    try: y = np.vstack(y).flatten()
    except: y = np.hstack(y).flatten()

    try: gt = np.vstack(gt).flatten()
    except: gt = np.hstack(gt).flatten()

    return y, gt

@general_metric(name="tvNMSE", output="scalar")
def tvNMSE_distributed_metric(*, prediction, groundTruth, dim, **_):
    y, gt = append_same_dim(prediction, groundTruth, dim)
    return ((y - gt) ** 2).mean() / (gt ** 2).mean()

@general_metric(name="tvMAE", output="scalar")
def tvMAE_distributed_metric(*, prediction, groundTruth, dim, **_):
    y, gt = append_same_dim(prediction, groundTruth, dim)
    return np.abs(y - gt).mean()

@general_metric(name="tvMAPE", output="scalar")
def tvMAPE_distributed_metric(*, prediction, groundTruth, dim, **_):
    y, gt = append_same_dim(prediction, groundTruth, dim)
    m = gt != 0
    return (np.abs((y[m] - gt[m]) / gt[m]).mean() * 100) if m.any() else np.nan

# stateful rolling NMSE – just use manager from kwargs
@general_metric(name="rollingNMSE", output="scalar")
def rollingNMSE_distributed_metric(*, prediction, groundTruth, manager, dim, **_):
    # y = groundTruth["s"]; yhat = prediction

    y, gt = append_same_dim(prediction, groundTruth, dim)
    
    manager._cumulative_energy += gt ** 2
    manager._nmse_n += (y - gt) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        nmse_n = np.where(manager._cumulative_energy != 0,
                          manager._nmse_n / manager._cumulative_energy, 0.0)
    return float(nmse_n.mean())

# stateful rolling MAE – just use manager from kwargs
@general_metric(name="rollingMAE", output="scalar")
def rollingMAE_distributed_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors['tvMAEsingle'][:i + 1]))
    except Exception:
        return np.nan

# stateful rolling NMSE – just use manager from kwargs
@general_metric(name="rollingMAPE", output="scalar")
def rollingMAPE_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors['tvMAPEsingle'][:i + 1]))
    except Exception:
        return np.nan

## TODO: Implement CCVARPartial

""" NOTE: 
1. The divergence can be arising from not normalizing the features as in the CC-VAR paper. This is done already. Did not work.
 a. More severe error in implementation it seems. 
i. Taking the c very small, like 5e-6, solved the divergence issue but one should look at the implementation also. Partial solution +-.

2. KGTMixing is also causing divergence. Hyperparameters should be optimized.
"""

""" ISSUES:  
1. For some reason, CC-VAR explodes even with local steps. This issue is partially solved +-
2. For each agent edge signals are full of data which should not be the case. This issue solved. ++
3. CC-VAR is wrongly used. The problem is that it takes all of the elements of the agent which should not be the case. Not started --
"""

def load_data(datasetParams):

    dataset_name = datasetParams.dataset_name
    data_name = datasetParams.data_name
    adjacencies_name = datasetParams.adj_name
    current_dir = Path.cwd() 
    root_name = (current_dir / ".." / ".." / "data" / "Input").resolve()


    # 2. Load Data
    try:
        m = sio.loadmat(root_name / dataset_name / data_name)
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

    # 4. Setup Complex & Parameters
    cellularComplex = {
        1: topology['B1'].astype(float),
        2: topology['B2'].astype(float)
    }
    return cc_data, cellularComplex

def get_output_dir(dataset_name):
    current_dir = Path.cwd() 
    output_dir = lambda dim: (current_dir / ".." / ".." / "data" / "Output" / dataset_name / f"Results_{dim}").resolve()
    # output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


@hydra.main(version_base=None, config_path='../conf', config_name='config.yaml')
def main(cfg: DictConfig):

    outputDir = get_output_dir(cfg.dataset.dataset_name)
    ensure_metrics_registered()

    cc_data, cellularComplex = load_data(cfg.dataset)



    clusters = instantiate(config=cfg.clustering, cellularComplex=cellularComplex)

    T = None
    agent_list = dict()

    for cluster_head in clusters.clustered_complexes:

        processed_data = dict()
        global_idx = clusters.global_to_local_idx[cluster_head]


        for dim in cc_data:
            processed_data[dim] = cc_data[dim][global_idx[dim],:]
            
        # print(processed_data)
        interface = dict()
        for head_tuple in clusters.interface:
            try: idx_head = head_tuple.index(cluster_head)
            except: continue
            idx_neighbor = 1 - idx_head
            interface[head_tuple[idx_neighbor]] = clusters.interface[head_tuple]

        Nout = dict()
        Nex = dict()

        some_list = []


        for head_tuple in clusters.Nout:
            if cluster_head not in head_tuple: continue

            if cluster_head == head_tuple[0]:
                Nout[head_tuple[1]] = clusters.Nout[head_tuple]
 
                
            elif cluster_head == head_tuple[1]:
                Nex[head_tuple[0]] = clusters.Nout[head_tuple]
        
                try:
                    for n in Nex[head_tuple[0]][dim]:

                        some_list.append(n in clusters.Nin[cluster_head][dim])
                except:
                    continue
            else: 
                continue
        
  
        protocol = instantiate(cfg.protocol)
        
        ccvarmodel = instantiate(cfg.model, cellularComplex = clusters.clustered_complexes[cluster_head]) ## Check the usage of clusters 
        ccdata = instantiate(cfg.ccdata,
                              data = processed_data,
                              interface = interface,
                              Nout = Nout,
                              Nex = Nex,
                              global_idx = global_idx)
        weights = dict()
        num_connected = len(clusters.agent_graph[cluster_head]) + 1
        weights['self'] = 1/num_connected
        for cluster_id in list(clusters.agent_graph[cluster_head]):
            weights[cluster_id] = 1/num_connected

        mixing = instantiate(cfg.mixing, weights = weights)
        imputer = instantiate(cfg.imputer)

        currAgent = BaseAgent(
            cluster_id = cluster_head,
            model = ccvarmodel,
            data = ccdata,
            protocol = protocol,
            mix = mixing,
            imputer = imputer,
            neighbors = clusters.agent_graph[cluster_head] #Check the usage of clusters
        )
        agent_list[cluster_head] = currAgent

        if T is None:
            T = currAgent._data._T_total
        else:
            if currAgent._data._T_total < T:
                T = currAgent._data._T_total
    

    metrics = dict()

    for dim in cc_data:
        dim_N = cc_data[dim].shape[0]
        curr_output = outputDir(dim).mkdir(parents = True, exist_ok = True)
        metrics[dim] = MetricManager(N = dim_N, T = T, savePath = curr_output)
    import pdb; pdb.set_trace()
    for t in tqdm(range(0,T)):

        for cluster_head in agent_list:
            agent_list[cluster_head].iterate_data(t)
            agent_list[cluster_head].send_data(t)
            data_box = agent_list[cluster_head].outbox['data']
            for cluster_id in data_box:
               agent_list[cluster_id].push_to_inbox(cluster_head, data_box[cluster_id], "data")

        for cluster_head in agent_list:
            agent_list[cluster_head].receive_data()
            agent_list[cluster_head].local_step()

        for cluster_head in agent_list:
            agent_list[cluster_head].prepare_params(t)
            agent_list[cluster_head].send_params(t)
            params_box = agent_list[cluster_head].outbox['params']
            for cluster_id in params_box:
                agent_list[cluster_id].push_to_inbox(cluster_head, params_box[cluster_id], "params")

            # print(params_box)
            # import pdb; pdb.set_trace()

        for cluster_head in agent_list:
            agent_list[cluster_head].receive_params()
            ## TODO: Implement metric calculation here.
            print(f"Forecasted Value: {agent_list[cluster_head].estimate(input_data = None)} ####")
            # agent_list[cluster_head].do_consensus()

if __name__ == "__main__":
    main()