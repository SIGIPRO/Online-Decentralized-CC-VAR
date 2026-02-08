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
import matplotlib.pyplot as plt
import pickle


def aggregate_to_global_vector(per_cluster_data, global_to_local_idx, dim, total_size):
    values = np.zeros(total_size, dtype=float)
    counts = np.zeros(total_size, dtype=float)

    for cluster_head, dim_data in per_cluster_data.items():
        if dim not in dim_data:
            continue
        local_values = np.asarray(dim_data[dim]).reshape(-1)
        global_idx = np.asarray(global_to_local_idx[cluster_head][dim], dtype=int)

        if local_values.size != global_idx.size:
            min_len = min(local_values.size, global_idx.size)
            local_values = local_values[:min_len]
            global_idx = global_idx[:min_len]

        values[global_idx] += local_values
        counts[global_idx] += 1.0

    aggregated = np.zeros(total_size, dtype=float)
    valid = counts > 0
    aggregated[valid] = values[valid] / counts[valid]
    return aggregated


def keep_only_metrics(mm, keep):
    keep = set(keep)
    mm._registry = {k: v for k, v in mm._registry.items() if k in keep}  # new dict, do not mutate global
    mm._errors = {}
    for k, meta in mm._registry.items():
        if meta["output"] == "scalar":
            mm._errors[k] = np.zeros(mm._T)
            if meta.get("single", True):
                mm._errors[k + "single"] = np.zeros(mm._T)
        else:
            mm._errors[k] = []
            if meta.get("single", True):
                mm._errors[k + "single"] = []
    mm.reset_rolling()


def save_metric_plots(metrics, output_dirs):
    metric_names = [
        "tvNMSE",
        "tvMAE",
        "tvMAPE",
        "rollingNMSE",
        "rollingMAE",
        "rollingMAPE",
    ]
    fig_handles = {}

    for dim in sorted(metrics.keys()):
        manager = metrics[dim]
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        axes = axes.flatten()

        for ax, metric_name in zip(axes, metric_names):
            metric_values = np.asarray(manager._errors.get(metric_name, []), dtype=float)
            ax.plot(metric_values, linewidth=1.5)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)

        axes[-2].set_xlabel("t")
        axes[-1].set_xlabel("t")
        fig.suptitle(f"Metric Curves (dim={dim})")
        fig.tight_layout()

        plot_path = output_dirs[dim] / f"metric_curves_dim_{dim}.pdf"
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        fig_handles[dim] = fig

    if fig_handles:
        merged_handle_path = next(iter(output_dirs.values())).parent / "metric_figure_handles.pkl"
        with open(merged_handle_path, "wb") as file:
            pickle.dump(fig_handles, file, protocol=pickle.HIGHEST_PROTOCOL)

    for dim, fig in fig_handles.items():
        per_dim_handle_path = output_dirs[dim] / f"metric_curves_dim_{dim}.pkl"
        with open(per_dim_handle_path, "wb") as file:
            pickle.dump(fig, file, protocol=pickle.HIGHEST_PROTOCOL)
        plt.close(fig)


def append_same_dim(prediction, groundTruth, dim=None):
    if dim is None:
        y = np.asarray(prediction).reshape(-1)
        if isinstance(groundTruth, dict):
            gt = np.asarray(groundTruth.get("s", np.array([]))).reshape(-1)
        else:
            gt = np.asarray(groundTruth).reshape(-1)
        return y, gt

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
def tvNMSE_distributed_metric(*, prediction, groundTruth, dim=None, **_):
    y, gt = append_same_dim(prediction, groundTruth, dim)
    return ((y - gt) ** 2).mean() / (gt ** 2).mean()

@general_metric(name="tvMAE", output="scalar")
def tvMAE_distributed_metric(*, prediction, groundTruth, dim=None, **_):
    y, gt = append_same_dim(prediction, groundTruth, dim)
    return np.abs(y - gt).mean()

@general_metric(name="tvMAPE", output="scalar")
def tvMAPE_distributed_metric(*, prediction, groundTruth, dim=None, **_):
    y, gt = append_same_dim(prediction, groundTruth, dim)
    m = gt != 0
    return (np.abs((y[m] - gt[m]) / gt[m]).mean() * 100) if m.any() else np.nan

# stateful rolling NMSE – just use manager from kwargs
@general_metric(name="rollingNMSE", output="scalar")
def rollingNMSE_distributed_metric(*, prediction, groundTruth, manager, dim=None, **_):
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

""" NOTE: 
1. The divergence can be arising from not normalizing the features as in the CC-VAR paper. This is done already. Did not work.
 a. More severe error in implementation it seems. 
i. Taking the c very small, like 5e-6, solved the divergence issue but one should look at the implementation also. Partial solution +-.

2. KGTMixing is also causing divergence. Hyperparameters should be optimized.
"""

""" ISSUES:  
1. For some reason, CC-VAR explodes even with local steps. This issue is partially solved +- (The reason maybe really the learning parameter. In the previous case it was automatically updating itself. Moving to divergence of KGTMixing.)
2. For each agent edge signals are full of data which should not be the case. This issue solved. ++
3. CC-VAR is wrongly used. The problem is that it takes all of the elements of the agent which should not be the case. Implemented. Look at get_gradient method of the CCVARPartial to make it complete. Also add CCVARPartialModel for completeness. ++
4. LabelPropagator was also implemented. ++
5. Metric manager is not implemented yet. ++ (Codex implementation)
6. Look at the dynamic regret alongside of MSE. --
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
            neighbors = clusters.agent_graph[cluster_head],
            cellularComplex = clusters.clustered_complexes[cluster_head]
        )
        agent_list[cluster_head] = currAgent

        if T is None:
            T = currAgent._data._T_total
        else:
            if currAgent._data._T_total < T:
                T = currAgent._data._T_total
    

    metrics = dict()
    output_dirs = dict()
    T_eval = T - 1
    if T_eval <= 0:
        raise ValueError("Need at least 2 time steps for one-step-ahead metric evaluation.")

    for dim in cc_data:
        dim_N = cc_data[dim].shape[0]
        curr_output = outputDir(dim)
        curr_output.mkdir(parents = True, exist_ok = True)
        output_dirs[dim] = curr_output
        metrics[dim] = MetricManager(N = dim_N, T = T_eval, savePath = str(curr_output))
        keep_only_metrics(mm = metrics[dim], 
                          keep = ["tvNMSE", "tvMAE", "tvMAPE", "rollingNMSE", "rollingMAE", "rollingMAPE"])

    pending_prediction_by_cluster = None
    progress_bar = tqdm(range(0, T))
    for t in progress_bar:

        for cluster_head in agent_list:
            agent_list[cluster_head].iterate_data(t)
            agent_list[cluster_head].send_data(t)
            data_box = agent_list[cluster_head].outbox['data']
            for cluster_id in data_box:
               agent_list[cluster_id].push_to_inbox(cluster_head, data_box[cluster_id], "data")

        for cluster_head in agent_list:
            agent_list[cluster_head].receive_data()

        # Evaluate one-step-ahead predictions generated at previous step against current data x_t.
        if pending_prediction_by_cluster is not None:
            ground_truth_by_cluster = {
                cluster_head: agent_list[cluster_head]._data.get_data()
                for cluster_head in agent_list
            }
            postfix = {}
            eval_i = t - 1
            for dim in sorted(cc_data.keys()):
                dim_size = cc_data[dim].shape[0]
                pred_vec = aggregate_to_global_vector(
                    per_cluster_data=pending_prediction_by_cluster,
                    global_to_local_idx=clusters.global_to_local_idx,
                    dim=dim,
                    total_size=dim_size,
                )
                gt_vec = aggregate_to_global_vector(
                    per_cluster_data=ground_truth_by_cluster,
                    global_to_local_idx=clusters.global_to_local_idx,
                    dim=dim,
                    total_size=dim_size,
                )
                metrics[dim].step_calculation(
                    i=eval_i,
                    prediction=pred_vec,
                    groundTruth={"s": gt_vec},
                    verbose=False,
                )
                postfix[f"NMSE{dim}"] = f"{metrics[dim]._errors['tvNMSE'][eval_i]:.3e}"
            progress_bar.set_postfix(postfix)

        for cluster_head in agent_list:
            agent_list[cluster_head].local_step()

        for cluster_head in agent_list:
            agent_list[cluster_head].prepare_params(t)
            agent_list[cluster_head].send_params(t)
            params_box = agent_list[cluster_head].outbox['params']
            for cluster_id in params_box:
                agent_list[cluster_id].push_to_inbox(cluster_head, params_box[cluster_id], "params")

            # print(params_box)
            # import pdb; pdb.set_trace()

        prediction_by_cluster = dict()
        for cluster_head in agent_list:
            agent_list[cluster_head].receive_params()
            prediction_by_cluster[cluster_head] = agent_list[cluster_head].estimate(input_data=None, steps=1)
            # agent_list[cluster_head].do_consensus()
        pending_prediction_by_cluster = prediction_by_cluster

    for dim in cc_data:
        metrics[dim].save_single(n=0)
        metrics[dim].save_full(n=1)
    save_metric_plots(metrics=metrics, output_dirs=output_dirs)

if __name__ == "__main__":
    main()
