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
from copy import deepcopy


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


def build_partial_indices(global_idx, nin_for_cluster):
    in_idx = {}
    out_idx = {}

    for dim, gidx_list in global_idx.items():
        local_all = list(range(len(gidx_list)))
        in_idx[dim] = local_all

        owned_global = set(nin_for_cluster.get(dim, []))
        local_owned = [i for i, g in enumerate(gidx_list) if g in owned_global]

        # Fallback: if a dimension has no owned cells after clustering/interface split,
        # keep local cells so model dimensions remain valid.
        out_idx[dim] = local_owned if local_owned else local_all

    return in_idx, out_idx


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

def _resolve_eval_arrays(prediction, groundTruth, dim=None):
    y, gt = append_same_dim(prediction, groundTruth, dim)
    mask = np.ones_like(gt, dtype=bool)
    if dim is None and isinstance(groundTruth, dict):
        incoming_mask = groundTruth.get("mask", None)
        if incoming_mask is not None:
            mask = np.asarray(incoming_mask, dtype=bool).reshape(-1)
            if mask.shape != gt.shape:
                mask = np.ones_like(gt, dtype=bool)
    return y, gt, mask

@general_metric(name="tvNMSE", output="scalar")
def tvNMSE_distributed_metric(*, prediction, groundTruth, dim=None, **_):
    y, gt, mask = _resolve_eval_arrays(prediction, groundTruth, dim)
    if not np.any(mask):
        return np.nan
    y = y[mask]
    gt = gt[mask]
    denom = (gt ** 2).mean()
    if denom == 0:
        return np.nan
    return ((y - gt) ** 2).mean() / denom

@general_metric(name="tvMAE", output="scalar")
def tvMAE_distributed_metric(*, prediction, groundTruth, dim=None, **_):
    y, gt, mask = _resolve_eval_arrays(prediction, groundTruth, dim)
    if not np.any(mask):
        return np.nan
    return np.abs(y[mask] - gt[mask]).mean()

@general_metric(name="tvMAPE", output="scalar")
def tvMAPE_distributed_metric(*, prediction, groundTruth, dim=None, **_):
    y, gt, mask = _resolve_eval_arrays(prediction, groundTruth, dim)
    m = mask & (gt != 0)
    return (np.abs((y[m] - gt[m]) / gt[m]).mean() * 100) if m.any() else np.nan

# stateful rolling NMSE – just use manager from kwargs
@general_metric(name="rollingNMSE", output="scalar")
def rollingNMSE_distributed_metric(*, prediction, groundTruth, manager, dim=None, **_):
    # y = groundTruth["s"]; yhat = prediction

    y, gt, mask = _resolve_eval_arrays(prediction, groundTruth, dim)
    if not np.any(mask):
        return np.nan
    
    cumulative_energy = np.zeros_like(gt, dtype=float)
    nmse_n = np.zeros_like(gt, dtype=float)
    cumulative_energy[mask] = gt[mask] ** 2
    nmse_n[mask] = (y[mask] - gt[mask]) ** 2
    manager._cumulative_energy += cumulative_energy
    manager._nmse_n += nmse_n

    valid = manager._cumulative_energy != 0
    if not np.any(valid):
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        nmse_n = np.where(valid, manager._nmse_n / manager._cumulative_energy, 0.0)
    return float(nmse_n[valid].mean())

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
    cluster_out_global_idx = dict()

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
        
        model_cfg = deepcopy(cfg.model)
        in_idx, out_idx = build_partial_indices(
            global_idx=global_idx,
            nin_for_cluster=clusters.Nin[cluster_head],
        )
        cluster_out_global_idx[cluster_head] = dict()
        for dim in global_idx:
            cluster_out_global_idx[cluster_head][dim] = np.asarray(
                [global_idx[dim][i] for i in out_idx[dim]],
                dtype=int,
            )
        model_cfg.algorithmParam.in_idx = in_idx
        model_cfg.algorithmParam.out_idx = out_idx
        ccvarmodel = instantiate(model_cfg, cellularComplex = clusters.clustered_complexes[cluster_head]) ## Check the usage of clusters 
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

    def aggregate_partial_prediction_to_global(prediction_by_cluster, dim):
        dim_size = cc_data[dim].shape[0]
        pred_sum = np.zeros(dim_size, dtype=float)
        pred_count = np.zeros(dim_size, dtype=float)

        for cluster_head, dim_prediction in prediction_by_cluster.items():
            if dim not in dim_prediction:
                continue

            pred_values = np.asarray(dim_prediction[dim]).reshape(-1)
            out_global_idx = np.asarray(
                cluster_out_global_idx.get(cluster_head, {}).get(dim, np.array([], dtype=int)),
                dtype=int,
            )
            if pred_values.size == 0 or out_global_idx.size == 0:
                continue
            if pred_values.size != out_global_idx.size:
                min_len = min(pred_values.size, out_global_idx.size)
                pred_values = pred_values[:min_len]
                out_global_idx = out_global_idx[:min_len]

            pred_sum[out_global_idx] += pred_values
            pred_count[out_global_idx] += 1.0

        pred_vec = np.zeros(dim_size, dtype=float)
        valid_mask = pred_count > 0
        pred_vec[valid_mask] = pred_sum[valid_mask] / pred_count[valid_mask]
        return pred_vec, valid_mask

    def build_ground_truth_vector_for_partial(dim, t_curr, valid_mask):
        gt_vec = np.zeros(cc_data[dim].shape[0], dtype=float)
        gt_vec[valid_mask] = cc_data[dim][valid_mask, t_curr]
        return gt_vec

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
                pred_vec, valid_mask = aggregate_partial_prediction_to_global(
                    prediction_by_cluster=pending_prediction_by_cluster,
                    dim=dim,
                )
                gt_vec = build_ground_truth_vector_for_partial(
                    dim=dim,
                    t_curr=t,
                    valid_mask=valid_mask,
                )
                metrics[dim].step_calculation(
                    i=eval_i,
                    prediction=pred_vec,
                    groundTruth={"s": gt_vec, "mask": valid_mask},
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
