from pathlib import Path

import numpy as np
import scipy.io as sio  # type: ignore[import-untyped]


def load_data(datasetParams):
    dataset_name = datasetParams.dataset_name
    data_name = datasetParams.data_name
    adjacencies_name = datasetParams.adj_name
    current_dir = Path.cwd()
    root_name = (current_dir / ".." / ".." / "data" / "Input").resolve()

    try:
        m = sio.loadmat(root_name / dataset_name / data_name)
        topology = sio.loadmat(root_name / dataset_name / adjacencies_name)
    except FileNotFoundError:
        print("Error: Data files not found. Check paths.")
        exit()

    try: signal_node = m["l"].T.astype(float)
    except: signal_node = None
    signal_edge = m["s"].T.astype(float)
    try: signal_poly = m["u"].T.astype(float)
    except: signal_poly = None
    cc_data = dict()
    # Center data so NMSE scale is comparable across runs.
    if signal_node is not None:
        signal_node -= np.mean(signal_node)
        cc_data[0] = signal_node
    signal_edge -= np.mean(signal_edge)
    cc_data[1] = signal_edge
    if signal_poly is not None:
        signal_poly -= np.mean(signal_poly)
        cc_data[2] = signal_poly
    


    cellularComplex = {
        1: topology["B1"].astype(float),
        2: topology["B2"].astype(float),
    }
    return cc_data, cellularComplex


def get_output_dir(dataset_name):
    current_dir = Path.cwd()
    output_dir = lambda dim: (current_dir / ".." / ".." / "data" / "Output" / dataset_name / f"Results_{dim}").resolve()
    return output_dir


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


def aggregate_partial_prediction_to_global(prediction_by_cluster, cluster_out_global_idx, dim_size, dim):
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


def build_ground_truth_vector_for_partial(cc_data, dim, t_curr, valid_mask):
    gt_vec = np.zeros(cc_data[dim].shape[0], dtype=float)
    gt_vec[valid_mask] = cc_data[dim][valid_mask, t_curr]
    return gt_vec
