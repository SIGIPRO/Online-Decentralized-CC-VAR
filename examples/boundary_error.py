from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm  # type: ignore[import-untyped]

from cellexp_util.registry.metric_registry import ensure_metrics_registered

from examples.utils.clustering_utils import create_cluster_agents
from examples.utils.data_utils import load_data
from examples.utils.metric_utils import init_metric_managers

ERROR_METRICS = [
    "tvNMSE",
    "tvMAE",
    "tvMAPE",
    "rollingNMSE",
    "rollingMAE",
    "rollingMAPE",
]


def _get_forecast_horizon(cfg):
    model_algorithm = cfg.model.get("algorithmParam", {})
    return max(1, int(model_algorithm.get("Tstep", 1)))


def _extract_horizon_vector(values, forecast_horizon):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2:
        horizon_idx = min(forecast_horizon - 1, arr.shape[1] - 1)
        return arr[:, horizon_idx].reshape(-1)
    flat = arr.reshape(arr.shape[0], -1)
    if flat.shape[1] >= forecast_horizon:
        return flat[:, forecast_horizon - 1].reshape(-1)
    return flat[:, -1].reshape(-1)


def _shift_algorithm_buffer_with_prediction(algorithm, prediction):
    for key in sorted(prediction.keys()):
        if key not in algorithm._data:
            continue
        old_data = algorithm._data[key][:, 1:]
        if old_data.size == 0:
            continue

        new_col = np.asarray(prediction[key], dtype=float).reshape(-1, 1)
        if new_col.shape[0] != old_data.shape[0]:
            in_idx = getattr(algorithm, "_in_idx", {}).get(key, [])
            if in_idx:
                in_idx = np.asarray(in_idx, dtype=int)
                valid_idx = in_idx[in_idx < new_col.shape[0]]
                if valid_idx.size == old_data.shape[0]:
                    new_col = new_col[valid_idx, :]

        if new_col.shape[0] != old_data.shape[0]:
            if new_col.shape[0] > old_data.shape[0]:
                new_col = new_col[: old_data.shape[0], :]
            else:
                padded = np.zeros((old_data.shape[0], 1), dtype=float)
                padded[: new_col.shape[0], 0] = new_col[:, 0]
                new_col = padded

        algorithm._data[key] = np.hstack([old_data, new_col])


def _recursive_rollout_forecast(forecaster, forecast_horizon):
    rollout_obj = deepcopy(forecaster)
    prediction = None
    for step in range(forecast_horizon):
        prediction = rollout_obj.estimate(input_data=None, steps=1)
        if step < forecast_horizon - 1:
            algorithm = rollout_obj._algorithm if hasattr(rollout_obj, "_algorithm") else rollout_obj
            _shift_algorithm_buffer_with_prediction(algorithm=algorithm, prediction=prediction)
    return prediction


def _forecast_model_with_horizon(model, forecast_horizon):
    if forecast_horizon == 1:
        return model.estimate(input_data=None, steps=1)
    try:
        return model.estimate(input_data=None, steps=forecast_horizon)
    except Exception:
        return _recursive_rollout_forecast(
            forecaster=model,
            forecast_horizon=forecast_horizon,
        )


def _select_forecast_horizon(prediction_by_cluster, forecast_horizon):
    selected_prediction = {}
    for cluster_head in sorted(prediction_by_cluster.keys()):
        selected_prediction[cluster_head] = {}
        for dim in sorted(prediction_by_cluster[cluster_head].keys()):
            selected_prediction[cluster_head][dim] = _extract_horizon_vector(
                values=prediction_by_cluster[cluster_head][dim],
                forecast_horizon=forecast_horizon,
            )
    return selected_prediction


def _build_interface_idx_by_dim(clusters, cc_data):
    interface_idx_by_dim = {dim: set() for dim in cc_data}
    for _, dim_map in clusters.interface.items():
        for dim, idx_list in dim_map.items():
            if dim in interface_idx_by_dim:
                interface_idx_by_dim[dim].update(int(idx) for idx in idx_list)
    return {
        dim: np.asarray(sorted(list(idx_set)), dtype=int)
        for dim, idx_set in interface_idx_by_dim.items()
    }


def _resolve_agent_global_map(
    cluster_head,
    dim,
    pred_values,
    global_to_local_idx,
    cluster_out_global_idx,
):
    pred_values = np.asarray(pred_values, dtype=float).reshape(-1)
    all_global_idx = np.asarray(global_to_local_idx.get(cluster_head, {}).get(dim, []), dtype=int)
    out_global_idx = np.asarray(cluster_out_global_idx.get(cluster_head, {}).get(dim, []), dtype=int)

    if pred_values.size == all_global_idx.size:
        return all_global_idx, pred_values
    if pred_values.size == out_global_idx.size:
        return out_global_idx, pred_values

    candidates = []
    if all_global_idx.size > 0:
        candidates.append(all_global_idx)
    if out_global_idx.size > 0:
        candidates.append(out_global_idx)
    if not candidates:
        return np.array([], dtype=int), np.array([], dtype=float)

    mapping = min(candidates, key=lambda idx: abs(idx.size - pred_values.size))
    min_len = min(mapping.size, pred_values.size)
    return mapping[:min_len], pred_values[:min_len]


def _aggregate_interface_predictions(
    prediction_by_cluster,
    cc_data,
    global_to_local_idx,
    cluster_out_global_idx,
    interface_idx_by_dim,
):
    pred_vec_by_dim = {}
    gt_vec_by_dim = {}
    mask_by_dim = {}

    for dim in sorted(cc_data.keys()):
        dim_size = cc_data[dim].shape[0]
        interface_idx = interface_idx_by_dim.get(dim, np.array([], dtype=int))
        interface_mask = np.zeros(dim_size, dtype=bool)
        valid_interface_idx = interface_idx[(interface_idx >= 0) & (interface_idx < dim_size)]
        interface_mask[valid_interface_idx] = True

        pred_sum = np.zeros(dim_size, dtype=float)
        pred_count = np.zeros(dim_size, dtype=float)

        for cluster_head, dim_prediction in prediction_by_cluster.items():
            if dim not in dim_prediction:
                continue
            global_idx, pred_values = _resolve_agent_global_map(
                cluster_head=cluster_head,
                dim=dim,
                pred_values=dim_prediction[dim],
                global_to_local_idx=global_to_local_idx,
                cluster_out_global_idx=cluster_out_global_idx,
            )
            if global_idx.size == 0 or pred_values.size == 0:
                continue

            in_interface = interface_mask[global_idx]
            if not np.any(in_interface):
                continue

            use_idx = global_idx[in_interface]
            use_pred = pred_values[in_interface]
            pred_sum[use_idx] += use_pred
            pred_count[use_idx] += 1.0

        pred_vec = np.zeros(dim_size, dtype=float)
        valid = pred_count > 0
        pred_vec[valid] = pred_sum[valid] / pred_count[valid]

        pred_vec_by_dim[dim] = pred_vec
        gt_vec_by_dim[dim] = np.zeros(dim_size, dtype=float)
        mask_by_dim[dim] = interface_mask

    return pred_vec_by_dim, gt_vec_by_dim, mask_by_dim


def _save_interface_summary(output_root, metrics, eval_len, interface_idx_by_dim):
    summary_path = output_root / "interface_error_summary.txt"
    lines = ["Interface-only error metrics (distributed case)", ""]
    for dim in sorted(metrics.keys()):
        lines.append(f"[dim={dim}]")
        interface_count = int(interface_idx_by_dim.get(dim, np.array([], dtype=int)).size)
        lines.append(f"interface_count: {interface_count}")
        mm = metrics[dim]
        for metric_name in ERROR_METRICS:
            values = np.asarray(mm._errors.get(metric_name, []), dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                last = np.nan
                tail_mean = np.nan
            else:
                last = values[-1]
                tail_len = max(1, int(np.ceil(eval_len * 0.1)))
                tail_mean = float(np.mean(values[-tail_len:]))
            lines.append(f"{metric_name}: last={last:.6e}, last10%={tail_mean:.6e}")
        lines.append("")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def _force_boundary_model_to_partial_in(cfg: DictConfig):
    cfg_local = deepcopy(cfg)
    if "model" not in cfg_local:
        raise ValueError("Config must contain a model section.")
    cfg_local.model._target_ = "src.implementations.models.ccvar.CCVARPartialInModel"
    return cfg_local


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    ensure_metrics_registered()
    cfg = _force_boundary_model_to_partial_in(cfg)
    print("Boundary model forced to: src.implementations.models.ccvar.CCVARPartialInModel")
    cc_data, cellular_complex = load_data(cfg.dataset)
    clusters = instantiate(config=cfg.clustering, cellularComplex=cellular_complex)

    agent_list, cluster_out_global_idx, T = create_cluster_agents(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters,
        force_in_equals_out=False,
        protocol_overrides={},
    )

    forecast_horizon = _get_forecast_horizon(cfg)
    if T <= forecast_horizon:
        raise ValueError(f"Need T ({T}) > Tstep ({forecast_horizon}).")

    output_root = (Path.cwd() / "outputs" / cfg.dataset.dataset_name / "boundary_error").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    metrics, _ = init_metric_managers(
        cc_data=cc_data,
        output_dir_fn=lambda dim: output_root / f"results_{dim}",
        T_eval=T - forecast_horizon,
    )

    interface_idx_by_dim = _build_interface_idx_by_dim(clusters=clusters, cc_data=cc_data)
    pending_prediction_by_eval_t = {}

    progress_bar = tqdm(range(0, T), desc="Boundary Error (Distributed)")
    for t in progress_bar:
        for cluster_head in agent_list:
            agent_list[cluster_head].iterate_data(t)
            agent_list[cluster_head].send_data(t)
            data_box = agent_list[cluster_head].outbox["data"]
            for cluster_id in data_box:
                agent_list[cluster_id].push_to_inbox(cluster_head, data_box[cluster_id], "data")

        for cluster_head in agent_list:
            agent_list[cluster_head].receive_data()

        pending_prediction_by_cluster = pending_prediction_by_eval_t.pop(t, None)
        if pending_prediction_by_cluster is not None:
            pred_vec_by_dim, gt_vec_by_dim, mask_by_dim = _aggregate_interface_predictions(
                prediction_by_cluster=pending_prediction_by_cluster,
                cc_data=cc_data,
                global_to_local_idx=clusters.global_to_local_idx,
                cluster_out_global_idx=cluster_out_global_idx,
                interface_idx_by_dim=interface_idx_by_dim,
            )
            eval_i = t - forecast_horizon
            postfix = {}
            for dim in sorted(cc_data.keys()):
                gt_vec_by_dim[dim][mask_by_dim[dim]] = cc_data[dim][mask_by_dim[dim], t]
                metrics[dim].step_calculation(
                    i=eval_i,
                    prediction=pred_vec_by_dim[dim],
                    groundTruth={"s": gt_vec_by_dim[dim], "mask": mask_by_dim[dim]},
                    verbose=False,
                )
                postfix[f"NMSE{dim}"] = f"{metrics[dim]._errors['tvNMSE'][eval_i]:.3e}"
            progress_bar.set_postfix(postfix)

        for cluster_head in agent_list:
            agent_list[cluster_head].local_step()

        for cluster_head in agent_list:
            agent_list[cluster_head].prepare_params(t)
            agent_list[cluster_head].send_params(t)
            params_box = agent_list[cluster_head].outbox["params"]
            for cluster_id in params_box:
                agent_list[cluster_id].push_to_inbox(cluster_head, params_box[cluster_id], "params")

        has_fresh_neighbor_params = {}
        for cluster_head in agent_list:
            has_fresh_neighbor_params[cluster_head] = agent_list[cluster_head].receive_params()

        for cluster_head in agent_list:
            if has_fresh_neighbor_params[cluster_head]:
                agent_list[cluster_head].do_consensus()

        raw_prediction_by_cluster = {}
        for cluster_head in agent_list:
            raw_prediction_by_cluster[cluster_head] = _forecast_model_with_horizon(
                model=agent_list[cluster_head]._model,
                forecast_horizon=forecast_horizon,
            )
        eval_t = t + forecast_horizon
        if eval_t < T:
            pending_prediction_by_eval_t[eval_t] = _select_forecast_horizon(
                prediction_by_cluster=raw_prediction_by_cluster,
                forecast_horizon=forecast_horizon,
            )

    for dim in metrics:
        metrics[dim].save_single(n=0)
        metrics[dim].save_full(n=1)
    _save_interface_summary(
        output_root=output_root,
        metrics=metrics,
        eval_len=T - forecast_horizon,
        interface_idx_by_dim=interface_idx_by_dim,
    )

    print(f"Boundary interface-only metrics saved under: {output_root}")


if __name__ == "__main__":
    main()
