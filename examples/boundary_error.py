from copy import deepcopy
from pathlib import Path
import pickle

import hydra
import matplotlib
import numpy as np
from ccvar import CCVAR
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm  # type: ignore[import-untyped]

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cellexp_util.registry.metric_registry import ensure_metrics_registered

from examples.utils.clustering_utils import create_cluster_agents
from examples.utils.data_utils import load_data


def _get_forecast_horizon(cfg):
    model_algorithm = cfg.model.get("algorithmParam", {})
    return max(1, int(model_algorithm.get("Tstep", 1)))


def _get_ccvar_algorithm_param(cfg):
    algorithm_param = deepcopy(OmegaConf.to_container(cfg.model.get("algorithmParam", {}), resolve=True))
    algorithm_param.pop("in_idx", None)
    algorithm_param.pop("out_idx", None)

    # Global CCVAR expects scalar K/mu per dimension, while CCVARPartial config
    # may store edge entries as pairs like [K_lower, K_upper].
    for key in ("K", "mu"):
        values = algorithm_param.get(key, None)
        if isinstance(values, list):
            flattened = []
            for item in values:
                if isinstance(item, (list, tuple)):
                    flattened.append(item[0] if len(item) > 0 else 0)
                else:
                    flattened.append(item)
            algorithm_param[key] = flattened

    return algorithm_param


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


def _recursive_rollout_forecast_model(forecaster, forecast_horizon):
    rollout_obj = deepcopy(forecaster)
    prediction = None
    for step in range(forecast_horizon):
        prediction = rollout_obj.estimate(input_data=None, steps=1)
        if step < forecast_horizon - 1:
            algorithm = rollout_obj._algorithm if hasattr(rollout_obj, "_algorithm") else rollout_obj
            _shift_algorithm_buffer_with_prediction(algorithm=algorithm, prediction=prediction)
    return prediction


def _recursive_rollout_forecast_ccvar(forecaster, forecast_horizon):
    rollout_obj = deepcopy(forecaster)
    prediction = None
    for step in range(forecast_horizon):
        prediction = rollout_obj.forecast(steps=1)
        if step < forecast_horizon - 1:
            _shift_algorithm_buffer_with_prediction(algorithm=rollout_obj, prediction=prediction)
    return prediction


def _forecast_model_with_horizon(model, forecast_horizon):
    if forecast_horizon == 1:
        return model.estimate(input_data=None, steps=1)
    try:
        return model.estimate(input_data=None, steps=forecast_horizon)
    except Exception:
        return _recursive_rollout_forecast_model(forecaster=model, forecast_horizon=forecast_horizon)


def _forecast_ccvar_with_horizon(model, forecast_horizon):
    if forecast_horizon == 1:
        return model.forecast(steps=1)
    try:
        return model.forecast(steps=forecast_horizon)
    except Exception:
        return _recursive_rollout_forecast_ccvar(forecaster=model, forecast_horizon=forecast_horizon)


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


def _select_global_forecast_horizon(prediction_by_dim, forecast_horizon):
    return {
        dim: _extract_horizon_vector(values=prediction_by_dim.get(dim, np.array([])), forecast_horizon=forecast_horizon)
        for dim in sorted(prediction_by_dim.keys())
    }


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


def _build_interface_mask_by_dim(interface_idx_by_dim, cc_data):
    mask_by_dim = {}
    for dim in sorted(cc_data.keys()):
        dim_size = cc_data[dim].shape[0]
        mask = np.zeros(dim_size, dtype=bool)
        idx = interface_idx_by_dim.get(dim, np.array([], dtype=int))
        valid_idx = idx[(idx >= 0) & (idx < dim_size)]
        mask[valid_idx] = True
        mask_by_dim[dim] = mask
    return mask_by_dim


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
        mask_by_dim[dim] = interface_mask

    return pred_vec_by_dim, mask_by_dim


def _compute_interface_nmse(pred_vec_by_dim, mask_by_dim, cc_data, t_eval):
    sq_error_sum = 0.0
    energy_sum = 0.0
    nmse_by_dim = {}
    for dim in sorted(cc_data.keys()):
        pred = np.asarray(pred_vec_by_dim.get(dim, np.array([])), dtype=float).reshape(-1)
        if pred.size == 0:
            nmse_by_dim[dim] = np.nan
            continue
        gt = np.asarray(cc_data[dim][:, t_eval], dtype=float).reshape(-1)
        mask = np.asarray(mask_by_dim.get(dim, np.zeros_like(gt, dtype=bool)), dtype=bool).reshape(-1)
        valid_mask = mask[: min(mask.size, gt.size, pred.size)]
        if not np.any(valid_mask):
            nmse_by_dim[dim] = np.nan
            continue
        pred_use = pred[: valid_mask.size][valid_mask]
        gt_use = gt[: valid_mask.size][valid_mask]
        dim_sq_error = float(np.sum((pred_use - gt_use) ** 2))
        dim_energy = float(np.sum(gt_use ** 2))
        if dim_energy <= 0:
            nmse_by_dim[dim] = np.nan
            continue
        nmse_by_dim[dim] = dim_sq_error / dim_energy
        sq_error_sum += dim_sq_error
        energy_sum += dim_energy
    overall_nmse = np.nan if energy_sum <= 0 else (sq_error_sum / energy_sum)
    return overall_nmse, nmse_by_dim


def _force_boundary_model_to_partial_in(cfg: DictConfig):
    cfg_local = deepcopy(cfg)
    if "model" not in cfg_local:
        raise ValueError("Config must contain a model section.")
    cfg_local.model._target_ = "src.implementations.models.ccvar.CCVARPartialInModel"
    return cfg_local


def _run_distributed_case(
    cfg,
    cc_data,
    clusters,
    interface_idx_by_dim,
    forecast_horizon,
    case_name,
    protocol_overrides,
    imputer_target=None,
):
    cfg_case = _force_boundary_model_to_partial_in(cfg)
    for key, value in protocol_overrides.items():
        cfg_case.protocol[key] = value
    if imputer_target is not None:
        cfg_case.imputer._target_ = imputer_target

    agent_list, cluster_out_global_idx, T = create_cluster_agents(
        cfg=deepcopy(cfg_case),
        cc_data=cc_data,
        clusters=clusters,
        force_in_equals_out=False,
        protocol_overrides=protocol_overrides,
    )
    if T <= forecast_horizon:
        raise ValueError(f"{case_name}: need T ({T}) > Tstep ({forecast_horizon}).")

    curve = np.full(T - forecast_horizon, np.nan, dtype=float)
    curve_by_dim = {
        dim: np.full(T - forecast_horizon, np.nan, dtype=float)
        for dim in sorted(cc_data.keys())
    }
    pending_prediction_by_eval_t = {}

    progress_bar = tqdm(range(0, T), desc=case_name)
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
            pred_vec_by_dim, mask_by_dim = _aggregate_interface_predictions(
                prediction_by_cluster=pending_prediction_by_cluster,
                cc_data=cc_data,
                global_to_local_idx=clusters.global_to_local_idx,
                cluster_out_global_idx=cluster_out_global_idx,
                interface_idx_by_dim=interface_idx_by_dim,
            )
            eval_i = t - forecast_horizon
            curve[eval_i], nmse_by_dim = _compute_interface_nmse(
                pred_vec_by_dim=pred_vec_by_dim,
                mask_by_dim=mask_by_dim,
                cc_data=cc_data,
                t_eval=t,
            )
            for dim, value in nmse_by_dim.items():
                if dim in curve_by_dim:
                    curve_by_dim[dim][eval_i] = value
            postfix = {"bNMSE": f"{curve[eval_i]:.3e}"}
            if 0 in nmse_by_dim and np.isfinite(nmse_by_dim[0]):
                postfix["NMSE0"] = f"{nmse_by_dim[0]:.3e}"
            if 2 in nmse_by_dim and np.isfinite(nmse_by_dim[2]):
                postfix["NMSE2"] = f"{nmse_by_dim[2]:.3e}"
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

    return {"overall": curve, "by_dim": curve_by_dim}


def _run_global_case(cfg, cc_data, cellular_complex, interface_mask_by_dim, forecast_horizon, case_name):
    algorithm_param = _get_ccvar_algorithm_param(cfg)
    global_ccvar = CCVAR(algorithmParam=algorithm_param, cellularComplex=cellular_complex)
    T = min(cc_data[dim].shape[1] for dim in cc_data)
    if T <= forecast_horizon:
        raise ValueError(f"{case_name}: need T ({T}) > Tstep ({forecast_horizon}).")

    curve = np.full(T - forecast_horizon, np.nan, dtype=float)
    curve_by_dim = {
        dim: np.full(T - forecast_horizon, np.nan, dtype=float)
        for dim in sorted(cc_data.keys())
    }
    pending_prediction_by_eval_t = {}
    progress_bar = tqdm(range(0, T), desc=case_name)

    for t in progress_bar:
        pending_prediction_by_dim = pending_prediction_by_eval_t.pop(t, None)
        if pending_prediction_by_dim is not None:
            eval_i = t - forecast_horizon
            curve[eval_i], nmse_by_dim = _compute_interface_nmse(
                pred_vec_by_dim=pending_prediction_by_dim,
                mask_by_dim=interface_mask_by_dim,
                cc_data=cc_data,
                t_eval=t,
            )
            for dim, value in nmse_by_dim.items():
                if dim in curve_by_dim:
                    curve_by_dim[dim][eval_i] = value
            postfix = {"bNMSE": f"{curve[eval_i]:.3e}"}
            if 0 in nmse_by_dim and np.isfinite(nmse_by_dim[0]):
                postfix["NMSE0"] = f"{nmse_by_dim[0]:.3e}"
            if 2 in nmse_by_dim and np.isfinite(nmse_by_dim[2]):
                postfix["NMSE2"] = f"{nmse_by_dim[2]:.3e}"
            progress_bar.set_postfix(postfix)

        input_data = {dim: cc_data[dim][:, t] for dim in cc_data}
        global_ccvar.update(inputData=input_data)
        global_forecast = _forecast_ccvar_with_horizon(model=global_ccvar, forecast_horizon=forecast_horizon)
        eval_t = t + forecast_horizon
        if eval_t < T:
            pending_prediction_by_eval_t[eval_t] = _select_global_forecast_horizon(
                prediction_by_dim=global_forecast,
                forecast_horizon=forecast_horizon,
            )

    return {"overall": curve, "by_dim": curve_by_dim}


def _save_boundary_plot_and_summary(output_root, curves):
    overall_curves = {label: payload["overall"] for label, payload in curves.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, curve in overall_curves.items():
        ax.plot(np.asarray(curve, dtype=float), linewidth=2.5, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("Interface tvNMSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    pdf_path = output_root / "boundary_error_comparison.pdf"
    pkl_path = output_root / "boundary_error_comparison.pkl"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    with open(pkl_path, "wb") as file:
        pickle.dump(fig, file, protocol=pickle.HIGHEST_PROTOCOL)
    plt.close(fig)

    for dim in (0, 2):
        has_dim = any(dim in payload["by_dim"] for payload in curves.values())
        if not has_dim:
            continue

        fig_dim, ax_dim = plt.subplots(figsize=(10, 6))
        for label, payload in curves.items():
            if dim in payload["by_dim"]:
                ax_dim.plot(np.asarray(payload["by_dim"][dim], dtype=float), linewidth=2.5, label=label)
        ax_dim.set_xlabel("t")
        ax_dim.set_ylabel(f"Interface tvNMSE (dim={dim})")
        ax_dim.grid(True, alpha=0.3)
        ax_dim.legend(loc="best")
        fig_dim.tight_layout()

        pdf_dim_path = output_root / f"boundary_error_nmse{dim}_comparison.pdf"
        pkl_dim_path = output_root / f"boundary_error_nmse{dim}_comparison.pkl"
        fig_dim.savefig(pdf_dim_path, format="pdf", bbox_inches="tight")
        with open(pkl_dim_path, "wb") as file:
            pickle.dump(fig_dim, file, protocol=pickle.HIGHEST_PROTOCOL)
        plt.close(fig_dim)

    summary_lines = ["Interface-only boundary tvNMSE summary", ""]
    for label, payload in curves.items():
        curve = payload["overall"]
        arr = np.asarray(curve, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            last = np.nan
            mean_last10 = np.nan
        else:
            last = float(finite[-1])
            tail_len = max(1, int(np.ceil(0.1 * finite.size)))
            mean_last10 = float(np.mean(finite[-tail_len:]))
        summary_lines.append(f"{label}: last={last:.6e}, last10%={mean_last10:.6e}")
        for dim in (0, 2):
            if dim not in payload["by_dim"]:
                continue
            dim_arr = np.asarray(payload["by_dim"][dim], dtype=float).reshape(-1)
            dim_finite = dim_arr[np.isfinite(dim_arr)]
            if dim_finite.size == 0:
                dim_last = np.nan
                dim_last10 = np.nan
            else:
                dim_last = float(dim_finite[-1])
                dim_tail_len = max(1, int(np.ceil(0.1 * dim_finite.size)))
                dim_last10 = float(np.mean(dim_finite[-dim_tail_len:]))
            summary_lines.append(f"  NMSE{dim}: last={dim_last:.6e}, last10%={dim_last10:.6e}")
    summary_path = output_root / "boundary_error_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines) + "\n")


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    ensure_metrics_registered()
    cc_data, cellular_complex = load_data(cfg.dataset)
    clusters = instantiate(config=cfg.clustering, cellularComplex=cellular_complex)

    forecast_horizon = _get_forecast_horizon(cfg)
    interface_idx_by_dim = _build_interface_idx_by_dim(clusters=clusters, cc_data=cc_data)
    interface_mask_by_dim = _build_interface_mask_by_dim(interface_idx_by_dim=interface_idx_by_dim, cc_data=cc_data)

    output_root = (Path.cwd() / "outputs" / cfg.dataset.dataset_name / "boundary_error").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    curves = {}
    curves["Distributed (C_data=10, C_param=10)"] = _run_distributed_case(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters,
        interface_idx_by_dim=interface_idx_by_dim,
        forecast_horizon=forecast_horizon,
        case_name="Boundary Dist (10,10)",
        protocol_overrides={"C_data": 10, "C_param": 10},
    )
    curves["Distributed (C_data=100, C_param=10)"] = _run_distributed_case(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters,
        interface_idx_by_dim=interface_idx_by_dim,
        forecast_horizon=forecast_horizon,
        case_name="Boundary Dist (100,10)",
        protocol_overrides={"C_data": 100, "C_param": 10},
    )
    # curves["Distributed + ZeroHold"] = _run_distributed_case(
    #     cfg=deepcopy(cfg),
    #     cc_data=cc_data,
    #     clusters=clusters,
    #     interface_idx_by_dim=interface_idx_by_dim,
    #     forecast_horizon=forecast_horizon,
    #     case_name="Boundary Dist ZeroHold",
    #     protocol_overrides={"C_data": 10, "C_param": 10},
    #     imputer_target="src.implementations.imputer.zero_padder.ZeroPadder",
    # )
    curves["Global CC-VAR"] = _run_global_case(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        cellular_complex=cellular_complex,
        interface_mask_by_dim=interface_mask_by_dim,
        forecast_horizon=forecast_horizon,
        case_name="Boundary Global",
    )

    _save_boundary_plot_and_summary(output_root=output_root, curves=curves)
    print(f"Boundary comparison artifacts saved under: {output_root}")


if __name__ == "__main__":
    main()
