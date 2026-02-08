import numpy as np
import matplotlib.pyplot as plt
import pickle

from cellexp_util.metric.metric_utils import MetricManager, general_metric
from examples.utils.data_utils import (
    aggregate_partial_prediction_to_global,
    build_ground_truth_vector_for_partial,
)


DEFAULT_KEEP_METRICS = [
    "tvNMSE",
    "tvMAE",
    "tvMAPE",
    "rollingNMSE",
    "rollingMAE",
    "rollingMAPE",
]


def keep_only_metrics(mm, keep):
    keep = set(keep)
    mm._registry = {k: v for k, v in mm._registry.items() if k in keep}
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


def init_metric_managers(cc_data, output_dir_fn, T_eval, keep_metrics=None):
    if T_eval <= 0:
        raise ValueError("Need at least 2 time steps for one-step-ahead metric evaluation.")

    keep = list(keep_metrics) if keep_metrics is not None else list(DEFAULT_KEEP_METRICS)
    metrics = {}
    output_dirs = {}
    for dim in cc_data:
        dim_N = cc_data[dim].shape[0]
        curr_output = output_dir_fn(dim)
        curr_output.mkdir(parents=True, exist_ok=True)
        output_dirs[dim] = curr_output
        metrics[dim] = MetricManager(N=dim_N, T=T_eval, savePath=str(curr_output))
        keep_only_metrics(mm=metrics[dim], keep=keep)

    return metrics, output_dirs


def evaluate_pending_predictions(metrics, pending_prediction_by_cluster, cc_data, cluster_out_global_idx, t):
    postfix = {}
    eval_i = t - 1
    for dim in sorted(cc_data.keys()):
        pred_vec, valid_mask = aggregate_partial_prediction_to_global(
            prediction_by_cluster=pending_prediction_by_cluster,
            cluster_out_global_idx=cluster_out_global_idx,
            dim_size=cc_data[dim].shape[0],
            dim=dim,
        )
        gt_vec = build_ground_truth_vector_for_partial(
            cc_data=cc_data,
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

    return postfix, eval_i


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

    gt = []
    y = []
    for cluster_head in groundTruth:
        curr_gt = groundTruth[cluster_head][dim]
        curr_y = prediction[cluster_head][dim]

        y.append(curr_y)
        gt.append(curr_gt)

    try:
        y = np.vstack(y).flatten()
    except Exception:
        y = np.hstack(y).flatten()

    try:
        gt = np.vstack(gt).flatten()
    except Exception:
        gt = np.hstack(gt).flatten()

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


def _extract_param_matrix(params):
    if isinstance(params, dict):
        if not params:
            return np.empty((0, 0), dtype=float)
        rows = []
        for key in sorted(params.keys()):
            rows.append(np.asarray(params[key], dtype=float).reshape(-1))
        min_len = min(row.shape[0] for row in rows)
        if min_len == 0:
            return np.empty((len(rows), 0), dtype=float)
        matrix = np.vstack([row[:min_len] for row in rows])
        return matrix

    arr = np.asarray(params, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _extract_global_vector(groundTruth):
    if isinstance(groundTruth, dict):
        for key in ("global_params", "global_vector", "global", "s"):
            if key in groundTruth:
                return np.asarray(groundTruth[key], dtype=float).reshape(-1)
        return np.array([], dtype=float)
    return np.asarray(groundTruth, dtype=float).reshape(-1)


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


@general_metric(name="rollingNMSE", output="scalar")
def rollingNMSE_distributed_metric(*, prediction, groundTruth, manager, dim=None, **_):
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


@general_metric(name="rollingMAE", output="scalar")
def rollingMAE_distributed_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors["tvMAEsingle"][: i + 1]))
    except Exception:
        return np.nan


@general_metric(name="rollingMAPE", output="scalar")
def rollingMAPE_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors["tvMAPEsingle"][: i + 1]))
    except Exception:
        return np.nan


@general_metric(name="tvSelfDisagreement", output="scalar")
def tvSelfDisagreement_metric(*, prediction, **_):
    theta = _extract_param_matrix(prediction)
    if theta.size == 0 or theta.shape[0] <= 1:
        return 0.0
    centered = theta - np.mean(theta, axis=0, keepdims=True)
    return float(np.mean(centered ** 2))


@general_metric(name="rollingSelfDisagreement", output="scalar")
def rollingSelfDisagreement_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors["tvSelfDisagreementsingle"][: i + 1]))
    except Exception:
        return np.nan


@general_metric(name="tvGlobalDisagreement", output="scalar")
def tvGlobalDisagreement_metric(*, prediction, groundTruth, **_):
    theta = _extract_param_matrix(prediction)
    if theta.size == 0:
        return np.nan

    global_theta = _extract_global_vector(groundTruth)
    if global_theta.size == 0:
        return np.nan

    dim = min(theta.shape[1], global_theta.shape[0])
    if dim == 0:
        return np.nan

    diff = theta[:, :dim] - global_theta[:dim]
    return float(np.mean(np.linalg.norm(diff, axis=1)))


@general_metric(name="rollingGlobalDisagreement", output="scalar")
def rollingGlobalDisagreement_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors["tvGlobalDisagreementsingle"][: i + 1]))
    except Exception:
        return np.nan


@general_metric(name="tvSelfRMS", output="scalar")
def tvSelfRMS_metric(*, prediction, **_):
    theta = _extract_param_matrix(prediction)
    if theta.size == 0 or theta.shape[0] <= 1:
        return 0.0
    centered = theta - np.mean(theta, axis=0, keepdims=True)
    sq_norms = np.sum(centered ** 2, axis=1)
    return float(np.sqrt(np.mean(sq_norms)))


@general_metric(name="rollingSelfRMS", output="scalar")
def rollingSelfRMS_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors["tvSelfRMSsingle"][: i + 1]))
    except Exception:
        return np.nan


@general_metric(name="tvCentroidToGlobal", output="scalar")
def tvCentroidToGlobal_metric(*, prediction, groundTruth, **_):
    theta = _extract_param_matrix(prediction)
    if theta.size == 0:
        return np.nan

    global_theta = _extract_global_vector(groundTruth)
    if global_theta.size == 0:
        return np.nan

    dim = min(theta.shape[1], global_theta.shape[0])
    if dim == 0:
        return np.nan

    theta_bar = np.mean(theta[:, :dim], axis=0)
    return float(np.linalg.norm(theta_bar - global_theta[:dim]))


@general_metric(name="rollingCentroidToGlobal", output="scalar")
def rollingCentroidToGlobal_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors["tvCentroidToGlobalsingle"][: i + 1]))
    except Exception:
        return np.nan


@general_metric(name="tvGlobalRMS", output="scalar")
def tvGlobalRMS_metric(*, prediction, groundTruth, **_):
    theta = _extract_param_matrix(prediction)
    if theta.size == 0:
        return np.nan

    global_theta = _extract_global_vector(groundTruth)
    if global_theta.size == 0:
        return np.nan

    dim = min(theta.shape[1], global_theta.shape[0])
    if dim == 0:
        return np.nan

    diff = theta[:, :dim] - global_theta[:dim]
    sq_norms = np.sum(diff ** 2, axis=1)
    return float(np.sqrt(np.mean(sq_norms)))


@general_metric(name="rollingGlobalRMS", output="scalar")
def rollingGlobalRMS_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors["tvGlobalRMSsingle"][: i + 1]))
    except Exception:
        return np.nan


@general_metric(name="tvGlobalRMSRelative", output="scalar")
def tvGlobalRMSRelative_metric(*, prediction, groundTruth, **_):
    global_theta = _extract_global_vector(groundTruth)
    if global_theta.size == 0:
        return np.nan
    denom = float(np.linalg.norm(global_theta))
    denom = max(denom, 1e-12)
    curr_rms = tvGlobalRMS_metric(prediction=prediction, groundTruth=groundTruth)
    if not np.isfinite(curr_rms):
        return np.nan
    return float(curr_rms / denom)


@general_metric(name="rollingGlobalRMSRelative", output="scalar")
def rollingGlobalRMSRelative_metric(*, manager, i, **_):
    try:
        return float(np.mean(manager._errors["tvGlobalRMSRelativesingle"][: i + 1]))
    except Exception:
        return np.nan
