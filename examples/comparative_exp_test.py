from copy import deepcopy
from pathlib import Path
import pickle

import hydra
import matplotlib
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm  # type: ignore[import-untyped]

from cellexp_util.registry.metric_registry import ensure_metrics_registered
from cellexp_util.metric.metric_utils import MetricManager
from ccvar import CCVAR

from examples.utils.clustering_utils import create_cluster_agents
from examples.utils.data_utils import load_data
from examples.utils.metric_utils import (
    evaluate_pending_predictions,
    init_metric_managers,
    keep_only_metrics,
)
from src.implementations.models.topolms import topoLMS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EDGE_DIM = 1
KEEP_METRICS = ["tvNMSE", "rollingNMSE"]
DISAGREEMENT_METRICS = [
    "tvSelfDisagreement",
    "tvGlobalDisagreement",
    "rollingSelfDisagreement",
    "rollingGlobalDisagreement",
    "tvSelfRMS",
    "rollingSelfRMS",
    "tvCentroidToGlobal",
    "rollingCentroidToGlobal",
    "tvCentroidToGlobalRelative",
    "rollingCentroidToGlobalRelative",
    "tvGlobalRMS",
    "rollingGlobalRMS",
    "tvGlobalRMSRelative",
    "rollingGlobalRMSRelative",
]
METRIC_DISPLAY_LABELS = {
    "tvGlobalRMSRelative": "Global Disagreement",
}
LINESTYLE_CYCLE = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


def _extract_horizon_vector(values, forecast_horizon: int):
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


def _select_forecast_horizon(prediction_by_cluster, forecast_horizon: int):
    selected_prediction = {}
    for cluster_head in sorted(prediction_by_cluster.keys()):
        selected_prediction[cluster_head] = {}
        for dim in sorted(prediction_by_cluster[cluster_head].keys()):
            selected_prediction[cluster_head][dim] = _extract_horizon_vector(
                values=prediction_by_cluster[cluster_head][dim],
                forecast_horizon=forecast_horizon,
            )
    return selected_prediction


def _mean_last_fraction(values, fraction=0.1):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.nan
    tail_len = max(1, int(np.ceil(arr.size * fraction)))
    tail = arr[-tail_len:]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return np.nan
    return float(np.mean(tail))


def _flatten_theta_dict(theta_dict):
    chunks = []
    for key in sorted(theta_dict.keys()):
        chunks.append(np.asarray(theta_dict[key], dtype=float).reshape(-1))
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def _snapshot_case_params(agent_list):
    snapshot = {}
    for cluster_head in sorted(agent_list.keys()):
        snapshot[cluster_head] = np.asarray(agent_list[cluster_head]._model.get_params(), dtype=float).reshape(-1).copy()
    return snapshot


def _to_plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _modularity_clustering_cfg(base_cfg: DictConfig):
    return {
        "_target_": "src.cc_utils.clustering.ModularityBasedClustering",
        "clusteringParameters": _to_plain(base_cfg.clustering.clusteringParameters),
    }


def _build_test_cases(base_cfg: DictConfig):
    common_protocol = {
        "C_data": int(base_cfg.protocol.C_data),
        "C_param": int(base_cfg.protocol.C_param),
    }
    clustering_cfg = _modularity_clustering_cfg(base_cfg=base_cfg)
    topolms_model_cfg = {
        "_target_": "src.implementations.models.topolms.topoLMSPartialModel",
        "_convert_": "partial",
        "algorithmParam": {
            "signal_key": EDGE_DIM,
            "M": 2,
            "P": 2,
            "T": 1,
            "in_idx": {},
            "out_idx": {},
        },
    }
    ccvar_model_cfg = {
        "_target_": "src.implementations.models.ccvar.CCVARPartialModel",
        "_convert_": "partial",
        "algorithmParam": {
            "Tstep": 1,
            "P": 2,
            "K": [2, [2, 2], 2],
            "mu": [0, [0, 0], 0],
            "lambda": 0.01,
            "gamma": 0.98,
            "enabler": [False, True, False],
            "LassoEn": False,
            "FeatureNormalzn": False,
            "BiasEn": False,
            "in_idx": {},
            "out_idx": {},
        },
    }
    kgt_mixing_cfg = {
        "_target_": "src.implementations.mixing.kgt.KGTMixingModel",
        "_convert_": "all",
        "eta": _to_plain(base_cfg.mixing.eta),
        "initial_aux_vars": _to_plain(base_cfg.mixing.initial_aux_vars),
    }
    atc_mixing_cfg = {
        "_target_": "src.implementations.mixing.diffusion_atc.DiffusionATCModel",
        "_convert_": "all",
        "initial_aux_vars": {},
        "eta": {},
    }

    return {
        "topolms_atc_modularity": {
            "name": "TopoLMS + Diffusion ATC (Modularity)",
            "label": "TopoLMS (ATC)",
            "family": "topolms",
            "consensus_mode": "gated",
            "protocol_overrides": common_protocol,
            "clustering": clustering_cfg,
            "model": topolms_model_cfg,
            "mixing": atc_mixing_cfg,
        },
        "topolms_kgt_modularity": {
            "name": "TopoLMS + KGT (Modularity)",
            "label": "TopoLMS (KGT)",
            "family": "topolms",
            "consensus_mode": "gated",
            "protocol_overrides": common_protocol,
            "clustering": clustering_cfg,
            "model": topolms_model_cfg,
            "mixing": kgt_mixing_cfg,
        },
        "ccvar_kgt_modularity": {
            "name": "CCVAR + KGT (Modularity)",
            "label": "CC-VAR (KGT)",
            "family": "ccvar",
            "consensus_mode": "gated",
            "protocol_overrides": common_protocol,
            "clustering": clustering_cfg,
            "model": ccvar_model_cfg,
            "mixing": kgt_mixing_cfg,
        },
        "ccvar_atc_modularity": {
            "name": "CCVAR + Diffusion ATC (Modularity)",
            "label": "CC-VAR (ATC)",
            "family": "ccvar",
            "consensus_mode": "gated",
            "protocol_overrides": common_protocol,
            "clustering": clustering_cfg,
            "model": ccvar_model_cfg,
            "mixing": atc_mixing_cfg,
        },
    }


def _build_case_cfg(base_cfg: DictConfig, case_def: dict):
    cfg_case = deepcopy(base_cfg)
    for section_name in ("model", "mixing", "clustering"):
        if section_name in case_def and case_def[section_name] is not None:
            setattr(cfg_case, section_name, OmegaConf.create(case_def[section_name]))
    return cfg_case


def _run_case(
    cfg_case: DictConfig,
    case_def: dict,
    cc_data,
    cellular_complex,
    forecast_horizon: int,
):
    cc_data_edge = {EDGE_DIM: cc_data[EDGE_DIM]}
    clusters = instantiate(config=cfg_case.clustering, cellularComplex=cellular_complex)
    protocol_overrides = case_def.get("protocol_overrides", {})
    consensus_mode = case_def.get("consensus_mode", "gated")
    agent_list, cluster_out_global_idx, T = create_cluster_agents(
        cfg=deepcopy(cfg_case),
        cc_data=cc_data_edge,
        clusters=clusters,
        force_in_equals_out=False,
        protocol_overrides=protocol_overrides,
        snapshot_agent=False,
    )
    if T is None or T <= forecast_horizon:
        raise ValueError(f"{case_def.get('name', 'case')}: need T ({T}) > forecast_horizon ({forecast_horizon}).")

    output_dir_fn = lambda dim: Path.cwd() / "outputs" / cfg_case.dataset.dataset_name / "comparative_exp_test" / case_def["name"].lower().replace(" ", "_") / f"results_{dim}"
    metrics, _ = init_metric_managers(
        cc_data=cc_data_edge,
        output_dir_fn=output_dir_fn,
        T_eval=T - forecast_horizon,
        keep_metrics=KEEP_METRICS,
    )

    pending_prediction_by_eval_t = {}
    param_history = []
    progress_bar = tqdm(range(0, T), desc=case_def.get("name", "Comparative case"))

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
            postfix, _ = evaluate_pending_predictions(
                metrics=metrics,
                pending_prediction_by_cluster=pending_prediction_by_cluster,
                cc_data=cc_data_edge,
                cluster_out_global_idx=cluster_out_global_idx,
                t=t,
                forecast_horizon=forecast_horizon,
            )
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

        if consensus_mode == "always":
            for cluster_head in agent_list:
                agent_list[cluster_head].do_consensus()
        elif consensus_mode == "gated":
            for cluster_head in agent_list:
                if has_fresh_neighbor_params[cluster_head]:
                    agent_list[cluster_head].do_consensus()

        param_history.append(_snapshot_case_params(agent_list))

        raw_prediction_by_cluster = {}
        for cluster_head in agent_list:
            raw_prediction_by_cluster[cluster_head] = agent_list[cluster_head].estimate(
                input_data=None, steps=forecast_horizon
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

    nmse_curve = np.asarray(metrics[EDGE_DIM]._errors.get("tvNMSE", []), dtype=float).reshape(-1)
    rolling_curve = np.asarray(metrics[EDGE_DIM]._errors.get("rollingNMSE", []), dtype=float).reshape(-1)
    return {
        "nmse_curve": nmse_curve,
        "rolling_nmse_curve": rolling_curve,
        "last10_tvnmse": _mean_last_fraction(nmse_curve, fraction=0.1),
        "last10_rolling_nmse": _mean_last_fraction(rolling_curve, fraction=0.1),
        "param_history": param_history,
        "C_data": protocol_overrides.get("C_data", cfg_case.protocol.get("C_data", "NA")),
        "C_param": protocol_overrides.get("C_param", cfg_case.protocol.get("C_param", "NA")),
    }


def _build_global_ccvar_param_history(case_def: dict, cc_data, cellular_complex, T):
    algorithm_param = deepcopy(case_def["model"]["algorithmParam"])
    algorithm_param.pop("in_idx", None)
    algorithm_param.pop("out_idx", None)
    algorithm_param["enabler"] = [False, True, False]
    global_model = CCVAR(algorithmParam=algorithm_param, cellularComplex=cellular_complex)

    history = []
    for t in range(T):
        global_model.update(inputData={EDGE_DIM: cc_data[EDGE_DIM][:, t]})
        history.append(_flatten_theta_dict(global_model._theta).copy())
    return history


def _build_global_topolms_param_history(case_def: dict, cc_data, cellular_complex, T):
    algorithm_param = deepcopy(case_def["model"]["algorithmParam"])
    M = int(algorithm_param.get("M", 2))
    B1 = np.asarray(cellular_complex.get(1), dtype=float)
    B2 = np.asarray(cellular_complex.get(2), dtype=float)
    L_lower = B1.T @ B1
    L_upper = B2 @ B2.T

    global_model = topoLMS({"L_lower": L_lower, "L_upper": L_upper, "M": M, "T": 1})
    history = []
    for t in range(T):
        global_model.updateParameters(incomingData={"s_next": cc_data[EDGE_DIM][:, t]})
        history.append(np.asarray(global_model.h, dtype=float).reshape(-1).copy())
    return history


def _compute_case_disagreement(result, global_history):
    T = min(len(result["param_history"]), len(global_history))
    mm = MetricManager(N=1, T=T, savePath=".")
    keep_only_metrics(mm=mm, keep=DISAGREEMENT_METRICS)

    for t in range(T):
        mm.step_calculation(
            i=t,
            prediction=result["param_history"][t],
            groundTruth={"global_vector": global_history[t]},
            verbose=False,
        )
    return {metric: np.asarray(mm._errors.get(metric, []), dtype=float).reshape(-1) for metric in DISAGREEMENT_METRICS}


def _save_outputs(output_root: Path, case_results: dict):
    output_root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.61, 6.65))
    for idx, (_, result) in enumerate(case_results.items()):
        ax.plot(
            result["nmse_curve"],
            linewidth=3.0,
            linestyle=LINESTYLE_CYCLE[idx % len(LINESTYLE_CYCLE)],
            label=result["label"],
        )
    ax.set_xlabel("t", fontsize=40, fontname="Helvetica")
    ax.set_ylabel("NMSE", fontsize=40, fontname="Helvetica")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", prop={"family": "Helvetica", "size": 25})
    ax.tick_params(axis="both", labelsize=25)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Helvetica")
    fig.tight_layout()
    fig.savefig(output_root / "edge_tvNMSE_comparison.pdf", format="pdf", bbox_inches="tight")
    with open(output_root / "edge_tvNMSE_comparison.pkl", "wb") as file:
        pickle.dump(fig, file, protocol=pickle.HIGHEST_PROTOCOL)
    plt.close(fig)

    md_lines = [
        "# Comparative Experiment Test (Modularity, Edges)",
        "",
        "| Case | C_data | C_param | Last 10% tvNMSE | Last 10% rollingNMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, result in case_results.items():
        tv_nmse = "nan" if not np.isfinite(result["last10_tvnmse"]) else f"{result['last10_tvnmse']:.6e}"
        rolling_nmse = (
            "nan"
            if not np.isfinite(result["last10_rolling_nmse"])
            else f"{result['last10_rolling_nmse']:.6e}"
        )
        md_lines.append(
            f"| {result['label']} | {result['C_data']} | {result['C_param']} | {tv_nmse} | {rolling_nmse} |"
        )
    with open(output_root / "edge_nmse_summary.md", "w", encoding="utf-8") as file:
        file.write("\n".join(md_lines) + "\n")


def _save_disagreement_outputs(output_root: Path, case_results: dict, disagreement_by_case: dict):
    dis_dir = output_root / "disagreement"
    dis_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Comparative Disagreement Summary (Last 10%)",
        "",
        "| Metric | " + " | ".join(case_results[case_id]["label"] for case_id in case_results) + " |",
        "|" + "---|" + "|".join("---:" for _ in case_results) + "|",
    ]

    for metric_name in DISAGREEMENT_METRICS:
        ylabel = METRIC_DISPLAY_LABELS.get(metric_name, metric_name)
        row_vals = []
        for case_id in case_results:
            metric_values = disagreement_by_case[case_id].get(metric_name, np.array([], dtype=float))
            row_vals.append(_mean_last_fraction(metric_values, fraction=0.1))
        formatted = [("nan" if not np.isfinite(v) else f"{v:.6e}") for v in row_vals]
        md_lines.append(f"| {metric_name} | " + " | ".join(formatted) + " |")

        fig, ax = plt.subplots(figsize=(7.61, 6.65))
        for idx, (case_id, result) in enumerate(case_results.items()):
            series = disagreement_by_case[case_id].get(metric_name, np.array([], dtype=float))
            ax.plot(
                series,
                linewidth=3.0,
                linestyle=LINESTYLE_CYCLE[idx % len(LINESTYLE_CYCLE)],
                label=result["label"],
            )
        ax.set_xlabel("t", fontsize=40, fontname="Helvetica")
        ax.set_ylabel(ylabel, fontsize=40, fontname="Helvetica")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", prop={"family": "Helvetica", "size": 25})
        ax.tick_params(axis="both", labelsize=25)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Helvetica")
        fig.tight_layout()

        pdf_path = dis_dir / f"edge_{metric_name}_comparison.pdf"
        pkl_path = dis_dir / f"edge_{metric_name}_comparison.pkl"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        with open(pkl_path, "wb") as file:
            pickle.dump(fig, file, protocol=pickle.HIGHEST_PROTOCOL)
        plt.close(fig)

    with open(dis_dir / "edge_disagreement_summary.md", "w", encoding="utf-8") as file:
        file.write("\n".join(md_lines) + "\n")


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    ensure_metrics_registered()

    cc_data, cellular_complex = load_data(cfg.dataset)
    forecast_horizon = 1
    case_defs = _build_test_cases(base_cfg=cfg)

    case_results = {}
    for case_id, case_def in case_defs.items():
        cfg_case = _build_case_cfg(base_cfg=cfg, case_def=case_def)
        result = _run_case(
            cfg_case=cfg_case,
            case_def=case_def,
            cc_data=cc_data,
            cellular_complex=cellular_complex,
            forecast_horizon=forecast_horizon,
        )
        result["label"] = case_def["label"]
        result["family"] = case_def.get("family", "")
        result["case_id"] = case_id
        case_results[case_id] = result

    T_ref = min(
        min(len(case_results[case_id]["param_history"]) for case_id in case_results),
        int(cc_data[EDGE_DIM].shape[1]),
    )
    global_ccvar_case = next(case_defs[case_id] for case_id in case_results if case_results[case_id]["family"] == "ccvar")
    global_topolms_case = next(case_defs[case_id] for case_id in case_results if case_results[case_id]["family"] == "topolms")
    global_param_history = {
        "ccvar": _build_global_ccvar_param_history(
            case_def=global_ccvar_case,
            cc_data=cc_data,
            cellular_complex=cellular_complex,
            T=T_ref,
        ),
        "topolms": _build_global_topolms_param_history(
            case_def=global_topolms_case,
            cc_data=cc_data,
            cellular_complex=cellular_complex,
            T=T_ref,
        ),
    }

    disagreement_by_case = {}
    for case_id, result in case_results.items():
        family = result["family"]
        disagreement_by_case[case_id] = _compute_case_disagreement(
            result=result,
            global_history=global_param_history[family],
        )

    output_root = (Path.cwd() / "outputs" / cfg.dataset.dataset_name / "comparative_exp_test").resolve()
    _save_outputs(output_root=output_root, case_results=case_results)
    _save_disagreement_outputs(
        output_root=output_root,
        case_results=case_results,
        disagreement_by_case=disagreement_by_case,
    )
    print(f"Comparative test experiment completed. Outputs saved under: {output_root}")


if __name__ == "__main__":
    main()
