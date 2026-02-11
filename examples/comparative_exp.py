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

from examples.utils.clustering_utils import create_cluster_agents
from examples.utils.data_utils import load_data
from examples.utils.metric_utils import evaluate_pending_predictions, init_metric_managers

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EDGE_DIM = 1
KEEP_METRICS = ["tvNMSE", "rollingNMSE"]


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")


def _to_plain_config(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


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


def _build_case_cfg(base_cfg: DictConfig, case_def: dict):
    cfg_case = deepcopy(base_cfg)
    for section_name in ("model", "mixing", "clustering"):
        if section_name in case_def and case_def[section_name] is not None:
            setattr(cfg_case, section_name, OmegaConf.create(case_def[section_name]))
    return cfg_case


def _run_distributed_edge_case(
    cfg_case: DictConfig,
    case_def: dict,
    cc_data,
    cellular_complex,
    case_output_dir: Path,
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
    )
    if T <= forecast_horizon:
        raise ValueError(f"{case_def.get('name', 'case')}: need T ({T}) > Tstep ({forecast_horizon}).")

    output_dir_fn = lambda dim: case_output_dir / f"results_{dim}"
    metrics, output_dirs = init_metric_managers(
        cc_data=cc_data_edge,
        output_dir_fn=output_dir_fn,
        T_eval=T - forecast_horizon,
        keep_metrics=KEEP_METRICS,
    )

    pending_prediction_by_eval_t = {}
    
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
        "C_data": protocol_overrides.get("C_data", cfg_case.protocol.get("C_data", "NA")),
        "C_param": protocol_overrides.get("C_param", cfg_case.protocol.get("C_param", "NA")),
    }


def _save_case_comparison_artifacts(output_root: Path, case_results: dict):
    fig, ax = plt.subplots(figsize=(7.61, 6.65))
    for _, result in case_results.items():
        ax.plot(result["nmse_curve"], linewidth=3.0, label=result["label"])
    ax.set_xlabel("t", fontsize=40, fontname="Helvetica")
    ax.set_ylabel("NMSE", fontsize=40, fontname="Helvetica")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", prop={"family": "Helvetica", "size": 25})
    ax.tick_params(axis="both", labelsize=25)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Helvetica")
    fig.tight_layout()

    fig_path = output_root / "edge_tvNMSE_comparison.pdf"
    pkl_path = output_root / "edge_tvNMSE_comparison.pkl"
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    with open(pkl_path, "wb") as file:
        pickle.dump(fig, file, protocol=pickle.HIGHEST_PROTOCOL)
    plt.close(fig)

    md_lines = [
        "# Comparative Experiment (Edges Only)",
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


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    ensure_metrics_registered()

    exp_cfg = cfg.get("experiment", {})
    run_cfg = exp_cfg.get("run", {})
    case_cfg_map = _to_plain_config(exp_cfg.get("cases", {})) or {}
    case_order = run_cfg.get("enabled_cases", list(case_cfg_map.keys()))
    forecast_horizon = int(run_cfg.get("forecast_horizon", 1))

    if not case_order:
        raise ValueError("No comparative cases configured. Please set experiment.run.enabled_cases.")

    cc_data, cellular_complex = load_data(cfg.dataset)
    output_root = (Path.cwd() / "outputs" / cfg.dataset.dataset_name / "comparative_exp").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    case_results = {}
    for case_id in case_order:
        if case_id not in case_cfg_map:
            continue
        case_def = deepcopy(case_cfg_map[case_id])
        case_cfg = _build_case_cfg(base_cfg=cfg, case_def=case_def)
        case_name = case_def.get("name", case_id)
        case_output_dir = output_root / _slugify(case_name)
        case_output_dir.mkdir(parents=True, exist_ok=True)

        result = _run_distributed_edge_case(
            cfg_case=case_cfg,
            case_def=case_def,
            cc_data=cc_data,
            cellular_complex=cellular_complex,
            case_output_dir=case_output_dir,
            forecast_horizon=forecast_horizon,
        )
        result["label"] = case_def.get("label", case_name)
        case_results[case_id] = result

    if not case_results:
        raise ValueError("No valid comparative cases were executed.")

    _save_case_comparison_artifacts(output_root=output_root, case_results=case_results)
    print(f"Comparative experiment completed. Outputs saved under: {output_root}")


if __name__ == "__main__":
    main()
