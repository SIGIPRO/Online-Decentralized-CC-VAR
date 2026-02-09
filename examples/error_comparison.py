from copy import deepcopy
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # type: ignore[import-untyped]

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ccvar import CCVAR
from cellexp_util.registry.metric_registry import ensure_metrics_registered

from examples.utils.data_utils import load_data
from examples.utils.clustering_utils import create_cluster_agents
from examples.utils.metric_utils import (
    evaluate_pending_predictions,
    init_metric_managers,
    save_metric_plots,
)

ERROR_METRICS = [
    "tvNMSE",
    "tvMAE",
    "tvMAPE",
    "rollingNMSE",
    "rollingMAE",
    "rollingMAPE",
]

TV_ERROR_METRICS = ["tvNMSE", "tvMAE", "tvMAPE"]

DEFAULT_CASE_DEFS = {
    "global": {
        "name": "Global CC-VAR (Direct CCVAR)",
        "label": "Global",
        "runner": "global_direct",
        "consensus_mode": "off",
        "force_in_equals_out": False,
        "protocol_overrides": {},
    },
    "pure_local": {
        "name": "Pure Local CC-VAR (Nin Open, No Comm)",
        "label": "Pure Local",
        "runner": "pure_local_direct",
        "consensus_mode": "off",
        "force_in_equals_out": False,
        "protocol_overrides": {},
    },
    "parameter_only": {
        "name": "Parameter-Only CC-VAR",
        "label": "Parameter Only",
        "runner": "distributed",
        "consensus_mode": "gated",
        "force_in_equals_out": True,
        "protocol_overrides": {"C_data": int(1e9)},
    },
    "parameter_dataset": {
        "name": "Parameter + Dataset CC-VAR (Gated)",
        "label": "Parameter + Dataset",
        "runner": "distributed",
        "consensus_mode": "gated",
        "force_in_equals_out": False,
        "protocol_overrides": {},
    },
}


def _slugify_case_name(case_name: str) -> str:
    return case_name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _to_plain_config(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _format_k_for_suffix(k_value):
    plain = _to_plain_config(k_value)
    if isinstance(plain, list):
        parts = []
        for item in plain:
            if isinstance(item, list):
                parts.append("-".join(str(x) for x in item))
            else:
                parts.append(str(item))
        return "k_" + "_".join(parts).replace(" ", "")
    return f"k_{str(plain).replace(' ', '')}"


def _format_k_for_display(k_value):
    plain = _to_plain_config(k_value)
    return str(plain).replace(" ", "")


def _get_run_parameters(cfg):
    c_data = cfg.protocol.get("C_data", "NA")
    c_param = cfg.protocol.get("C_param", "NA")
    mixing_eta = cfg.mixing.get("eta", {})
    c_val = mixing_eta.get("c", "NA")
    clustering_params = cfg.clustering.get("clusteringParameters", {})
    q_hop = clustering_params.get("Q-hop", "NA")
    model_algorithm = cfg.model.get("algorithmParam", {})
    k_value = model_algorithm.get("K", "NA")
    return c_data, c_param, c_val, q_hop, _format_k_for_display(k_value), _format_k_for_suffix(k_value)


def _parameter_suffix(c_data, c_param, c_val, q_hop, k_suffix):
    return f"c_data_{c_data}_c_param_{c_param}_c_{c_val}_q_hop_{q_hop}_{k_suffix}"


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


def _merge_case_def(base_def, override_def):
    merged = deepcopy(base_def)
    for key, value in override_def.items():
        if key == "protocol_overrides":
            merged_protocol = dict(merged.get("protocol_overrides", {}))
            if value is not None:
                merged_protocol.update(dict(value))
            merged["protocol_overrides"] = merged_protocol
        else:
            merged[key] = value
    return merged


def _resolve_case_plan(cfg):
    exp_cfg = cfg.get("experiment", {})
    run_cfg = exp_cfg.get("run", {})
    enabled_cases = run_cfg.get("enabled_cases", list(DEFAULT_CASE_DEFS.keys()))
    cases_cfg = exp_cfg.get("cases", {})

    plan = []
    for case_id in enabled_cases:
        if case_id not in DEFAULT_CASE_DEFS:
            continue
        override_def = _to_plain_config(cases_cfg.get(case_id, {})) or {}
        resolved = _merge_case_def(DEFAULT_CASE_DEFS[case_id], override_def)
        resolved["id"] = case_id
        plan.append(resolved)
    return plan


def _get_ccvar_algorithm_param(cfg):
    algorithm_param = deepcopy(_to_plain_config(cfg.model.get("algorithmParam", {})) or {})
    algorithm_param.pop("in_idx", None)
    algorithm_param.pop("out_idx", None)
    return algorithm_param


def _init_case_metrics(case_output_dir: Path, cc_data, T_eval: int):
    output_dir_fn = lambda dim: case_output_dir / f"results_{dim}"
    return init_metric_managers(
        cc_data=cc_data,
        output_dir_fn=output_dir_fn,
        T_eval=T_eval,
    )


def _save_case_metrics(metrics, output_dirs):
    for dim in metrics:
        metrics[dim].save_single(n=0)
        metrics[dim].save_full(n=1)
    save_metric_plots(metrics=metrics, output_dirs=output_dirs)


def _save_last10pct_error_tables(output_root: Path, case_metric_managers, comparison_cases, cfg):
    c_data, c_param, c_val, q_hop, k_display, k_suffix = _get_run_parameters(cfg)
    suffix = _parameter_suffix(c_data, c_param, c_val, q_hop, k_suffix)
    table_dir = output_root / "comparison_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    if not case_metric_managers:
        return
    sample_case = next(iter(case_metric_managers.values()))
    dims = sorted(sample_case.keys())

    for dim in dims:
        rows = []
        for metric_name in TV_ERROR_METRICS:
            vals = []
            for case_name, _ in comparison_cases:
                case_mm = case_metric_managers[case_name][dim]
                vals.append(_mean_last_fraction(case_mm._errors.get(metric_name, []), fraction=0.1))
            rows.append((metric_name, vals))

        labels = [label for _, label in comparison_cases]
        col_spec = "c" * len(labels)

        md_lines = [
            f"# Errors Dim {dim} (Last 10%)",
            "",
            f"Parameters: `C_data={c_data}`, `C_param={c_param}`, `c={c_val}`, `Q-hop={q_hop}`, `K={k_display}`",
            "",
            "| Metric | " + " | ".join(labels) + " |",
            "|" + "---|" + "|".join("---:" for _ in labels) + "|",
        ]
        for metric_name, vals in rows:
            formatted = [("nan" if not np.isfinite(v) else f"{v:.6e}") for v in vals]
            md_lines.append(f"| {metric_name} | " + " | ".join(formatted) + " |")

        md_path = table_dir / f"errors_dim_{dim}_{suffix}.md"
        with open(md_path, "w", encoding="utf-8") as file:
            file.write("\n".join(md_lines) + "\n")

        tex_lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Last 10\\% mean of TV error metrics for dim={dim} ($C_{{data}}={c_data}$, $C_{{param}}={c_param}$, $c={c_val}$, $Q\\text{{-}}hop={q_hop}$, $K={k_display}$).}}",
            f"\\begin{{tabular}}{{l{col_spec}}}",
            "\\hline",
            "Metric & " + " & ".join(labels) + " \\\\",
            "\\hline",
        ]
        for metric_name, vals in rows:
            formatted = [("nan" if not np.isfinite(v) else f"{v:.6e}") for v in vals]
            tex_lines.append(f"{metric_name} & " + " & ".join(formatted) + " \\\\")
        tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

        tex_path = table_dir / f"errors_dim_{dim}_{suffix}.tex"
        with open(tex_path, "w", encoding="utf-8") as file:
            file.write("\n".join(tex_lines) + "\n")


def _save_error_comparison_plots(output_root: Path, case_metric_managers, comparison_cases, cfg):
    c_data, c_param, c_val, q_hop, k_display, k_suffix = _get_run_parameters(cfg)
    suffix = _parameter_suffix(c_data, c_param, c_val, q_hop, k_suffix)
    figure_dir = output_root / "comparison_plots"
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not case_metric_managers:
        return
    sample_case = next(iter(case_metric_managers.values()))
    dims = sorted(sample_case.keys())
    annotation_text = f"C_data={c_data}, C_param={c_param}, c={c_val}, Q-hop={q_hop}, K={k_display}"

    for dim in dims:
        for metric_name in ERROR_METRICS:
            fig, ax = plt.subplots(figsize=(9, 5))
            for case_name, label in comparison_cases:
                series = np.asarray(case_metric_managers[case_name][dim]._errors.get(metric_name, []), dtype=float).reshape(-1)
                ax.plot(series, linewidth=1.7, label=label)

            ax.set_title(f"{metric_name} (dim={dim})")
            ax.set_xlabel("t")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            ax.text(
                0.02,
                0.98,
                annotation_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            fig.tight_layout()

            file_stem = f"errors_dim_{dim}_{metric_name}_{suffix}"
            pdf_path = figure_dir / f"{file_stem}.pdf"
            pkl_path = figure_dir / f"{file_stem}.pkl"
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            with open(pkl_path, "wb") as file:
                pickle.dump(fig, file, protocol=pickle.HIGHEST_PROTOCOL)
            plt.close(fig)


def _build_cluster_out_idx_from_nin(clusters, cc_data):
    cluster_out_global_idx = {}
    for cluster_head in sorted(clusters.Nin.keys()):
        cluster_out_global_idx[cluster_head] = {}
        for dim in sorted(cc_data.keys()):
            cluster_out_global_idx[cluster_head][dim] = np.asarray(
                sorted(list(clusters.Nin[cluster_head].get(dim, []))),
                dtype=int,
            )
    return cluster_out_global_idx


def _project_global_prediction_to_cluster_nin(global_prediction, cluster_out_global_idx):
    prediction_by_cluster = {}
    for cluster_head in sorted(cluster_out_global_idx.keys()):
        prediction_by_cluster[cluster_head] = {}
        for dim, idx in cluster_out_global_idx[cluster_head].items():
            full_vec = np.asarray(global_prediction.get(dim, np.array([])), dtype=float).reshape(-1)
            if idx.size == 0 or full_vec.size == 0:
                prediction_by_cluster[cluster_head][dim] = np.array([], dtype=float)
                continue
            valid_idx = idx[idx < full_vec.size]
            prediction_by_cluster[cluster_head][dim] = full_vec[valid_idx]
    return prediction_by_cluster


def _create_pure_local_ccvar_states(cfg, cc_data, clusters):
    T = None
    state_by_cluster = {}
    cluster_out_global_idx = {}
    algorithm_param = _get_ccvar_algorithm_param(cfg)

    for cluster_head in sorted(clusters.Nin.keys()):
        nin_idx = {}
        processed_data = {}
        for dim in sorted(clusters.Nin[cluster_head].keys()):
            nin_idx[dim] = sorted(list(clusters.Nin[cluster_head].get(dim, [])))
            if dim in cc_data.keys():
                processed_data[dim] = cc_data[dim][nin_idx[dim], :]

        local_complex = {}
        if 1 in clusters.cellularComplex:
            local_complex[1] = clusters.cellularComplex[1][np.ix_(nin_idx[0], nin_idx[1])]
        if 2 in clusters.cellularComplex:
            local_complex[2] = clusters.cellularComplex[2][np.ix_(nin_idx[1], nin_idx[2])]

        state_by_cluster[cluster_head] = {
            "model": CCVAR(algorithmParam=algorithm_param, cellularComplex=local_complex),
            "data": processed_data,
        }
        cluster_out_global_idx[cluster_head] = {
            dim: np.asarray(nin_idx[dim], dtype=int) for dim in nin_idx
        }

        local_T = min(processed_data[dim].shape[1] for dim in processed_data)
        if T is None or local_T < T:
            T = local_T

    return state_by_cluster, cluster_out_global_idx, T


def _run_distributed_forecast_case(
    case_name: str,
    case_output_dir: Path,
    agent_list,
    cluster_out_global_idx,
    cc_data,
    T: int,
    consensus_mode: str,
):
    metrics, output_dirs = _init_case_metrics(case_output_dir=case_output_dir, cc_data=cc_data, T_eval=T - 1)
    pending_prediction_by_cluster = None

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

        if pending_prediction_by_cluster is not None:
            postfix, _ = evaluate_pending_predictions(
                metrics=metrics,
                pending_prediction_by_cluster=pending_prediction_by_cluster,
                cc_data=cc_data,
                cluster_out_global_idx=cluster_out_global_idx,
                t=t,
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

        prediction_by_cluster = {}
        for cluster_head in agent_list:
            prediction_by_cluster[cluster_head] = agent_list[cluster_head].estimate(input_data=None, steps=1)
        pending_prediction_by_cluster = prediction_by_cluster

    _save_case_metrics(metrics=metrics, output_dirs=output_dirs)
    return metrics


def _run_global_ccvar_forecast_case(case_name: str, case_output_dir: Path, cc_data, cellular_complex, clusters, cfg):
    algorithm_param = _get_ccvar_algorithm_param(cfg)
    global_ccvar = CCVAR(algorithmParam=algorithm_param, cellularComplex=cellular_complex)

    T = min(cc_data[dim].shape[1] for dim in cc_data)
    metrics, output_dirs = _init_case_metrics(case_output_dir=case_output_dir, cc_data=cc_data, T_eval=T - 1)
    global_cluster_out_idx = _build_cluster_out_idx_from_nin(clusters=clusters, cc_data=cc_data)

    pending_prediction = None
    progress_bar = tqdm(range(0, T), desc=case_name)
    for t in progress_bar:
        if pending_prediction is not None:
            postfix, _ = evaluate_pending_predictions(
                metrics=metrics,
                pending_prediction_by_cluster=pending_prediction,
                cc_data=cc_data,
                cluster_out_global_idx=global_cluster_out_idx,
                t=t,
            )
            progress_bar.set_postfix(postfix)

        input_data = {dim: cc_data[dim][:, t] for dim in cc_data}
        global_ccvar.update(inputData=input_data)
        global_forecast = global_ccvar.forecast(steps=1)
        pending_prediction = _project_global_prediction_to_cluster_nin(
            global_prediction=global_forecast,
            cluster_out_global_idx=global_cluster_out_idx,
        )

    _save_case_metrics(metrics=metrics, output_dirs=output_dirs)
    return metrics


def _run_pure_local_ccvar_forecast_case(
    case_name: str,
    case_output_dir: Path,
    state_by_cluster,
    cluster_out_global_idx,
    cc_data,
    T: int,
):
    metrics, output_dirs = _init_case_metrics(case_output_dir=case_output_dir, cc_data=cc_data, T_eval=T - 1)
    pending_prediction_by_cluster = None

    progress_bar = tqdm(range(0, T), desc=case_name)
    for t in progress_bar:
        if pending_prediction_by_cluster is not None:
            postfix, _ = evaluate_pending_predictions(
                metrics=metrics,
                pending_prediction_by_cluster=pending_prediction_by_cluster,
                cc_data=cc_data,
                cluster_out_global_idx=cluster_out_global_idx,
                t=t,
            )
            progress_bar.set_postfix(postfix)

        prediction_by_cluster = {}
        for cluster_head in sorted(state_by_cluster.keys()):
            state = state_by_cluster[cluster_head]
            input_data = {dim: state["data"][dim][:, t] for dim in sorted(state["data"].keys())}
            state["model"].update(inputData=input_data)
            prediction_by_cluster[cluster_head] = state["model"].forecast(steps=1)
        pending_prediction_by_cluster = prediction_by_cluster

    _save_case_metrics(metrics=metrics, output_dirs=output_dirs)
    return metrics


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    ensure_metrics_registered()

    cc_data, cellular_complex = load_data(cfg.dataset)
    output_root = (Path.cwd() / "outputs" / cfg.dataset.dataset_name / "error_comparison").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    case_metric_managers = {}
    comparison_cases = []

    clusters = instantiate(config=cfg.clustering, cellularComplex=cellular_complex)
    case_plan = _resolve_case_plan(cfg)

    pure_local_state_cache = None
    for case_def in case_plan:
        case_name = case_def.get("name", case_def["id"])
        case_label = case_def.get("label", case_name)
        runner = case_def.get("runner", "")
        case_output_dir = output_root / _slugify_case_name(case_name)

        if runner == "global_direct":
            case_metric_managers[case_name] = _run_global_ccvar_forecast_case(
                case_name=case_name,
                case_output_dir=case_output_dir,
                cc_data=cc_data,
                cellular_complex=cellular_complex,
                clusters=clusters,
                cfg=cfg,
            )
        elif runner == "pure_local_direct":
            if pure_local_state_cache is None:
                pure_local_state_cache = _create_pure_local_ccvar_states(
                    cfg=cfg,
                    cc_data=cc_data,
                    clusters=clusters,
                )
            pure_local_states, pure_local_out_idx, T_pure_local = pure_local_state_cache
            case_metric_managers[case_name] = _run_pure_local_ccvar_forecast_case(
                case_name=case_name,
                case_output_dir=case_output_dir,
                state_by_cluster=pure_local_states,
                cluster_out_global_idx=pure_local_out_idx,
                cc_data=cc_data,
                T=T_pure_local,
            )
        elif runner == "distributed":
            agents, out_idx, T_case = create_cluster_agents(
                cfg=deepcopy(cfg),
                cc_data=cc_data,
                clusters=clusters,
                force_in_equals_out=bool(case_def.get("force_in_equals_out", False)),
                protocol_overrides=case_def.get("protocol_overrides", {}),
            )
            case_metric_managers[case_name] = _run_distributed_forecast_case(
                case_name=case_name,
                case_output_dir=case_output_dir,
                agent_list=agents,
                cluster_out_global_idx=out_idx,
                cc_data=cc_data,
                T=T_case,
                consensus_mode=case_def.get("consensus_mode", "gated"),
            )
        else:
            continue

        comparison_cases.append((case_name, case_label))

    _save_last10pct_error_tables(
        output_root=output_root,
        case_metric_managers=case_metric_managers,
        comparison_cases=comparison_cases,
        cfg=cfg,
    )
    _save_error_comparison_plots(
        output_root=output_root,
        case_metric_managers=case_metric_managers,
        comparison_cases=comparison_cases,
        cfg=cfg,
    )

    print(f"Forecast error comparison completed. Outputs saved under: {output_root}")


if __name__ == "__main__":
    main()
