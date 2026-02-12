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

from examples.utils.cases import run_case
from examples.utils.data_utils import load_data
from examples.utils.clustering_utils import create_cluster_agents
from examples.utils.metric_utils import keep_only_metrics
from cellexp_util.metric.metric_utils import MetricManager
from cellexp_util.registry.metric_registry import ensure_metrics_registered


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

MATLAB_FIGSIZE = (7.61, 6.65)
MATLAB_LINEWIDTH = 3.0
MATLAB_AXIS_FONTSIZE = 25
MATLAB_LABEL_FONTSIZE = 40
MATLAB_FONTNAME = "Helvetica"
LINESTYLE_CYCLE = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

DEFAULT_CASE_DEFS = {
    "global": {
        "name": "Global CC-VAR",
        "label": "Global",
        "runner": "global_direct",
        "consensus_mode": "off",
        "force_in_equals_out": False,
        "protocol_overrides": {},
    },
    "pure_local": {
        "name": "Local CC-VAR (Nin Open, No Comm)",
        "label": "Local",
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
        "name": "Distributed CC-VAR (Gated)",
        "label": "Distributed",
        "runner": "distributed",
        "consensus_mode": "gated",
        "force_in_equals_out": False,
        "protocol_overrides": {},
    },
}


def _snapshot_case_params(agent_list):
    return {
        cluster_head: np.asarray(agent_list[cluster_head]._model.get_params(), dtype=float).reshape(-1)
        for cluster_head in sorted(agent_list.keys())
    }


def _init_case_metric_manager(base_output_dir: Path, case_name: str, T: int):
    safe_case_name = case_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    case_output_dir = base_output_dir / safe_case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    mm = MetricManager(N=1, T=T, savePath=str(case_output_dir))
    keep_only_metrics(mm=mm, keep=DISAGREEMENT_METRICS)
    return mm, case_output_dir


def _save_case_metrics(mm, case_output_dir: Path):
    mm.save_single(n=0)
    mm.save_full(n=1)

    summary_lines = []
    for metric_name in DISAGREEMENT_METRICS:
        values = np.asarray(mm._errors.get(metric_name, []), dtype=float).reshape(-1)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            summary_lines.append(f"{metric_name}: nan")
        else:
            summary_lines.append(f"{metric_name}: {finite_values[-1]:.6e}")

    summary_path = case_output_dir / "disagreement_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines) + "\n")


def _flatten_theta_dict(theta_dict):
    chunks = []
    for key in sorted(theta_dict.keys()):
        chunks.append(np.asarray(theta_dict[key], dtype=float).reshape(-1))
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


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


def _save_last10pct_tables(output_root: Path, case_metric_managers, comparison_cases, cfg):
    c_data, c_param, c_val, q_hop, k_display, k_suffix = _get_run_parameters(cfg)
    suffix = _parameter_suffix(c_data, c_param, c_val, q_hop, k_suffix)
    table_dir = output_root / "comparison_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    if not comparison_cases:
        return
    tv_metrics = [m for m in DISAGREEMENT_METRICS if m.startswith("tv")]
    rows = []
    for metric_name in tv_metrics:
        vals = []
        for case_name, _ in comparison_cases:
            case_mm = case_metric_managers[case_name]
            vals.append(_mean_last_fraction(case_mm._errors.get(metric_name, []), fraction=0.1))
        rows.append((metric_name, vals))

    labels = [label for _, label in comparison_cases]
    col_spec = "c" * len(labels)

    md_lines = [
        f"# Disagreement (Last 10%)",
        f"",
        f"Parameters: `C_data={c_data}`, `C_param={c_param}`, `c={c_val}`, `Q-hop={q_hop}`, `K={k_display}`",
        f"",
        "| Metric | " + " | ".join(labels) + " |",
        "|" + "---|" + "|".join("---:" for _ in labels) + "|",
    ]
    for metric_name, vals in rows:
        formatted = [("nan" if not np.isfinite(v) else f"{v:.6e}") for v in vals]
        md_lines.append(f"| {metric_name} | " + " | ".join(formatted) + " |")

    md_path = table_dir / f"disagreement_{suffix}.md"
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("\n".join(md_lines) + "\n")

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Last 10\\% mean of TV disagreement metrics ($C_{{data}}={c_data}$, $C_{{param}}={c_param}$, $c={c_val}$, $Q\\text{{-}}hop={q_hop}$, $K={k_display}$).}}",
        f"\\begin{{tabular}}{{l{col_spec}}}",
        "\\hline",
        "Metric & " + " & ".join(labels) + " \\\\",
        "\\hline",
    ]
    for metric_name, vals in rows:
        formatted = [("nan" if not np.isfinite(v) else f"{v:.6e}") for v in vals]
        tex_lines.append(f"{metric_name} & " + " & ".join(formatted) + " \\\\")
    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    tex_path = table_dir / f"disagreement_{suffix}.tex"
    with open(tex_path, "w", encoding="utf-8") as file:
        file.write("\n".join(tex_lines) + "\n")


def _save_metric_comparison_plots(output_root: Path, case_metric_managers, comparison_cases, cfg):
    c_data, c_param, c_val, q_hop, k_display, k_suffix = _get_run_parameters(cfg)
    suffix = _parameter_suffix(c_data, c_param, c_val, q_hop, k_suffix)
    figure_dir = output_root / "comparison_plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not comparison_cases:
        return

    for metric_name in DISAGREEMENT_METRICS:
        ylabel = METRIC_DISPLAY_LABELS.get(metric_name, metric_name)
        fig, ax = plt.subplots(figsize=MATLAB_FIGSIZE)
        for idx, (case_name, label) in enumerate(comparison_cases):
            series = np.asarray(case_metric_managers[case_name]._errors.get(metric_name, []), dtype=float).reshape(-1)
            ax.plot(
                series,
                linewidth=MATLAB_LINEWIDTH,
                linestyle=LINESTYLE_CYCLE[idx % len(LINESTYLE_CYCLE)],
                label=label,
            )

        ax.set_xlabel("t", fontsize=MATLAB_LABEL_FONTSIZE, fontname=MATLAB_FONTNAME)
        ax.set_ylabel(ylabel, fontsize=MATLAB_LABEL_FONTSIZE, fontname=MATLAB_FONTNAME)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", prop={"family": MATLAB_FONTNAME, "size": MATLAB_AXIS_FONTSIZE})
        ax.tick_params(axis="both", labelsize=MATLAB_AXIS_FONTSIZE)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname(MATLAB_FONTNAME)
        fig.tight_layout()

        file_stem = f"disagreement_{metric_name}_{suffix}"
        pdf_path = figure_dir / f"{file_stem}.pdf"
        pkl_path = figure_dir / f"{file_stem}.pkl"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        with open(pkl_path, "wb") as file:
            pickle.dump(fig, file, protocol=pickle.HIGHEST_PROTOCOL)
        plt.close(fig)


def _create_callback(metric_manager, global_param_history):
    def _callback(t, agent_list, progress_bar):
        if not global_param_history:
            return
        snapshot = _snapshot_case_params(agent_list)
        ref_index = min(t, len(global_param_history) - 1)
        metric_manager.step_calculation(
            i=t,
            prediction=snapshot,
            groundTruth={"global_vector": global_param_history[ref_index]},
            verbose=False,
        )
        progress_bar.set_postfix(
            {
                "selfD": f"{metric_manager._errors['tvSelfDisagreement'][t]:.3e}",
                "globD": f"{metric_manager._errors['tvGlobalDisagreement'][t]:.3e}",
                "selfRMS": f"{metric_manager._errors['tvSelfRMS'][t]:.3e}",
                "globRMS": f"{metric_manager._errors['tvGlobalRMS'][t]:.3e}",
            }
        )

    return _callback


def _create_pure_local_ccvar_states(cfg, cc_data, clusters):
    T = None
    state_by_cluster = {}
    algorithm_param = _get_ccvar_algorithm_param(cfg)

    for cluster_head in sorted(clusters.Nin.keys()):
        nin_idx = {}
        processed_data = {}
        for dim in sorted(clusters.Nin[cluster_head].keys()):
            nin_idx[dim] = sorted(list(clusters.Nin[cluster_head].get(dim, [])))
            if dim in cc_data.keys():
                processed_data[dim] = cc_data[dim][nin_idx[dim], :]

        local_complex = {}
        # try:
        if 1 in clusters.cellularComplex:
            local_complex[1] = clusters.cellularComplex[1][np.ix_(nin_idx[0], nin_idx[1])]
        if 2 in clusters.cellularComplex:
            local_complex[2] = clusters.cellularComplex[2][np.ix_(nin_idx[1], nin_idx[2])]
        # except: import pdb; pdb.set_trace()

        state_by_cluster[cluster_head] = {
            "model": CCVAR(algorithmParam=algorithm_param, cellularComplex=local_complex),
            "data": processed_data,
        }

        local_T = min(processed_data[dim].shape[1] for dim in processed_data)
        if T is None or local_T < T:
            T = local_T

    return state_by_cluster, T


def _run_pure_local_ccvar_case(case_name, state_by_cluster, T, metric_manager, global_param_history):
    progress_bar = tqdm(range(T), desc=case_name)
    for t in progress_bar:
        snapshot = {}
        for cluster_head in sorted(state_by_cluster.keys()):
            state = state_by_cluster[cluster_head]
            input_data = {dim: state["data"][dim][:, t] for dim in sorted(state["data"].keys())}
            state["model"].update(inputData=input_data)
            snapshot[cluster_head] = _flatten_theta_dict(state["model"]._theta)

        ref_index = min(t, len(global_param_history) - 1)
        metric_manager.step_calculation(
            i=t,
            prediction=snapshot,
            groundTruth={"global_vector": global_param_history[ref_index]},
            verbose=False,
        )
        progress_bar.set_postfix(
            {
                "selfD": f"{metric_manager._errors['tvSelfDisagreement'][t]:.3e}",
                "globD": f"{metric_manager._errors['tvGlobalDisagreement'][t]:.3e}",
                "selfRMS": f"{metric_manager._errors['tvSelfRMS'][t]:.3e}",
                "globRMS": f"{metric_manager._errors['tvGlobalRMS'][t]:.3e}",
            }
        )


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    ensure_metrics_registered()
    cc_data, cellular_complex = load_data(cfg.dataset)
    output_root = (Path.cwd() / "outputs" / cfg.dataset.dataset_name / "dynamic_regret").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    case_plan = _resolve_case_plan(cfg)
    case_ids = {case_def["id"] for case_def in case_plan}
    comparison_cases = []

    global_param_history = []
    case_metric_managers = {}

    global_case_def = next((case_def for case_def in case_plan if case_def["id"] == "global"), deepcopy(DEFAULT_CASE_DEFS["global"]))
    global_enabled = "global" in case_ids
    global_case_name = global_case_def.get("name", DEFAULT_CASE_DEFS["global"]["name"])
    algorithm_param = _get_ccvar_algorithm_param(cfg)
    global_ccvar = CCVAR(algorithmParam=algorithm_param, cellularComplex=cellular_complex)
    T_global = min(cc_data[dim].shape[1] for dim in cc_data)
    if global_enabled:
        global_mm, global_output = _init_case_metric_manager(output_root, global_case_name, T=T_global)
    else:
        global_mm, global_output = None, None

    global_progress = tqdm(range(T_global), desc=global_case_name)
    for t in global_progress:
        input_data = {dim: cc_data[dim][:, t] for dim in cc_data}
        global_ccvar.update(inputData=input_data)
        global_vector = _flatten_theta_dict(global_ccvar._theta)
        global_param_history.append(global_vector.copy())
        if global_mm is not None:
            snapshot = {0: global_vector}
            global_mm.step_calculation(
                i=t,
                prediction=snapshot,
                groundTruth={"global_vector": global_vector},
                verbose=False,
            )
            global_progress.set_postfix(
                {
                    "selfD": f"{global_mm._errors['tvSelfDisagreement'][t]:.3e}",
                    "globD": f"{global_mm._errors['tvGlobalDisagreement'][t]:.3e}",
                    "selfRMS": f"{global_mm._errors['tvSelfRMS'][t]:.3e}",
                    "globRMS": f"{global_mm._errors['tvGlobalRMS'][t]:.3e}",
                }
            )
    if global_mm is not None and global_output is not None:
        _save_case_metrics(global_mm, global_output)

    clusters = instantiate(config=cfg.clustering, cellularComplex=cellular_complex)
    pure_local_state_cache = None
    for case_def in case_plan:
        if case_def["id"] == "global":
            continue

        case_name = case_def.get("name", case_def["id"])
        case_label = case_def.get("label", case_name)
        runner = case_def.get("runner", "")

        if runner == "pure_local_direct":
            if pure_local_state_cache is None:
                pure_local_state_cache = _create_pure_local_ccvar_states(
                    cfg=cfg,
                    cc_data=cc_data,
                    clusters=clusters,
                )
            pure_local_states, T_pure_local = pure_local_state_cache
            curr_mm, curr_output = _init_case_metric_manager(
                output_root, case_name, T=T_pure_local
            )
            _run_pure_local_ccvar_case(
                case_name=case_name,
                state_by_cluster=pure_local_states,
                T=T_pure_local,
                metric_manager=curr_mm,
                global_param_history=global_param_history,
            )
        elif runner == "distributed":
            curr_agents, _, T_case = create_cluster_agents(
                cfg=deepcopy(cfg),
                cc_data=cc_data,
                clusters=clusters,
                force_in_equals_out=bool(case_def.get("force_in_equals_out", False)),
                protocol_overrides=case_def.get("protocol_overrides", {}),
            )
            curr_mm, curr_output = _init_case_metric_manager(output_root, case_name, T=T_case)
            run_case(
                case_name=case_name,
                agent_list=curr_agents,
                T=T_case,
                consensus_mode=case_def.get("consensus_mode", "gated"),
                on_step_end=_create_callback(metric_manager=curr_mm, global_param_history=global_param_history),
            )
        else:
            continue

        _save_case_metrics(curr_mm, curr_output)
        case_metric_managers[case_name] = curr_mm
        comparison_cases.append((case_name, case_label))

    _save_last10pct_tables(
        output_root=output_root,
        case_metric_managers=case_metric_managers,
        comparison_cases=comparison_cases,
        cfg=cfg,
    )
    _save_metric_comparison_plots(
        output_root=output_root,
        case_metric_managers=case_metric_managers,
        comparison_cases=comparison_cases,
        cfg=cfg,
    )

    print(f"Dynamic regret runs finished. Disagreement metrics and comparison artifacts saved under: {output_root}")


if __name__ == "__main__":
    main()
