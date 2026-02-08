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
    "tvGlobalRMS",
    "rollingGlobalRMS",
    "tvGlobalRMSRelative",
    "rollingGlobalRMSRelative",
]

COMPARISON_CASES = [
    ("Pure Local CC-VAR (Nin Open, No Comm)", "Pure Local"),
    ("Parameter-Only CC-VAR", "Parameter Only"),
    ("Parameter + Dataset CC-VAR (Gated)", "Parameter + Dataset"),
]


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


def _get_run_parameters(cfg):
    c_data = cfg.protocol.get("C_data", "NA")
    c_param = cfg.protocol.get("C_param", "NA")
    mixing_eta = cfg.mixing.get("eta", {})
    c_val = mixing_eta.get("c", "NA")
    return c_data, c_param, c_val


def _parameter_suffix(c_data, c_param, c_val):
    return f"c_data_{c_data}_c_param_{c_param}_c_{c_val}"


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


def _save_last10pct_tables(output_root: Path, case_metric_managers, cfg):
    c_data, c_param, c_val = _get_run_parameters(cfg)
    suffix = _parameter_suffix(c_data, c_param, c_val)
    table_dir = output_root / "comparison_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    tv_metrics = [m for m in DISAGREEMENT_METRICS if m.startswith("tv")]
    rows = []
    for metric_name in tv_metrics:
        vals = []
        for case_name, _ in COMPARISON_CASES:
            case_mm = case_metric_managers[case_name]
            vals.append(_mean_last_fraction(case_mm._errors.get(metric_name, []), fraction=0.1))
        rows.append((metric_name, vals))

    md_lines = [
        f"# Disagreement (Last 10%)",
        f"",
        f"Parameters: `C_data={c_data}`, `C_param={c_param}`, `c={c_val}`",
        f"",
        "| Metric | Pure Local | Parameter Only | Parameter + Dataset |",
        "|---|---:|---:|---:|",
    ]
    for metric_name, vals in rows:
        formatted = [("nan" if not np.isfinite(v) else f"{v:.6e}") for v in vals]
        md_lines.append(f"| {metric_name} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")

    md_path = table_dir / f"disagreement_{suffix}.md"
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("\n".join(md_lines) + "\n")

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Last 10\\% mean of TV disagreement metrics ($C_{{data}}={c_data}$, $C_{{param}}={c_param}$, $c={c_val}$).}}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Metric & Pure Local & Parameter Only & Parameter + Dataset \\\\",
        "\\hline",
    ]
    for metric_name, vals in rows:
        formatted = [("nan" if not np.isfinite(v) else f"{v:.6e}") for v in vals]
        tex_lines.append(
            f"{metric_name} & {formatted[0]} & {formatted[1]} & {formatted[2]} \\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    tex_path = table_dir / f"disagreement_{suffix}.tex"
    with open(tex_path, "w", encoding="utf-8") as file:
        file.write("\n".join(tex_lines) + "\n")


def _save_metric_comparison_plots(output_root: Path, case_metric_managers, cfg):
    c_data, c_param, c_val = _get_run_parameters(cfg)
    suffix = _parameter_suffix(c_data, c_param, c_val)
    figure_dir = output_root / "comparison_plots"
    figure_dir.mkdir(parents=True, exist_ok=True)

    annotation_text = f"C_data={c_data}, C_param={c_param}, c={c_val}"

    for metric_name in DISAGREEMENT_METRICS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for case_name, label in COMPARISON_CASES:
            series = np.asarray(case_metric_managers[case_name]._errors.get(metric_name, []), dtype=float).reshape(-1)
            ax.plot(series, linewidth=1.7, label=label)

        ax.set_title(metric_name)
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


def _create_pure_local_ccvar_states(cc_data, clusters):
    T = None
    state_by_cluster = {}
    model_cfg_path = Path(__file__).resolve().parents[1] / "conf" / "model" / "ccvar.yaml"
    model_cfg = OmegaConf.load(model_cfg_path)
    algorithm_param = OmegaConf.to_container(model_cfg.algorithmParam, resolve=True)

    for cluster_head in sorted(clusters.Nin.keys()):
        nin_idx = {}
        processed_data = {}
        for dim in sorted(cc_data.keys()):
            nin_idx[dim] = sorted(list(clusters.Nin[cluster_head].get(dim, [])))
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

    global_param_history = []
    case_metric_managers = {}

    model_cfg_path = Path(__file__).resolve().parents[1] / "conf" / "model" / "ccvar.yaml"
    model_cfg = OmegaConf.load(model_cfg_path)
    algorithm_param = OmegaConf.to_container(model_cfg.algorithmParam, resolve=True)
    global_ccvar = CCVAR(algorithmParam=algorithm_param, cellularComplex=cellular_complex)
    T_global = min(cc_data[dim].shape[1] for dim in cc_data)
    global_mm, global_output = _init_case_metric_manager(output_root, "Global CC-VAR", T=T_global)

    global_progress = tqdm(range(T_global), desc="Global CC-VAR (Direct CCVAR)")
    for t in global_progress:
        input_data = {dim: cc_data[dim][:, t] for dim in cc_data}
        global_ccvar.update(inputData=input_data)
        global_vector = _flatten_theta_dict(global_ccvar._theta)
        snapshot = {0: global_vector}
        global_param_history.append(global_vector.copy())
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
    _save_case_metrics(global_mm, global_output)

    clusters = instantiate(config=cfg.clustering, cellularComplex=cellular_complex)

    pure_local_states, T_pure_local = _create_pure_local_ccvar_states(
        cc_data=cc_data,
        clusters=clusters,
    )
    pure_local_mm, pure_local_output = _init_case_metric_manager(
        output_root, "Pure Local CC-VAR (Nin Open, No Comm)", T=T_pure_local
    )
    _run_pure_local_ccvar_case(
        case_name="Pure Local CC-VAR (Nin Open, No Comm)",
        state_by_cluster=pure_local_states,
        T=T_pure_local,
        metric_manager=pure_local_mm,
        global_param_history=global_param_history,
    )
    _save_case_metrics(pure_local_mm, pure_local_output)
    case_metric_managers["Pure Local CC-VAR (Nin Open, No Comm)"] = pure_local_mm

    param_only_agents, _, T_param_only = create_cluster_agents(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters,
        force_in_equals_out=True,
        protocol_overrides={"C_data": int(1e9)},
    )
    param_only_mm, param_only_output = _init_case_metric_manager(output_root, "Parameter-Only CC-VAR", T=T_param_only)
    run_case(
        case_name="Parameter-Only CC-VAR",
        agent_list=param_only_agents,
        T=T_param_only,
        consensus_mode="gated",
        on_step_end=_create_callback(metric_manager=param_only_mm, global_param_history=global_param_history),
    )
    _save_case_metrics(param_only_mm, param_only_output)
    case_metric_managers["Parameter-Only CC-VAR"] = param_only_mm

    current_agents, _, T_current = create_cluster_agents(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters,
    )
    current_mm, current_output = _init_case_metric_manager(output_root, "Parameter + Dataset CC-VAR (Gated)", T=T_current)
    run_case(
        case_name="Parameter + Dataset CC-VAR (Gated)",
        agent_list=current_agents,
        T=T_current,
        consensus_mode="gated",
        on_step_end=_create_callback(metric_manager=current_mm, global_param_history=global_param_history),
    )
    _save_case_metrics(current_mm, current_output)
    case_metric_managers["Parameter + Dataset CC-VAR (Gated)"] = current_mm

    _save_last10pct_tables(output_root=output_root, case_metric_managers=case_metric_managers, cfg=cfg)
    _save_metric_comparison_plots(output_root=output_root, case_metric_managers=case_metric_managers, cfg=cfg)

    print(f"All four cases finished. Disagreement metrics and comparison artifacts saved under: {output_root}")


if __name__ == "__main__":
    main()
