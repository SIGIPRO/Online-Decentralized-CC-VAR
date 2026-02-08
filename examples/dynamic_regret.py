from copy import deepcopy
from pathlib import Path

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


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    ensure_metrics_registered()
    cc_data, cellular_complex = load_data(cfg.dataset)
    output_root = (Path.cwd() / "outputs" / cfg.dataset.dataset_name / "dynamic_regret").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    global_param_history = []

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
            }
        )
    _save_case_metrics(global_mm, global_output)

    clusters_local = instantiate(config=cfg.clustering, cellularComplex=cellular_complex)
    local_agents, _, T_local = create_cluster_agents(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters_local,
    )
    local_mm, local_output = _init_case_metric_manager(output_root, "Local CC-VAR (No Consensus)", T=T_local)

    def _local_step_callback(t, agent_list, progress_bar):
        if not global_param_history:
            return
        snapshot = _snapshot_case_params(agent_list)
        ref_index = min(t, len(global_param_history) - 1)
        local_mm.step_calculation(
            i=t,
            prediction=snapshot,
            groundTruth={"global_vector": global_param_history[ref_index]},
            verbose=False,
        )
        progress_bar.set_postfix(
            {
                "selfD": f"{local_mm._errors['tvSelfDisagreement'][t]:.3e}",
                "globD": f"{local_mm._errors['tvGlobalDisagreement'][t]:.3e}",
            }
        )

    run_case(
        case_name="Local CC-VAR (No Consensus)",
        agent_list=local_agents,
        T=T_local,
        consensus_mode="off",
        on_step_end=_local_step_callback,
    )
    _save_case_metrics(local_mm, local_output)

    clusters_current = instantiate(config=cfg.clustering, cellularComplex=cellular_complex)
    current_agents, _, T_current = create_cluster_agents(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters_current,
    )
    current_mm, current_output = _init_case_metric_manager(output_root, "Current Case (Gated Consensus)", T=T_current)

    def _current_step_callback(t, agent_list, progress_bar):
        if not global_param_history:
            return
        snapshot = _snapshot_case_params(agent_list)
        ref_index = min(t, len(global_param_history) - 1)
        current_mm.step_calculation(
            i=t,
            prediction=snapshot,
            groundTruth={"global_vector": global_param_history[ref_index]},
            verbose=False,
        )
        progress_bar.set_postfix(
            {
                "selfD": f"{current_mm._errors['tvSelfDisagreement'][t]:.3e}",
                "globD": f"{current_mm._errors['tvGlobalDisagreement'][t]:.3e}",
            }
        )

    run_case(
        case_name="Current Case (Gated Consensus)",
        agent_list=current_agents,
        T=T_current,
        consensus_mode="gated",
        on_step_end=_current_step_callback,
    )
    _save_case_metrics(current_mm, current_output)

    print(f"All three cases finished. Disagreement metrics saved under: {output_root}")


if __name__ == "__main__":
    main()
