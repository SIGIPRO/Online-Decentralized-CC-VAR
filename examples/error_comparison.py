from copy import deepcopy
from pathlib import Path

import numpy as np
from tqdm import tqdm  # type: ignore[import-untyped]

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ccvar import CCVAR
from cellexp_util.registry.metric_registry import ensure_metrics_registered

from src.core import BaseAgent
from examples.utils.data_utils import load_data
from examples.utils.clustering_utils import create_cluster_agents
from examples.utils.metric_utils import (
    evaluate_pending_predictions,
    init_metric_managers,
    save_metric_plots,
)


def _slugify_case_name(case_name: str) -> str:
    return case_name.lower().replace(" ", "_").replace("(", "").replace(")", "")


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


def _create_pure_local_agents(cfg, cc_data, clusters, ccvar_model_cfg):
    T = None
    agent_list = {}
    cluster_out_global_idx = {}

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

        protocol_cfg = deepcopy(cfg.protocol)
        protocol_cfg.C_data = int(1e9)
        protocol_cfg.C_param = int(1e9)
        protocol = instantiate(protocol_cfg)

        model = instantiate(ccvar_model_cfg, cellularComplex=local_complex)
        data = instantiate(
            cfg.ccdata,
            data=processed_data,
            interface={},
            Nout={},
            Nex={},
            global_idx=nin_idx,
        )
        mixing = instantiate(cfg.mixing, weights={"self": 1.0})
        imputer = instantiate(cfg.imputer)

        agent = BaseAgent(
            cluster_id=cluster_head,
            model=model,
            data=data,
            protocol=protocol,
            mix=mixing,
            imputer=imputer,
            neighbors=set(),
            cellularComplex=local_complex,
        )
        agent_list[cluster_head] = agent
        cluster_out_global_idx[cluster_head] = {
            dim: np.asarray(nin_idx[dim], dtype=int) for dim in nin_idx
        }

        if T is None or data._T_total < T:
            T = data._T_total

    return agent_list, cluster_out_global_idx, T


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


def _run_global_ccvar_forecast_case(case_name: str, case_output_dir: Path, cc_data, cellular_complex):
    model_cfg_path = Path(__file__).resolve().parents[1] / "conf" / "model" / "ccvar.yaml"
    model_cfg = OmegaConf.load(model_cfg_path)
    algorithm_param = OmegaConf.to_container(model_cfg.algorithmParam, resolve=True)
    global_ccvar = CCVAR(algorithmParam=algorithm_param, cellularComplex=cellular_complex)

    T = min(cc_data[dim].shape[1] for dim in cc_data)
    metrics, output_dirs = _init_case_metrics(case_output_dir=case_output_dir, cc_data=cc_data, T_eval=T - 1)
    global_cluster_out_idx = {
        0: {dim: np.arange(cc_data[dim].shape[0], dtype=int) for dim in cc_data}
    }

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
        pending_prediction = {0: global_ccvar.forecast(steps=1)}

    _save_case_metrics(metrics=metrics, output_dirs=output_dirs)


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    ensure_metrics_registered()

    cc_data, cellular_complex = load_data(cfg.dataset)
    output_root = (Path.cwd() / "outputs" / cfg.dataset.dataset_name / "error_comparison").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    clusters = instantiate(config=cfg.clustering, cellularComplex=cellular_complex)

    case_name = "Global CC-VAR (Direct CCVAR)"
    _run_global_ccvar_forecast_case(
        case_name=case_name,
        case_output_dir=output_root / _slugify_case_name(case_name),
        cc_data=cc_data,
        cellular_complex=cellular_complex,
    )

    model_cfg_path = Path(__file__).resolve().parents[1] / "conf" / "model" / "ccvar.yaml"
    ccvar_model_cfg = OmegaConf.load(model_cfg_path)
    pure_local_agents, pure_local_out_idx, T_pure_local = _create_pure_local_agents(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters,
        ccvar_model_cfg=ccvar_model_cfg,
    )
    case_name = "Pure Local CC-VAR (Nin Open, No Comm)"
    _run_distributed_forecast_case(
        case_name=case_name,
        case_output_dir=output_root / _slugify_case_name(case_name),
        agent_list=pure_local_agents,
        cluster_out_global_idx=pure_local_out_idx,
        cc_data=cc_data,
        T=T_pure_local,
        consensus_mode="off",
    )

    param_only_agents, param_only_out_idx, T_param_only = create_cluster_agents(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters,
        force_in_equals_out=True,
        protocol_overrides={"C_data": int(1e9)},
    )
    case_name = "Parameter-Only CC-VAR"
    _run_distributed_forecast_case(
        case_name=case_name,
        case_output_dir=output_root / _slugify_case_name(case_name),
        agent_list=param_only_agents,
        cluster_out_global_idx=param_only_out_idx,
        cc_data=cc_data,
        T=T_param_only,
        consensus_mode="gated",
    )

    current_agents, current_out_idx, T_current = create_cluster_agents(
        cfg=deepcopy(cfg),
        cc_data=cc_data,
        clusters=clusters,
    )
    case_name = "Parameter + Dataset CC-VAR (Gated)"
    _run_distributed_forecast_case(
        case_name=case_name,
        case_output_dir=output_root / _slugify_case_name(case_name),
        agent_list=current_agents,
        cluster_out_global_idx=current_out_idx,
        cc_data=cc_data,
        T=T_current,
        consensus_mode="gated",
    )

    print(f"Forecast error comparison completed. Outputs saved under: {output_root}")


if __name__ == "__main__":
    main()
