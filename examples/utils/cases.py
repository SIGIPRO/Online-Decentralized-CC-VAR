from copy import deepcopy

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm  # type: ignore[import-untyped]

from src.core import BaseAgent


def _build_global_ccvar_model_cfg(cfg: DictConfig):
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg._target_ = "src.implementations.models.ccvar.CCVARModel"
    algorithm_param = deepcopy(model_cfg.get("algorithmParam", {}))
    algorithm_param.pop("in_idx", None)
    algorithm_param.pop("out_idx", None)
    model_cfg.algorithmParam = algorithm_param
    return model_cfg


def create_global_agent(cfg: DictConfig, cc_data, cellular_complex):
    model_cfg = _build_global_ccvar_model_cfg(cfg)
    processed_data = {dim: data.copy() for dim, data in cc_data.items()}
    global_idx = {dim: list(range(processed_data[dim].shape[0])) for dim in processed_data}

    protocol = instantiate(cfg.protocol)
    model = instantiate(model_cfg, cellularComplex=cellular_complex)
    data = instantiate(
        cfg.ccdata,
        data=processed_data,
        interface={},
        Nout={},
        Nex={},
        global_idx=global_idx,
    )
    mixing = instantiate(cfg.mixing, weights={"self": 1.0})
    imputer = instantiate(cfg.imputer)

    agent = BaseAgent(
        cluster_id=0,
        model=model,
        data=data,
        protocol=protocol,
        mix=mixing,
        imputer=imputer,
        neighbors=set(),
        cellularComplex=cellular_complex,
    )
    return {0: agent}, data._T_total


def run_case(case_name: str, agent_list, T: int, consensus_mode: str, on_step_end=None):
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

        for cluster_head in agent_list:
            agent_list[cluster_head].estimate(input_data=None, steps=1)

        if on_step_end is not None:
            on_step_end(t=t, agent_list=agent_list, progress_bar=progress_bar)
