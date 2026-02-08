from copy import deepcopy

import numpy as np
from hydra.utils import instantiate

from src.core import BaseAgent
from examples.utils.data_utils import build_partial_indices


def create_cluster_agents(cfg, cc_data, clusters):
    T = None
    agent_list = {}
    cluster_out_global_idx = {}

    for cluster_head in clusters.clustered_complexes:
        processed_data = {}
        global_idx = clusters.global_to_local_idx[cluster_head]

        for dim in cc_data:
            processed_data[dim] = cc_data[dim][global_idx[dim], :]

        interface = {}
        for head_tuple in clusters.interface:
            try:
                idx_head = head_tuple.index(cluster_head)
            except ValueError:
                continue
            idx_neighbor = 1 - idx_head
            interface[head_tuple[idx_neighbor]] = clusters.interface[head_tuple]

        Nout = {}
        Nex = {}
        for head_tuple in clusters.Nout:
            if cluster_head not in head_tuple:
                continue
            if cluster_head == head_tuple[0]:
                Nout[head_tuple[1]] = clusters.Nout[head_tuple]
            elif cluster_head == head_tuple[1]:
                Nex[head_tuple[0]] = clusters.Nout[head_tuple]

        protocol = instantiate(cfg.protocol)

        model_cfg = deepcopy(cfg.model)
        in_idx, out_idx = build_partial_indices(
            global_idx=global_idx,
            nin_for_cluster=clusters.Nin[cluster_head],
        )
        cluster_out_global_idx[cluster_head] = {}
        for dim in global_idx:
            cluster_out_global_idx[cluster_head][dim] = np.asarray(
                [global_idx[dim][i] for i in out_idx[dim]],
                dtype=int,
            )

        model_cfg.algorithmParam.in_idx = in_idx
        model_cfg.algorithmParam.out_idx = out_idx

        ccvarmodel = instantiate(
            model_cfg,
            cellularComplex=clusters.clustered_complexes[cluster_head],
        )
        ccdata = instantiate(
            cfg.ccdata,
            data=processed_data,
            interface=interface,
            Nout=Nout,
            Nex=Nex,
            global_idx=global_idx,
        )

        weights = {}
        num_connected = len(clusters.agent_graph[cluster_head]) + 1
        weights["self"] = 1 / num_connected
        for cluster_id in list(clusters.agent_graph[cluster_head]):
            weights[cluster_id] = 1 / num_connected

        mixing = instantiate(cfg.mixing, weights=weights)
        imputer = instantiate(cfg.imputer)

        currAgent = BaseAgent(
            cluster_id=cluster_head,
            model=ccvarmodel,
            data=ccdata,
            protocol=protocol,
            mix=mixing,
            imputer=imputer,
            neighbors=clusters.agent_graph[cluster_head],
            cellularComplex=clusters.clustered_complexes[cluster_head],
        )
        agent_list[cluster_head] = currAgent

        if T is None or currAgent._data._T_total < T:
            T = currAgent._data._T_total

    return agent_list, cluster_out_global_idx, T
