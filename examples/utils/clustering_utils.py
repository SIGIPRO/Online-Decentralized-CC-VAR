from copy import deepcopy

import numpy as np
from hydra.utils import instantiate
from tqdm import tqdm  # type: ignore[import-untyped]

from src.core import BaseAgent
from src.implementations.agent import SnapshotAgent
from examples.utils.data_utils import build_partial_indices


def _slice_cellular_complex_by_positions(cellular_complex, positions_by_dim):
    sliced = {}
    if 1 in cellular_complex:
        row_pos = positions_by_dim.get(0, list(range(cellular_complex[1].shape[0])))
        col_pos = positions_by_dim.get(1, list(range(cellular_complex[1].shape[1])))
        sliced[1] = cellular_complex[1][np.ix_(row_pos, col_pos)]
    if 2 in cellular_complex:
        row_pos = positions_by_dim.get(1, list(range(cellular_complex[2].shape[0])))
        col_pos = positions_by_dim.get(2, list(range(cellular_complex[2].shape[1])))
        sliced[2] = cellular_complex[2][np.ix_(row_pos, col_pos)]
    return sliced


def _filter_neighbor_dim_map_by_global_idx(neighbor_dim_map, allowed_global_idx_by_dim):
    filtered = {}
    for neighbor_id, dim_map in neighbor_dim_map.items():
        kept_dim_map = {}
        for dim, idx_list in dim_map.items():
            allowed = set(allowed_global_idx_by_dim.get(dim, []))
            kept = [gidx for gidx in idx_list if gidx in allowed]
            if kept:
                kept_dim_map[dim] = kept
        if kept_dim_map:
            filtered[neighbor_id] = kept_dim_map
    return filtered


def create_cluster_agents(
    cfg,
    cc_data,
    clusters,
    force_in_equals_out=False,
    protocol_overrides=None,
    disable_neighbors=False,
    show_progress=True,
    snapshot_agent=False,
    snapshot_data_stream=None,
):
    T = None
    agent_list = {}
    cluster_out_global_idx = {}
    degree_by_cluster = {
        cluster_id: len(neighbors)
        for cluster_id, neighbors in clusters.agent_graph.items()
    }

    cluster_heads = list(clusters.clustered_complexes.keys())
    iterator = cluster_heads
    if show_progress:
        iterator = tqdm(cluster_heads, desc="Creating agents", leave=False)

    if snapshot_agent:
        if snapshot_data_stream is None:
            raise ValueError("snapshot_data_stream is required when snapshot_agent=True.")
        SnapshotAgent.set_shared_data_stream(snapshot_data_stream)
        T = getattr(snapshot_data_stream, "_T_total", None)

    for cluster_head in iterator:
        global_idx = deepcopy(clusters.global_to_local_idx[cluster_head])
        model_cellular_complex = clusters.clustered_complexes[cluster_head]

        if snapshot_agent:
            processed_data = {}
            interface = {}
            Nout = {}
            Nex = {}
        else:
            processed_data = {}
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

        protocol_cfg = deepcopy(cfg.protocol)
        if protocol_overrides:
            for key, value in protocol_overrides.items():
                protocol_cfg[key] = value
        protocol = instantiate(protocol_cfg)

        model_cfg = deepcopy(cfg.model)
        in_idx, out_idx = build_partial_indices(
            global_idx=global_idx,
            nin_for_cluster=clusters.Nin[cluster_head],
        )
        source_out_positions = deepcopy(out_idx)
        if force_in_equals_out and not snapshot_agent:
            compact_global_idx = {}
            compact_processed_data = {}
            compact_pos_idx = {}
            for dim in global_idx:
                selected_pos = source_out_positions.get(dim, [])
                compact_global_idx[dim] = [global_idx[dim][i] for i in selected_pos]
                if dim in processed_data.keys():
                    compact_processed_data[dim] = processed_data[dim][selected_pos, :]
                compact_pos_idx[dim] = list(range(len(selected_pos)))

            global_idx = compact_global_idx
            processed_data = compact_processed_data
            in_idx = compact_pos_idx
            out_idx = compact_pos_idx
            model_cellular_complex = _slice_cellular_complex_by_positions(
                cellular_complex=clusters.clustered_complexes[cluster_head],
                positions_by_dim=source_out_positions,
            )

            interface = _filter_neighbor_dim_map_by_global_idx(
                neighbor_dim_map=interface,
                allowed_global_idx_by_dim=global_idx,
            )
            Nout = _filter_neighbor_dim_map_by_global_idx(
                neighbor_dim_map=Nout,
                allowed_global_idx_by_dim=global_idx,
            )
            Nex = _filter_neighbor_dim_map_by_global_idx(
                neighbor_dim_map=Nex,
                allowed_global_idx_by_dim=global_idx,
            )

        cluster_out_global_idx[cluster_head] = {}
        for dim in global_idx:
            if force_in_equals_out:
                cluster_out_global_idx[cluster_head][dim] = np.asarray(global_idx[dim], dtype=int)
            else:
                map_positions = out_idx[dim]
                cluster_out_global_idx[cluster_head][dim] = np.asarray(
                    [global_idx[dim][i] for i in map_positions],
                    dtype=int,
                )

        model_cfg.algorithmParam.in_idx = in_idx
        model_cfg.algorithmParam.out_idx = out_idx

        ccvarmodel = instantiate(
            model_cfg,
            cellularComplex=model_cellular_complex,
        )
        neighbors = [] if disable_neighbors else list(clusters.agent_graph[cluster_head])
        weights = {}
        row_sum = 0.0
        for cluster_id in neighbors:
            mh_weight = 1.0 / (
                1.0
                + max(
                    degree_by_cluster.get(cluster_head, 0),
                    degree_by_cluster.get(cluster_id, 0),
                )
            )
            weights[cluster_id] = mh_weight
            row_sum += mh_weight
        # MH self-weight closes the row to 1 for a row-stochastic matrix.
        weights["self"] = 1.0 - row_sum

        mixing = instantiate(cfg.mixing, weights=weights)
        if snapshot_agent:
            currAgent = SnapshotAgent(
                cluster_id=cluster_head,
                model=ccvarmodel,
                protocol=protocol,
                mix=mixing,
                neighbors=set(neighbors),
            )
        else:
            ccdata = instantiate(
                cfg.ccdata,
                data=processed_data,
                interface=interface,
                Nout=Nout,
                Nex=Nex,
                global_idx=global_idx,
            )
            imputer = instantiate(cfg.imputer)

            currAgent = BaseAgent(
                cluster_id=cluster_head,
                model=ccvarmodel,
                data=ccdata,
                protocol=protocol,
                mix=mixing,
                imputer=imputer,
                neighbors=set(neighbors),
                cellularComplex=model_cellular_complex,
            )
        agent_list[cluster_head] = currAgent

        if not snapshot_agent:
            if T is None or currAgent._data._T_total < T:
                T = currAgent._data._T_total

    return agent_list, cluster_out_global_idx, T
