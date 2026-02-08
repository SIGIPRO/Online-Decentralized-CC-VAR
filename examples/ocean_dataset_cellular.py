import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from cellexp_util.registry.metric_registry import ensure_metrics_registered
from tqdm import tqdm # type: ignore[import-untyped]
from examples.utils.metric_utils import (
    evaluate_pending_predictions,
    init_metric_managers,
    save_metric_plots,
)
from examples.utils.data_utils import load_data, get_output_dir
from examples.utils.clustering_utils import create_cluster_agents




""" NOTE: 
1. The divergence can be arising from not normalizing the features as in the CC-VAR paper. This is done already. Did not work.
 a. More severe error in implementation it seems. 
i. Taking the c very small, like 5e-6, solved the divergence issue but one should look at the implementation also. Partial solution +-.

2. KGTMixing is also causing divergence. Hyperparameters should be optimized.
"""

""" ISSUES:  
1. For some reason, CC-VAR explodes even with local steps. This issue is partially solved ++ (The reason maybe really the learning parameter. In the previous case it was automatically updating itself. Moving to divergence of KGTMixing.)
2. For each agent edge signals are full of data which should not be the case. This issue solved. ++
3. CC-VAR is wrongly used. The problem is that it takes all of the elements of the agent which should not be the case. Implemented. Look at get_gradient method of the CCVARPartial to make it complete. Also add CCVARPartialModel for completeness. ++
4. LabelPropagator was also implemented. ++
5. Metric manager is not implemented yet. ++ (Codex implementation)
6. Look at the dynamic regret alongside of MSE. --
7. Check the error when KGTMixing is applied. --
"""

@hydra.main(version_base=None, config_path='../conf', config_name='config.yaml')
def main(cfg: DictConfig):

    outputDir = get_output_dir(cfg.dataset.dataset_name)
    ensure_metrics_registered()

    cc_data, cellularComplex = load_data(cfg.dataset)



    clusters = instantiate(config=cfg.clustering, cellularComplex=cellularComplex)

    agent_list, cluster_out_global_idx, T = create_cluster_agents(
        cfg=cfg,
        cc_data=cc_data,
        clusters=clusters,
    )
    

    T_eval = T - 1
    metrics, output_dirs = init_metric_managers(
        cc_data=cc_data,
        output_dir_fn=outputDir,
        T_eval=T_eval,
    )

    pending_prediction_by_cluster = None

    progress_bar = tqdm(range(0, T))
    for t in progress_bar:

        for cluster_head in agent_list:
            agent_list[cluster_head].iterate_data(t)
            agent_list[cluster_head].send_data(t)
            data_box = agent_list[cluster_head].outbox['data']
            for cluster_id in data_box:
               agent_list[cluster_id].push_to_inbox(cluster_head, data_box[cluster_id], "data")

        for cluster_head in agent_list:
            agent_list[cluster_head].receive_data()

        # Evaluate one-step-ahead predictions generated at previous step against current data x_t.
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
            params_box = agent_list[cluster_head].outbox['params']
            for cluster_id in params_box:
                agent_list[cluster_id].push_to_inbox(cluster_head, params_box[cluster_id], "params")

            # print(params_box)
            # import pdb; pdb.set_trace()

        prediction_by_cluster = dict()
        has_fresh_neighbor_params = dict()
        for cluster_head in agent_list:
            has_fresh_neighbor_params[cluster_head] = agent_list[cluster_head].receive_params()
        for cluster_head in agent_list:
            if has_fresh_neighbor_params[cluster_head]:
                agent_list[cluster_head].do_consensus()
        for cluster_head in agent_list:
            prediction_by_cluster[cluster_head] = agent_list[cluster_head].estimate(input_data=None, steps=1)
        pending_prediction_by_cluster = prediction_by_cluster

    for dim in cc_data:
        metrics[dim].save_single(n=0)
        metrics[dim].save_full(n=1)
    save_metric_plots(metrics=metrics, output_dirs=output_dirs)

if __name__ == "__main__":
    main()
