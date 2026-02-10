from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig

from examples.utils.data_utils import load_data


@hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    _, cellular_complex = load_data(cfg.dataset)
    clusters = instantiate(config=cfg.clustering, cellularComplex=cellular_complex)
    clusters.plot_clusters()


if __name__ == "__main__":
    main()
