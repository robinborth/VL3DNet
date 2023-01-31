import pathlib

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config() -> DictConfig | ListConfig:
    """Loads all the configuration files from the conf folder

    Returns:
        OmegaConf: The combined configuration stored in a OmegaConf object.
    """
    path = pathlib.Path(__file__).parent.parent.resolve()

    base = OmegaConf.load(f"{path}/conf/static.yaml")
    base.root_path = path
    return base


static_cfg = load_config()
