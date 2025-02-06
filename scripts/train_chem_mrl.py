import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from chem_mrl.schemas import BaseConfig, ChemMRLConfig, WandbConfig
from chem_mrl.trainers import ChemMRLTrainer, WandBTrainerExecutor

cs = ConfigStore.instance()
cs.store(name="base_config_schema", node=BaseConfig)
cs.store(name="wandb_schema", node=WandbConfig)
cs.store(group="model", name="chem_mrl_schema", node=ChemMRLConfig)


@hydra.main(
    config_path="../chem_mrl/conf",
    config_name="chem_mrl_config",
    version_base="1.2",
)
def main(_cfg: DictConfig):
    cfg = OmegaConf.to_object(_cfg)
    assert isinstance(cfg, BaseConfig)
    assert isinstance(cfg.wandb, WandbConfig)
    assert isinstance(cfg.model, ChemMRLConfig)
    trainer = ChemMRLTrainer(cfg)
    executor = WandBTrainerExecutor(trainer)
    executor.execute()


if __name__ == "__main__":
    main()
