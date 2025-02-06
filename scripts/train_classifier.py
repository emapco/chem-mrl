import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from chem_mrl.schemas import BaseConfig, ClassifierConfig, WandbConfig
from chem_mrl.trainers import ClassifierTrainer, WandBTrainerExecutor

cs = ConfigStore.instance()
cs.store(name="base_config_schema", node=BaseConfig)
cs.store(name="wandb_schema", node=WandbConfig)
cs.store(group="model", name="classifier_schema", node=ClassifierConfig)


@hydra.main(
    config_path="../chem_mrl/conf",
    config_name="classifier_config",
    version_base="1.2",
)
def main(_cfg: DictConfig):
    cfg = OmegaConf.to_object(_cfg)
    assert isinstance(cfg, BaseConfig)
    assert isinstance(cfg.wandb, WandbConfig)
    assert isinstance(cfg.model, ClassifierConfig)
    trainer = ClassifierTrainer(cfg)
    executor = WandBTrainerExecutor(trainer)
    executor.execute()


if __name__ == "__main__":
    main()
