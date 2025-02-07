import hydra
from omegaconf import DictConfig, OmegaConf

from chem_mrl.schemas import (
    BaseConfig,
    ClassifierConfig,
    WandbConfig,
    register_classifier_configs,
)
from chem_mrl.trainers import ClassifierTrainer, WandBTrainerExecutor

register_classifier_configs()


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
