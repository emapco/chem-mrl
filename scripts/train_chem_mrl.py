import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from chem_mrl.schemas import (
    BaseConfig,
    ChemMRLConfig,
    ClassifierConfig,
    register_chem_mrl_configs,
    register_classifier_configs,
)
from chem_mrl.trainers import ChemMRLTrainer, ClassifierTrainer

logger = logging.getLogger(__name__)
register_chem_mrl_configs()
register_classifier_configs()


def verify_model_config(cfg: BaseConfig):
    """Verify that the model config matches the expected type."""
    if isinstance(cfg.model, ChemMRLConfig):
        return "chem_mrl"
    elif isinstance(cfg.model, ClassifierConfig):
        return "classifier"
    else:
        raise ValueError(
            f"Unknown model type: {type(cfg.model)}"
            f"Expected one of: chem_mrl, chem_2d_mrl, classifier, dice_loss_classifier"
        )


@hydra.main(
    config_path="../chem_mrl/conf",
    config_name="base",
    version_base="1.2",
)
def main(_cfg: DictConfig):
    cfg = OmegaConf.to_object(_cfg)
    assert isinstance(cfg, BaseConfig)
    model_type = verify_model_config(cfg)
    trainer = ChemMRLTrainer(cfg) if model_type == "chem_mrl" else ClassifierTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
