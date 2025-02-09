from __future__ import annotations

from hydra.core.config_store import ConfigStore

from . import Enums
from .BaseConfig import BaseConfig, WandbConfig
from .ChemMRLConfig import ChemMRLConfig
from .ClassifierConfig import ClassifierConfig
from .LatentAttentionConfig import LatentAttentionConfig


def register_chem_mrl_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config_schema", node=BaseConfig)
    cs.store(name="wandb_schema", node=WandbConfig)
    cs.store(group="model", name="chem_mrl_schema", node=ChemMRLConfig)


def register_classifier_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config_schema", node=BaseConfig)
    cs.store(name="wandb_schema", node=WandbConfig)
    cs.store(group="model", name="classifier_schema", node=ClassifierConfig)


__all__ = [
    "Enums",
    "BaseConfig",
    "WandbConfig",
    "ChemMRLConfig",
    "LatentAttentionConfig",
    "register_chem_mrl_configs",
    "ClassifierConfig",
    "register_classifier_configs",
]
