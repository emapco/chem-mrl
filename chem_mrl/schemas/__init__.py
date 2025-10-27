# Copyright 2025 Emmanuel Cortes. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from . import Enums
from .BaseConfig import BaseConfig
from .ChemMRLConfig import ChemMRLConfig
from .ClassifierConfig import ClassifierConfig
from .DatasetConfig import DatasetConfig, SplitConfig


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config_schema", node=BaseConfig)
    cs.store(group="model", name="chem_mrl_schema", node=ChemMRLConfig)
    cs.store(group="model", name="classifier_schema", node=ClassifierConfig)


__all__ = [
    "Enums",
    "BaseConfig",
    "ChemMRLConfig",
    "ClassifierConfig",
    "DatasetConfig",
    "SplitConfig",
    "register_configs",
]
