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

import tempfile
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from chem_mrl.schemas import BaseConfig

from .BaseTrainer import BoundTrainerType

BoundTrainerExecutorType = TypeVar("BoundTrainerExecutorType", bound="_BaseTrainerExecutor")


class _BaseTrainerExecutor(ABC, Generic[BoundTrainerType]):
    """Base abstract executor class.
    Executors are used to execute a trainer with additional functionality.
    For example, an executor can be used to execute a trainer within a context manager.
    """

    def __init__(self, trainer: BoundTrainerType):
        self.__trainer = trainer

    @property
    def trainer(self) -> BoundTrainerType:
        return self.__trainer

    @property
    def config(self) -> BaseConfig:
        return self.__trainer.config

    @abstractmethod
    def execute(self, **kwargs: Any) -> float | None:
        raise NotImplementedError


class TempDirTrainerExecutor(_BaseTrainerExecutor[BoundTrainerType]):
    """
    Executor that runs the trainer within a temporary directory.
    All files stored during execution are removed once the program exits.
    """

    def __init__(self, trainer: BoundTrainerType):
        super().__init__(trainer)
        self._temp_dir = tempfile.TemporaryDirectory()
        # overwrite both since training_args.output_dir is overwritten to include a checkpoints dir
        self.trainer._training_args.output_dir = self._temp_dir.name
        self.trainer._root_output_dir = self._temp_dir.name
        self.trainer._is_testing = True

    def execute(self, **kwargs: Any) -> float | None:
        """
        Execute the trainer within the temporary directory context.
        """
        return self.trainer.train(**kwargs)

    def cleanup(self) -> None:
        """
        Cleanup temporary directory.
        """
        self._temp_dir.cleanup()

    def __del__(self):
        """
        Ensure cleanup occurs when the instance is deleted.
        """
        self.cleanup()
