import tempfile
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

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
    def execute(self) -> float:
        raise NotImplementedError


class TempDirTrainerExecutor(_BaseTrainerExecutor[BoundTrainerType]):
    """
    Executor that runs the trainer within a temporary directory.
    All files stored during execution are removed once the program exits.
    """

    def __init__(self, trainer: BoundTrainerType):
        super().__init__(trainer)
        self._temp_dir = tempfile.TemporaryDirectory()
        self.trainer.model_save_dir = self._temp_dir.name

    def execute(self) -> float:
        """
        Execute the trainer within the temporary directory context.
        """
        return self.trainer.train()

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
