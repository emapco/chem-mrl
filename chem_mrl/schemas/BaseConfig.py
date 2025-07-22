from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from .DatasetConfig import DatasetConfig

BoundConfigType = TypeVar("BoundConfigType", bound="BaseConfig")


@dataclass
class BaseConfig:
    # Hydra's structured config schema doesn't support
    # generics nor unions of containers (e.g. ChemMRLConfig)
    model: Any
    training_args: Any
    datasets: list[DatasetConfig]
    early_stopping_patience: int | None = None
    scale_learning_rate: bool = False
    use_normalized_weight_decay: bool = False
    asdict = asdict

    def __post_init__(self):
        # check types
        if not isinstance(self.datasets, list):
            raise TypeError("datasets must be a list")
        if not all(isinstance(dataset, DatasetConfig) for dataset in self.datasets):
            raise TypeError("all items in datasets must be DatasetConfig instances")
        if not isinstance(self.early_stopping_patience, int | None):
            raise TypeError("early_stopping_patience must be an integer or None")
        if not isinstance(self.scale_learning_rate, bool):
            raise TypeError("scale_learning_rate must be a boolean")
        if not isinstance(self.use_normalized_weight_decay, bool):
            raise TypeError("use_normalized_weight_decay must be a boolean")
        # check values
        if len(self.datasets) == 0:
            raise ValueError("at least one dataset config must be provided")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be greater than 0")

        datasets_with_train_dataset = [
            dataset for dataset in self.datasets if dataset.train_dataset is not None
        ]
        if len(datasets_with_train_dataset) == 0:
            raise ValueError("at least one dataset config must have a train_dataset")
