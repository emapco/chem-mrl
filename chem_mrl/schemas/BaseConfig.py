from dataclasses import asdict, dataclass
from typing import Any, TypeVar

BoundConfigType = TypeVar("BoundConfigType", bound="BaseConfig")


@dataclass
class BaseConfig:
    # Hydra's structured config schema doesn't support
    # generics nor unions of containers (e.g. ChemMRLConfig)
    model: Any
    training_args: Any
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str | None = None
    train_datasets_split: str = "train"
    val_datasets_split: str = "train"
    test_datasets_split: str = "train"
    smiles_a_column_name: str = "smiles_a"
    smiles_b_column_name: str | None = "smiles_b"
    label_column_name: str = "similarity"
    n_train_samples: int | None = None
    n_val_samples: int | None = None
    n_test_samples: int | None = None
    early_stopping_patience: int | None = None
    scale_learning_rate: bool = False
    use_normalized_weight_decay: bool = False
    asdict = asdict

    def __post_init__(self):
        # check types
        if not isinstance(self.train_dataset_path, str):
            raise TypeError("train_dataset_path must be a string")
        if not isinstance(self.val_dataset_path, str):
            raise TypeError("val_dataset_path must be a string")
        if not isinstance(self.test_dataset_path, str | None):
            raise TypeError("test_dataset_path must be a string or None")
        if not isinstance(self.train_datasets_split, str):
            raise TypeError("train_datasets_split must be a string")
        if not isinstance(self.val_datasets_split, str):
            raise TypeError("val_datasets_split must be a string")
        if not isinstance(self.test_datasets_split, str):
            raise TypeError("test_datasets_split must be a string")
        if not isinstance(self.smiles_a_column_name, str):
            raise TypeError("smiles_a_column_name must be a string")
        if not isinstance(self.smiles_b_column_name, str | None):
            raise TypeError("smiles_b_column_name must be a string or None")
        if not isinstance(self.label_column_name, str):
            raise TypeError("label_column_name must be a string")
        if self.n_train_samples is not None and not isinstance(self.n_train_samples, int):
            raise TypeError("n_train_samples must be an integer or None")
        if not isinstance(self.n_val_samples, int) and self.n_val_samples is not None:
            raise TypeError("n_val_samples must be an integer or None")
        if self.n_test_samples is not None and not isinstance(self.n_test_samples, int):
            raise TypeError("n_test_samples must be an integer or None")
        if not isinstance(self.early_stopping_patience, int | None):
            raise TypeError("early_stopping_patience must be an integer or None")
        if not isinstance(self.scale_learning_rate, bool):
            raise TypeError("scale_learning_rate must be a boolean")
        if not isinstance(self.use_normalized_weight_decay, bool):
            raise TypeError("use_normalized_weight_decay must be a boolean")
        # check values
        if self.train_dataset_path == "":
            raise ValueError("train_dataset_path must be set")
        if self.val_dataset_path == "":
            raise ValueError("val_dataset_path must be set")
        if self.test_dataset_path is not None and self.test_dataset_path == "":
            raise ValueError("test_dataset_path must be set")
        if self.train_datasets_split == "":
            raise ValueError("train_datasets_split must be set")
        if self.val_datasets_split == "":
            raise ValueError("val_datasets_split must be set")
        if self.test_datasets_split == "":
            raise ValueError("test_datasets_split must be set")
        if self.smiles_a_column_name == "":
            raise ValueError("smiles_a_column_name must be set")
        if self.smiles_b_column_name is not None and self.smiles_b_column_name == "":
            raise ValueError("smiles_b_column_name must be set")
        if self.label_column_name == "":
            raise ValueError("label_column_name must be set")
        if self.n_train_samples is not None and self.n_train_samples < 1:
            raise ValueError("n_train_samples must be greater than 0")
        if self.n_val_samples is not None and self.n_val_samples < 1:
            raise ValueError("n_val_samples must be greater than 0")
        if self.n_test_samples is not None and self.n_test_samples < 1:
            raise ValueError("n_test_samples must be greater than 0")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be greater than 0")
