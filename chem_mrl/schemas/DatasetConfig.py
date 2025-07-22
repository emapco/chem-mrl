from dataclasses import asdict, dataclass

from .Enums import FieldTypeOption


@dataclass
class SplitConfig:
    name: str
    split_key: str = "train"
    label_cast_type: FieldTypeOption = FieldTypeOption.float32
    sample_size: int | None = None

    def __post_init__(self):
        # check types
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not isinstance(self.split_key, str):
            raise TypeError("train_split_key must be a string")
        if not isinstance(self.label_cast_type, FieldTypeOption):
            raise TypeError("label_cast_type must be a FieldTypeOption")
        if not isinstance(self.sample_size, int | None):
            raise TypeError("sample_size must be an integer or None")
        # check values
        if self.name == "":
            raise ValueError("name must be set")
        if self.split_key == "":
            raise ValueError("split_key must be set")
        if self.label_cast_type not in FieldTypeOption:
            raise ValueError("label_cast_type must be a FieldTypeOption")
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError("sample_size must be greater than 0")


@dataclass
class DatasetConfig:
    key: str
    train_dataset: SplitConfig | None = None
    val_dataset: SplitConfig | None = None
    test_dataset: SplitConfig | None = None
    smiles_a_column_name: str = "smiles_a"
    smiles_b_column_name: str | None = "smiles_b"
    label_column_name: str = "similarity"
    asdict = asdict

    def __post_init__(self):
        # check types
        if not isinstance(self.key, str):
            raise TypeError("key must be a string")
        if not isinstance(self.train_dataset, SplitConfig | None):
            raise TypeError("train_dataset must be a SplitConfig instance or None")
        if not isinstance(self.val_dataset, SplitConfig | None):
            raise TypeError("val_dataset must be a SplitConfig instance or None")
        if not isinstance(self.test_dataset, SplitConfig | None):
            raise TypeError("test_dataset must be a SplitConfig instance or None")
        if not isinstance(self.smiles_a_column_name, str):
            raise TypeError("smiles_a_column_name must be a string")
        if not isinstance(self.smiles_b_column_name, str | None):
            raise TypeError("smiles_b_column_name must be a string or None")
        if not isinstance(self.label_column_name, str):
            raise TypeError("label_column_name must be a string")
        # check values
        if self.key == "":
            raise ValueError("name must be set")
        if self.train_dataset is None and self.val_dataset is None:
            raise ValueError("either train_dataset or val_dataset must be set")
        if self.test_dataset is not None and self.test_dataset == "":
            raise ValueError("test_dataset must be set")
        if self.smiles_a_column_name == "":
            raise ValueError("smiles_a_column_name must be set")
        if self.smiles_b_column_name is not None and self.smiles_b_column_name == "":
            raise ValueError("smiles_b_column_name must be set")
        if self.label_column_name == "":
            raise ValueError("label_column_name must be set")
