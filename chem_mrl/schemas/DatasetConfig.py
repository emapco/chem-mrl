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

from dataclasses import asdict, dataclass

from .Enums import FieldTypeOption


@dataclass
class SplitConfig:
    """Configuration for a dataset split.

    Attributes:
        name: HuggingFace dataset name or path should load as a datasets.DatasetDict.
        subset: (Optional) The subset to load from the dataset.
        split_key: If loading from different local files, this should be set to train.
        label_cast_type: float64, float32, or float16 (used by chem_mrl model).
        sample_size: Number of training samples to load. Uses seeded sampling if a seed is set.
    """

    name: str
    subset: str | None = None
    split_key: str = "train"
    label_cast_type: FieldTypeOption = FieldTypeOption.float32
    sample_size: int | None = None

    def __post_init__(self):
        # check types
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not isinstance(self.subset, str | None):
            raise TypeError("subset must be a string or None")
        if not isinstance(self.split_key, str):
            raise TypeError("train_split_key must be a string")
        if not isinstance(self.label_cast_type, FieldTypeOption):
            raise TypeError("label_cast_type must be a FieldTypeOption")
        if not isinstance(self.sample_size, int | None):
            raise TypeError("sample_size must be an integer or None")
        # check values
        if self.name == "":
            raise ValueError("name must be set")
        if self.subset is not None and self.subset == "":
            raise ValueError("subset must be set")
        if self.split_key == "":
            raise ValueError("split_key must be set")
        if self.label_cast_type not in FieldTypeOption:
            raise ValueError("label_cast_type must be a FieldTypeOption")
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError("sample_size must be greater than 0")


@dataclass
class DatasetConfig:
    """Configuration for a dataset.

    Attributes:
        key: Should be unique.
        train_dataset: may be set to null if only using a val_dataset.
        val_dataset: May be set to null if only using a train_dataset.
        test_dataset: may be set to null to disable evaluation on a test dataset.
        smiles_a_column_name: smiles_a.
        smiles_b_column_name: Can be set to `null` when training a classifier model.
        label_column_name: similarity.
    """

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
