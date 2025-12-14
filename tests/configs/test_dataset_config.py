# type: ignore
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

import pytest

from chem_mrl.schemas import DatasetConfig, SplitConfig
from chem_mrl.schemas.Enums import FieldTypeOption


def test_split_config_custom_values():
    """Test SplitConfig with custom values"""
    config = SplitConfig(
        name="validation_split",
        subset="clean",
        split_key="val",
        label_cast_type=FieldTypeOption.int64,
        sample_size=1000,
    )
    assert config.name == "validation_split"
    assert config.subset == "clean"
    assert config.split_key == "val"
    assert config.label_cast_type == FieldTypeOption.int64
    assert config.sample_size == 1000


def test_split_config_value_validation():
    """Test SplitConfig validation"""
    with pytest.raises(ValueError, match="name must be set"):
        SplitConfig(name="")
    with pytest.raises(ValueError, match="subset must be set"):
        SplitConfig(name="test", subset="")
    with pytest.raises(ValueError, match="split_key must be set"):
        SplitConfig(name="test", split_key="")
    with pytest.raises(ValueError, match="sample_size must be greater than 0"):
        SplitConfig(name="test", sample_size=0)
    with pytest.raises(ValueError, match="sample_size must be greater than 0"):
        SplitConfig(name="test", sample_size=-1)


def test_split_config_type_validation():
    """Test SplitConfig type validation"""
    with pytest.raises(TypeError, match="name must be a string"):
        SplitConfig(name=123)
    with pytest.raises(TypeError, match="subset must be a string or None"):
        SplitConfig(name="test", subset=123)
    with pytest.raises(TypeError, match="train_split_key must be a string"):
        SplitConfig(name="test", split_key=123)
    with pytest.raises(TypeError, match="label_cast_type must be a FieldTypeOption"):
        SplitConfig(name="test", label_cast_type="float32")
    with pytest.raises(TypeError, match="sample_size must be an integer or None"):
        SplitConfig(name="test", sample_size=1.5)
    with pytest.raises(TypeError, match="sample_size must be an integer or None"):
        SplitConfig(name="test", sample_size="1000")


def test_dataset_config_custom_values():
    """Test DatasetConfig with custom values"""
    train_split = SplitConfig(name="train_data", split_key="train", sample_size=1000)
    val_split = SplitConfig(name="val_data", split_key="validation", sample_size=500)
    test_split = SplitConfig(name="test_data", split_key="test", sample_size=200)

    config = DatasetConfig(
        key="test_dataset",
        train_dataset=train_split,
        val_dataset=val_split,
        test_dataset=test_split,
        smiles_a_column_name="molecule_a",
        smiles_b_column_name=None,
        label_column_name="target_score",
    )

    assert config.key == "test_dataset"
    assert config.train_dataset == train_split
    assert config.val_dataset == val_split
    assert config.test_dataset == test_split
    assert config.smiles_a_column_name == "molecule_a"
    assert config.smiles_b_column_name is None
    assert config.label_column_name == "target_score"


def test_dataset_config_value_validation():
    """Test DatasetConfig validation"""
    train_split = SplitConfig(name="train_data")

    with pytest.raises(ValueError, match="name must be set"):
        DatasetConfig(key="", train_dataset=train_split)
    with pytest.raises(ValueError, match="either train_dataset or val_dataset must be set"):
        DatasetConfig(key="test", train_dataset=None, val_dataset=None)
    with pytest.raises(ValueError, match="smiles_a_column_name must be set"):
        DatasetConfig(key="test", train_dataset=train_split, smiles_a_column_name="")
    with pytest.raises(ValueError, match="smiles_b_column_name must be set"):
        DatasetConfig(key="test", train_dataset=train_split, smiles_b_column_name="")
    with pytest.raises(ValueError, match="label_column_name must be set"):
        DatasetConfig(key="test", train_dataset=train_split, label_column_name="")


def test_dataset_config_type_validation():
    """Test DatasetConfig type validation"""
    train_split = SplitConfig(name="train_data")

    with pytest.raises(TypeError, match="key must be a string"):
        DatasetConfig(key=123, train_dataset=train_split)
    with pytest.raises(TypeError, match="train_dataset must be a SplitConfig instance or None"):
        DatasetConfig(key="test", train_dataset="invalid")
    with pytest.raises(TypeError, match="val_dataset must be a SplitConfig instance or None"):
        DatasetConfig(key="test", train_dataset=train_split, val_dataset="invalid")
    with pytest.raises(TypeError, match="test_dataset must be a SplitConfig instance or None"):
        DatasetConfig(key="test", train_dataset=train_split, test_dataset="invalid")
    with pytest.raises(TypeError, match="smiles_a_column_name must be a string"):
        DatasetConfig(key="test", train_dataset=train_split, smiles_a_column_name=123)
    with pytest.raises(TypeError, match="smiles_b_column_name must be a string or None"):
        DatasetConfig(key="test", train_dataset=train_split, smiles_b_column_name=123)
    with pytest.raises(TypeError, match="label_column_name must be a string"):
        DatasetConfig(key="test", train_dataset=train_split, label_column_name=123)


def test_dataset_config_with_val_only():
    """Test DatasetConfig with only validation dataset"""
    val_split = SplitConfig(name="val_data")
    config = DatasetConfig(key="test_dataset", val_dataset=val_split)

    assert config.key == "test_dataset"
    assert config.train_dataset is None
    assert config.val_dataset == val_split
    assert config.test_dataset is None


def test_dataset_config_asdict():
    """Test DatasetConfig asdict functionality"""
    train_split = SplitConfig(name="train_data", sample_size=1000)
    config = DatasetConfig(key="test_dataset", train_dataset=train_split)

    config_dict = config.asdict()
    assert isinstance(config_dict, dict)
    assert config_dict["key"] == "test_dataset"
    assert "train_dataset" in config_dict
