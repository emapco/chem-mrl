# type: ignore
from typing import Any

import pytest
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.schemas import BaseConfig, ChemMRLConfig, DatasetConfig, SplitConfig

test_dir = ""
test_args: dict[str, Any] = {"eval_strategy": "epoch"}


def test_base_config_custom_values():
    train_split = SplitConfig(name="train_data")
    val_split = SplitConfig(name="val_data")
    test_split = SplitConfig(name="test_data")
    dataset_config = DatasetConfig(
        key="test_dataset",
        train_dataset=train_split,
        val_dataset=val_split,
        test_dataset=test_split,
        smiles_a_column_name="molecule_a",
        smiles_b_column_name=None,
        label_column_name="target_score",
    )
    config = BaseConfig(
        model=ChemMRLConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        datasets=[dataset_config],
        early_stopping_patience=5,
        scale_learning_rate=True,
        use_normalized_weight_decay=True,
    )
    assert len(config.datasets) == 1
    assert config.datasets[0].train_dataset == train_split
    assert config.datasets[0].val_dataset == val_split
    assert config.datasets[0].test_dataset == test_split
    assert config.datasets[0].smiles_a_column_name == "molecule_a"
    assert config.datasets[0].smiles_b_column_name is None
    assert config.datasets[0].label_column_name == "target_score"
    assert config.early_stopping_patience == 5
    assert config.scale_learning_rate is True
    assert config.use_normalized_weight_decay is True


def test_base_config_validation():
    with pytest.raises(ValueError, match="at least one dataset config must be provided"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            datasets=[],
        )
    with pytest.raises(ValueError, match="early_stopping_patience must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            datasets=[DatasetConfig(key="train", train_dataset=SplitConfig(name="train_data"))],
            early_stopping_patience=0,
        )


def test_config_asdict():
    dataset_config = DatasetConfig(key="train", train_dataset=SplitConfig(name="train_data"))
    base_config = BaseConfig(
        model=ChemMRLConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        datasets=[dataset_config],
    )
    base_dict = base_config.asdict()
    assert isinstance(base_dict, dict)
    assert "datasets" in base_dict
    assert len(base_dict["datasets"]) == 1


def test_base_config_multiple_datasets():
    """Test BaseConfig with multiple datasets"""
    train_split = SplitConfig(name="train1.parquet")
    train_split2 = SplitConfig(name="train2.parquet")
    val_split = SplitConfig(name="train1.parquet")
    val_split2 = SplitConfig(name="train2.parquet")
    dataset1 = DatasetConfig(
        key="ds1",
        train_dataset=train_split,
        val_dataset=val_split,
    )
    dataset2 = DatasetConfig(
        key="ds2",
        train_dataset=train_split2,
        val_dataset=val_split2,
        test_dataset=SplitConfig(name="test2.parquet"),
    )
    config = BaseConfig(
        model=ChemMRLConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        datasets=[dataset1, dataset2],
    )
    assert len(config.datasets) == 2
    assert config.datasets[0].train_dataset.name == "train1.parquet"
    assert config.datasets[1].train_dataset.name == "train2.parquet"
    assert config.datasets[0].test_dataset is None
    assert config.datasets[1].test_dataset.name == "test2.parquet"


def test_base_config_type_validation():
    """Test type validation for base config parameters"""
    train_split = SplitConfig(name="train_data")
    with pytest.raises(TypeError, match="datasets must be a list"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            datasets="not_a_list",
        )
    with pytest.raises(TypeError, match="all items in datasets must be DatasetConfig instances"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            datasets=["not_a_dataset_config"],
        )
    with pytest.raises(TypeError, match="early_stopping_patience must be an integer or None"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            datasets=[DatasetConfig(key="test", train_dataset=train_split)],
            early_stopping_patience=1.5,
        )
    with pytest.raises(TypeError, match="scale_learning_rate must be a boolean"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            datasets=[DatasetConfig(key="test", train_dataset=train_split)],
            scale_learning_rate=123,
        )
    with pytest.raises(TypeError, match="use_normalized_weight_decay must be a boolean"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            datasets=[DatasetConfig(key="test", train_dataset=train_split)],
            use_normalized_weight_decay=123,
        )


def test_base_config_invalid_dataset_config():
    """Test BaseConfig that at least one dataset config has a train_dataset defined"""
    val_split1 = SplitConfig(name="val1.parquet")
    val_split2 = SplitConfig(name="val2.parquet")
    val_dataset_config1 = DatasetConfig(
        key="dataset1",
        val_dataset=val_split1,
        smiles_a_column_name="smiles_a",
        label_column_name="label",
    )
    val_dataset_config2 = DatasetConfig(
        key="dataset2",
        val_dataset=val_split2,
        smiles_a_column_name="smiles_a",
        label_column_name="label",
    )
    with pytest.raises(ValueError, match="at least one dataset config must have a train_dataset"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            datasets=[val_dataset_config1, val_dataset_config2],
        )
