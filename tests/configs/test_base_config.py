# type: ignore
from typing import Any

import pytest
from sentence_transformers import SentenceTransformerTrainingArguments

from chem_mrl.schemas import ChemMRLConfig
from chem_mrl.schemas.BaseConfig import (
    BaseConfig,
)

test_dir = ""
test_args: dict[str, Any] = {"eval_strategy": "epoch"}


def test_base_config_custom_values():
    config = BaseConfig(
        model=ChemMRLConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        test_dataset_path="test.parquet",
        train_datasets_split="train1",
        val_datasets_split="train2",
        test_datasets_split="train3",
        smiles_a_column_name="asdf",
        smiles_b_column_name=None,
        label_column_name="asdf",
        n_train_samples=1000,
        n_val_samples=500,
        n_test_samples=200,
        early_stopping_patience=5,
        scale_learning_rate=True,
        use_normalized_weight_decay=True,
    )
    assert config.train_dataset_path == "train.parquet"
    assert config.val_dataset_path == "val.parquet"
    assert config.test_dataset_path == "test.parquet"
    assert config.train_datasets_split == "train1"
    assert config.val_datasets_split == "train2"
    assert config.test_datasets_split == "train3"
    assert config.smiles_a_column_name == "asdf"
    assert config.smiles_b_column_name is None
    assert config.label_column_name == "asdf"
    assert config.n_train_samples == 1000
    assert config.n_val_samples == 500
    assert config.n_test_samples == 200
    assert config.early_stopping_patience == 5
    assert config.scale_learning_rate is True
    assert config.use_normalized_weight_decay is True


def test_base_config_validation():
    with pytest.raises(ValueError, match="train_dataset_path must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="",
            val_dataset_path="test",
        )
    with pytest.raises(ValueError, match="val_dataset_path must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="",
        )
    with pytest.raises(ValueError, match="test_dataset_path must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            test_dataset_path="",
        )
    with pytest.raises(ValueError, match="train_datasets_split must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            train_datasets_split="",
        )
    with pytest.raises(ValueError, match="val_datasets_split must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            val_datasets_split="",
        )
    with pytest.raises(ValueError, match="test_datasets_split must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            test_datasets_split="",
        )
    with pytest.raises(ValueError, match="smiles_a_column_name must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            smiles_a_column_name="",
        )
    with pytest.raises(ValueError, match="smiles_b_column_name must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            smiles_b_column_name="",
        )
    with pytest.raises(ValueError, match="label_column_name must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            label_column_name="",
        )
    with pytest.raises(ValueError, match="n_train_samples must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            n_train_samples=0,
        )
    with pytest.raises(ValueError, match="n_val_samples must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            n_val_samples=0,
        )
    with pytest.raises(ValueError, match="n_test_samples must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            n_test_samples=0,
        )
    with pytest.raises(ValueError, match="early_stopping_patience must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
            train_dataset_path="test",
            val_dataset_path="test",
            early_stopping_patience=0,
        )


def test_config_asdict():
    base_config = BaseConfig(
        model=ChemMRLConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path="test",
        val_dataset_path="test",
    )
    base_dict = base_config.asdict()
    assert isinstance(base_dict, dict)
    assert "smiles_a_column_name" in base_dict


def test_base_config_type_validation():
    """Test type validation for base config parameters"""
    with pytest.raises(TypeError):
        BaseConfig(train_dataset_path=123)
    with pytest.raises(TypeError):
        BaseConfig(val_dataset_path=123)
    with pytest.raises(TypeError):
        BaseConfig(test_dataset_path=123)
    with pytest.raises(TypeError):
        BaseConfig(train_datasets_split=123)
    with pytest.raises(TypeError):
        BaseConfig(val_datasets_split=123)
    with pytest.raises(TypeError):
        BaseConfig(test_datasets_split=123)
    with pytest.raises(TypeError):
        ChemMRLConfig(smiles_a_column_name=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(smiles_b_column_name=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(label_column_name=1)
    with pytest.raises(TypeError):
        BaseConfig(n_train_samples=1.5)
    with pytest.raises(TypeError):
        BaseConfig(n_val_samples=1.5)
    with pytest.raises(TypeError):
        BaseConfig(n_test_samples=1.5)
    with pytest.raises(TypeError):
        BaseConfig(n_train_samples="123")
    with pytest.raises(TypeError):
        BaseConfig(n_val_samples="123")
    with pytest.raises(TypeError):
        BaseConfig(n_test_samples="123")
    with pytest.raises(TypeError):
        BaseConfig(early_stopping_patience=1.5)
    with pytest.raises(TypeError):
        BaseConfig(scale_learning_rate=123)
    with pytest.raises(TypeError):
        BaseConfig(use_normalized_weight_decay=123)
