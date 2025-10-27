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

import os
from typing import Any

import pytest
from constants import TEST_CLASSIFICATION_PATH
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from chem_mrl.constants import CHEM_MRL_DIMENSIONS
from chem_mrl.losses import SelfAdjDiceLoss
from chem_mrl.schemas import BaseConfig, ClassifierConfig, DatasetConfig, SplitConfig
from chem_mrl.schemas.Enums import (
    ClassifierEvalMetricOption,
    ClassifierLossFctOption,
    DiceReductionOption,
)
from chem_mrl.trainers import ClassifierTrainer, TempDirTrainerExecutor

test_dir = "/tmp"
test_args: dict[str, Any] = {
    "num_train_epochs": 2.0,
    "do_eval": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
}


def create_test_config(
    model_config: ClassifierConfig | None = None,
    training_args_kwargs: dict[str, Any] | None = None,
    include_test_dataset: bool = False,
) -> BaseConfig:
    """Helper function to create test configs with the new DatasetConfig structure."""
    if model_config is None:
        model_config = ClassifierConfig()

    if training_args_kwargs is None:
        training_args_kwargs = {}

    merged_args = {**test_args, **training_args_kwargs}

    split_config = SplitConfig(name=TEST_CLASSIFICATION_PATH)
    dataset_config = DatasetConfig(
        key="test_chem_mrl",
        train_dataset=split_config,
        val_dataset=split_config,
        test_dataset=split_config if include_test_dataset else None,
    )

    return BaseConfig(
        model=model_config,
        training_args=SentenceTransformerTrainingArguments(test_dir, **merged_args),
        datasets=[dataset_config],
    )


def test_classifier_trainer_instantiation():
    config = create_test_config()
    trainer = ClassifierTrainer(config)
    assert isinstance(trainer, ClassifierTrainer)
    assert isinstance(trainer.config, BaseConfig)
    assert isinstance(trainer.config.model, ClassifierConfig)


def test_classifier_resume_from_checkpoint():
    config = create_test_config()
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    executor.execute()

    config.training_args.resume_from_checkpoint = os.path.join(executor._temp_dir.name, "checkpoint-1")
    trainer = ClassifierTrainer(config)
    resume_executor = TempDirTrainerExecutor(trainer)
    resume_executor.execute()


def test_classifier_test_evaluator():
    config = create_test_config(include_test_dataset=True)
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("weight_decay", [0.0, 1e-8, 1e-4, 1e-2, 0.1])
def test_chem_mrl_test_weight_decay(weight_decay):
    config = create_test_config(training_args_kwargs={"weight_decay": weight_decay})
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("dimension", CHEM_MRL_DIMENSIONS)
def test_classifier_classifier_hidden_dimensions(
    dimension,
):
    config = create_test_config(model_config=ClassifierConfig(classifier_hidden_dimension=dimension))
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert trainer.model.truncate_dim == dimension
    assert trainer.loss_function.smiles_embedding_dimension == dimension
    assert isinstance(result, float)


@pytest.mark.parametrize("eval_metric", ClassifierEvalMetricOption)
def test_classifier_eval_metrics(eval_metric):
    config = create_test_config(model_config=ClassifierConfig(eval_metric=eval_metric))
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


def test_classifier_freeze_internal_model():
    config = create_test_config(model_config=ClassifierConfig(freeze_model=True))
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert trainer.loss_function.freeze_model is True
    assert isinstance(result, float)


def test_classifier_num_labels():
    config = create_test_config(model_config=ClassifierConfig(freeze_model=True))
    trainer = ClassifierTrainer(config)
    assert trainer.loss_function.num_labels == 4  # testing dataset only has 4 classes


@pytest.mark.parametrize("dropout_p", [0.0, 0.1, 0.5, 1.0])
def test_classifier_dropout(dropout_p):
    config = create_test_config(model_config=ClassifierConfig(dropout_p=dropout_p))
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert trainer.loss_function.dropout_p == dropout_p
    assert isinstance(result, float)


def test_dice_loss_classifier_trainer_instantiation():
    config = create_test_config(model_config=ClassifierConfig(loss_func=ClassifierLossFctOption.selfadjdice))
    trainer = ClassifierTrainer(config)
    assert isinstance(trainer, ClassifierTrainer)
    assert isinstance(trainer.loss_function, SelfAdjDiceLoss)
    assert trainer.config.model.loss_func == "selfadjdice"


@pytest.mark.parametrize("dice_reduction", DiceReductionOption)
def test_dice_loss_classifier_dice_reduction_options(dice_reduction):
    config = create_test_config(
        model_config=ClassifierConfig(loss_func=ClassifierLossFctOption.selfadjdice, dice_reduction=dice_reduction)
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(trainer.loss_function, SelfAdjDiceLoss)
    assert isinstance(result, float)


@pytest.mark.parametrize("dice_gamma", [0.0, 0.5, 1.0, 2.0])
def test_dice_loss_classifier_dice_gamma_values(dice_gamma):
    config = create_test_config(
        model_config=ClassifierConfig(loss_func=ClassifierLossFctOption.selfadjdice, dice_gamma=dice_gamma)
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(trainer.loss_function, SelfAdjDiceLoss)
    assert isinstance(result, float)


@pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
def test_classifier_batch_sizes(batch_size):
    config = create_test_config(
        training_args_kwargs={
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
        }
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("lr", [1e-6, 1e-4, 1e-2])
def test_classifier_learning_rates(lr):
    config = create_test_config(training_args_kwargs={"learning_rate": lr})
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("path", ["test_output", "custom/nested/path", "model_outputs/test"])
def test_classifier_output_paths(path):
    # Note: For this test, we need to override the output_dir in training_args
    # We'll create the config manually since we need to set the output_dir differently
    split_config = SplitConfig(name=TEST_CLASSIFICATION_PATH)
    dataset_config = DatasetConfig(
        key="test",
        train_dataset=split_config,
        val_dataset=split_config,
    )
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(path, **test_args),
        datasets=[dataset_config],
    )
    trainer = ClassifierTrainer(config)
    assert path in trainer.config.training_args.output_dir


def test_multi_dataset_trainer_initialization():
    split_config = SplitConfig(name=TEST_CLASSIFICATION_PATH)
    split_config_2 = SplitConfig(name=TEST_CLASSIFICATION_PATH)
    dataset_config_1 = DatasetConfig(
        key="test",
        train_dataset=split_config,
        val_dataset=split_config,
        test_dataset=split_config,
    )
    dataset_config_2 = DatasetConfig(
        key="test_2",
        train_dataset=split_config_2,
        val_dataset=split_config_2,
        test_dataset=split_config_2,
    )
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        datasets=[dataset_config_1, dataset_config_2],
    )
    trainer = ClassifierTrainer(config)

    assert isinstance(trainer, ClassifierTrainer)
    assert isinstance(trainer.config, BaseConfig)
    assert len(trainer.config.datasets) == 2
    assert isinstance(trainer.train_dataset, dict)
    assert isinstance(trainer.eval_dataset, dict)
    assert "test" in trainer.train_dataset
    assert "test_2" in trainer.train_dataset
    assert "test" in trainer.eval_dataset
    assert "test_2" in trainer.eval_dataset
    assert len(trainer.train_dataset["test"]) > 0
    assert len(trainer.train_dataset["test_2"]) > 0
    assert len(trainer.eval_dataset["test"]) > 0
    assert len(trainer.eval_dataset["test_2"]) > 0


def test_multi_dataset_training_execution():
    split_config = SplitConfig(name=TEST_CLASSIFICATION_PATH)
    split_config_2 = SplitConfig(name=TEST_CLASSIFICATION_PATH)
    dataset_config_1 = DatasetConfig(key="test", train_dataset=split_config, val_dataset=split_config)
    dataset_config_2 = DatasetConfig(key="test_2", train_dataset=split_config_2, val_dataset=split_config_2)
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(
            test_dir,
            num_train_epochs=1.0,
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
        ),
        datasets=[dataset_config_1, dataset_config_2],
    )

    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)

    result = executor.execute()
    assert isinstance(result, float)
    assert len(trainer.train_dataset) == 2
    assert len(trainer.eval_dataset) == 2
    assert trainer.steps_per_epoch > len(trainer.train_dataset["test"]), (
        "Steps per epoch should be greater than the number of training samples for one dataset"
    )


def test_multi_evaluation_dataset():
    split_config = SplitConfig(name=TEST_CLASSIFICATION_PATH)
    split_config_2 = SplitConfig(name=TEST_CLASSIFICATION_PATH)
    dataset_config_1 = DatasetConfig(key="test", train_dataset=split_config, val_dataset=split_config)
    dataset_config_2 = DatasetConfig(key="test_2", val_dataset=split_config_2)

    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(
            test_dir,
            num_train_epochs=1.0,
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
        ),
        datasets=[dataset_config_1, dataset_config_2],
    )

    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)

    result = executor.execute()
    assert isinstance(result, float)
    assert len(trainer.train_dataset) == 1
    assert len(trainer.eval_dataset) == 2


def test_multi_dataset_identical_datasets_error_handling():
    dataset_configs = []
    split_config = SplitConfig(name=TEST_CLASSIFICATION_PATH)
    for _ in range(3):
        dataset_configs.append(
            DatasetConfig(
                key="test",
                train_dataset=split_config,
                val_dataset=split_config,
            )
        )
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(
            test_dir,
            num_train_epochs=1.0,
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
        ),
        datasets=dataset_configs,
    )

    with pytest.raises(ValueError, match="Duplicate dataset name found"):
        ClassifierTrainer(config)
