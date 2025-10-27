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
from constants import TEST_CHEM_MRL_PATH
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from chem_mrl.schemas import (
    BaseConfig,
    ChemMRLConfig,
    DatasetConfig,
    SplitConfig,
)
from chem_mrl.schemas.Enums import (
    ChemMrlEvalMetricOption,
    ChemMrlLossFctOption,
    EmbeddingPoolingOption,
    EvalSimilarityFctOption,
    TanimotoSimilarityBaseLossFctOption,
)
from chem_mrl.trainers import ChemMRLTrainer, TempDirTrainerExecutor

test_dir = "/tmp"
test_args: dict[str, Any] = {
    "num_train_epochs": 2.0,
    "do_eval": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
}


def create_test_config(
    model_config: ChemMRLConfig | None = None,
    training_args_kwargs: dict[str, Any] | None = None,
    include_test_dataset: bool = False,
) -> BaseConfig:
    """Helper function to create test configs with the new DatasetConfig structure."""
    if model_config is None:
        model_config = ChemMRLConfig()

    if training_args_kwargs is None:
        training_args_kwargs = {}

    merged_args = {**test_args, **training_args_kwargs}

    split_config = SplitConfig(name=TEST_CHEM_MRL_PATH)
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


def test_chem_mrl_trainer_instantiation():
    config = create_test_config()
    trainer = ChemMRLTrainer(config)
    assert isinstance(trainer, ChemMRLTrainer)
    assert isinstance(trainer.config, BaseConfig)
    assert isinstance(trainer.config.model, ChemMRLConfig)


def test_chem_mrl_resume_from_checkpoint():
    config = create_test_config()
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    executor.execute()

    config.training_args.resume_from_checkpoint = os.path.join(executor._temp_dir.name, "checkpoint-1")
    trainer = ChemMRLTrainer(config)
    resume_executor = TempDirTrainerExecutor(trainer)
    resume_executor.execute()


def test_chem_mrl_test_evaluator():
    config = create_test_config(model_config=ChemMRLConfig(n_dims_per_step=4), include_test_dataset=True)
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("weight_decay", [0.0, 1e-8, 1e-4, 1e-2, 0.1])
def test_chem_mrl_test_weight_decay(weight_decay):
    config = create_test_config(training_args_kwargs={"weight_decay": weight_decay})
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("pooling", EmbeddingPoolingOption)
def test_chem_mrl_pooling_options(pooling):
    config = create_test_config(model_config=ChemMRLConfig(embedding_pooling=pooling))
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(trainer.config.model, ChemMRLConfig)
    assert trainer.config.model.embedding_pooling == pooling
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "loss_func",
    [
        ChemMrlLossFctOption.angleloss,
        ChemMrlLossFctOption.cosentloss,
        ChemMrlLossFctOption.tanimotosentloss,
    ],
)
def test_chem_mrl_loss_functions(
    loss_func: ChemMrlLossFctOption | ChemMrlLossFctOption | ChemMrlLossFctOption,
):
    # can't test tanimotosimilarityloss since it requires an additional parameter
    config = create_test_config(model_config=ChemMRLConfig(loss_func=loss_func))
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("base_loss", TanimotoSimilarityBaseLossFctOption)
def test_chem_mrl_tanimoto_similarity_loss(base_loss):
    config = create_test_config(
        model_config=ChemMRLConfig(
            loss_func=ChemMrlLossFctOption.tanimotosimilarityloss,
            tanimoto_similarity_loss_func=base_loss,
        )
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("eval_similarity_fct", EvalSimilarityFctOption)
def test_chem_mrl_eval_similarity(eval_similarity_fct):
    config = create_test_config(model_config=ChemMRLConfig(eval_similarity_fct=eval_similarity_fct))
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("eval_metric", ChemMrlEvalMetricOption)
def test_chem_mrl_eval_metrics(eval_metric):
    config = create_test_config(model_config=ChemMRLConfig(eval_metric=eval_metric))
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


def test_chem_2d_mrl_trainer_instantiation():
    config = create_test_config(model_config=ChemMRLConfig(use_2d_matryoshka=True))
    trainer = ChemMRLTrainer(config)
    assert isinstance(trainer, ChemMRLTrainer)
    assert isinstance(trainer.config.model, ChemMRLConfig)
    assert trainer.config.model.use_2d_matryoshka is True


def test_mrl_dimension_weights_validation():
    with pytest.raises(ValueError, match="Dimension weights must be in increasing order"):
        config = create_test_config(
            model_config=ChemMRLConfig(mrl_dimension_weights=(2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
        )
        ChemMRLTrainer(config)


def test_2d_mrl_layer_weights():
    config = create_test_config(
        model_config=ChemMRLConfig(
            use_2d_matryoshka=True,
            last_layer_weight=2.0,
            prior_layers_weight=1.0,
        )
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
def test_chem_mrl_batch_sizes(batch_size):
    config = create_test_config(
        training_args_kwargs={
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
        }
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("lr", [1e-6, 1e-4, 1e-2])
def test_chem_mrl_learning_rates(lr):
    config = create_test_config(training_args_kwargs={"learning_rate": lr})
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("path", ["test_output", "custom/nested/path", "model_outputs/test"])
def test_classifier_output_paths(path):
    # Note: For this test, we need to override the output_dir in training_args
    # We'll create the config manually since we need to set the output_dir differently
    split_config = SplitConfig(name=TEST_CHEM_MRL_PATH)
    dataset_config = DatasetConfig(
        key="test",
        train_dataset=split_config,
        val_dataset=split_config,
    )
    config = BaseConfig(
        model=ChemMRLConfig(),
        training_args=SentenceTransformerTrainingArguments(path, **test_args),
        datasets=[dataset_config],
    )
    trainer = ChemMRLTrainer(config)
    assert path in trainer.config.training_args.output_dir


def test_multi_dataset_trainer_initialization():
    split_config = SplitConfig(name=TEST_CHEM_MRL_PATH)
    split_config_2 = SplitConfig(name=TEST_CHEM_MRL_PATH)
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
        model=ChemMRLConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        datasets=[dataset_config_1, dataset_config_2],
    )
    trainer = ChemMRLTrainer(config)

    assert isinstance(trainer, ChemMRLTrainer)
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
    split_config = SplitConfig(name=TEST_CHEM_MRL_PATH)
    split_config_2 = SplitConfig(name=TEST_CHEM_MRL_PATH)
    dataset_config_1 = DatasetConfig(key="test", train_dataset=split_config, val_dataset=split_config)
    dataset_config_2 = DatasetConfig(key="test_2", train_dataset=split_config_2, val_dataset=split_config_2)
    config = BaseConfig(
        model=ChemMRLConfig(n_dims_per_step=4),
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

    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)

    result = executor.execute()
    assert isinstance(result, float)
    assert len(trainer.train_dataset) == 2
    assert len(trainer.eval_dataset) == 2
    assert trainer.steps_per_epoch > len(trainer.train_dataset["test"]), (
        "Steps per epoch should be greater than the number of training samples for one dataset"
    )


def test_multi_evaluation_dataset():
    split_config = SplitConfig(name=TEST_CHEM_MRL_PATH)
    split_config_2 = SplitConfig(name=TEST_CHEM_MRL_PATH)
    dataset_config_1 = DatasetConfig(key="test", train_dataset=split_config, val_dataset=split_config)
    dataset_config_2 = DatasetConfig(key="test_2", val_dataset=split_config_2)

    config = BaseConfig(
        model=ChemMRLConfig(n_dims_per_step=4),
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

    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)

    result = executor.execute()
    assert isinstance(result, float)
    assert len(trainer.train_dataset) == 1
    assert len(trainer.eval_dataset) == 2


def test_multi_dataset_identical_datasets_error_handling():
    dataset_configs = []
    split_config = SplitConfig(name=TEST_CHEM_MRL_PATH)
    for _ in range(3):
        dataset_configs.append(
            DatasetConfig(
                key="test",
                train_dataset=split_config,
                val_dataset=split_config,
            )
        )
    config = BaseConfig(
        model=ChemMRLConfig(n_dims_per_step=4),
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
        ChemMRLTrainer(config)
