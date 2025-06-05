import os
from typing import Any

import pytest
from constants import TEST_CLASSIFICATION_PATH
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from chem_mrl.constants import CHEM_MRL_DIMENSIONS
from chem_mrl.losses import SelfAdjDiceLoss
from chem_mrl.schemas import BaseConfig, ClassifierConfig
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


def test_classifier_trainer_instantiation():
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    assert isinstance(trainer, ClassifierTrainer)
    assert isinstance(trainer.config, BaseConfig)
    assert isinstance(trainer.config.model, ClassifierConfig)


def test_classifier_resume_from_checkpoint():
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    executor.execute()

    config.training_args.resume_from_checkpoint = os.path.join(
        executor._temp_dir.name, "checkpoints", "checkpoint-1"
    )
    trainer = ClassifierTrainer(config)
    resume_executor = TempDirTrainerExecutor(trainer)
    resume_executor.execute()


def test_classifier_test_evaluator():
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        test_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("weight_decay", [0.0, 1e-8, 1e-4, 1e-2, 0.1])
def test_chem_mrl_test_weight_decay(weight_decay):
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(
            test_dir,
            weight_decay=weight_decay,
            **test_args,
        ),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("dimension", CHEM_MRL_DIMENSIONS)
def test_classifier_classifier_hidden_dimensions(
    dimension,
):
    config = BaseConfig(
        model=ClassifierConfig(classifier_hidden_dimension=dimension),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert trainer.model.truncate_dim == dimension
    assert trainer.loss_function.smiles_embedding_dimension == dimension
    assert isinstance(result, float)


@pytest.mark.parametrize("eval_metric", ClassifierEvalMetricOption)
def test_classifier_eval_metrics(eval_metric):
    config = BaseConfig(
        model=ClassifierConfig(eval_metric=eval_metric),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


def test_classifier_freeze_internal_model():
    config = BaseConfig(
        model=ClassifierConfig(freeze_model=True),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert trainer.loss_function.freeze_model is True
    assert isinstance(result, float)


def test_classifier_num_labels():
    config = BaseConfig(
        model=ClassifierConfig(freeze_model=True),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    assert trainer.loss_function.num_labels == 4  # testing dataset only has 4 classes


@pytest.mark.parametrize("dropout_p", [0.0, 0.1, 0.5, 1.0])
def test_classifier_dropout(dropout_p):
    config = BaseConfig(
        model=ClassifierConfig(dropout_p=dropout_p),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert trainer.loss_function.dropout_p == dropout_p
    assert isinstance(result, float)


def test_dice_loss_classifier_trainer_instantiation():
    config = BaseConfig(
        model=ClassifierConfig(loss_func=ClassifierLossFctOption.selfadjdice),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    assert isinstance(trainer, ClassifierTrainer)
    assert isinstance(trainer.loss_function, SelfAdjDiceLoss)
    assert trainer.config.model.loss_func == "selfadjdice"


@pytest.mark.parametrize("dice_reduction", DiceReductionOption)
def test_dice_loss_classifier_dice_reduction_options(dice_reduction):
    config = BaseConfig(
        model=ClassifierConfig(
            loss_func=ClassifierLossFctOption.selfadjdice, dice_reduction=dice_reduction
        ),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(trainer.loss_function, SelfAdjDiceLoss)
    assert isinstance(result, float)


@pytest.mark.parametrize("dice_gamma", [0.0, 0.5, 1.0, 2.0])
def test_dice_loss_classifier_dice_gamma_values(dice_gamma):
    config = BaseConfig(
        model=ClassifierConfig(
            loss_func=ClassifierLossFctOption.selfadjdice, dice_gamma=dice_gamma
        ),
        training_args=SentenceTransformerTrainingArguments(test_dir, **test_args),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(trainer.loss_function, SelfAdjDiceLoss)
    assert isinstance(result, float)


@pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
def test_classifier_batch_sizes(batch_size):
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(
            test_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            **test_args,
        ),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("lr", [1e-6, 1e-4, 1e-2])
def test_classifier_learning_rates(lr):
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(
            test_dir,
            learning_rate=lr,
            **test_args,
        ),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("path", ["test_output", "custom/nested/path", "model_outputs/test"])
def test_classifier_output_paths(path):
    config = BaseConfig(
        model=ClassifierConfig(),
        training_args=SentenceTransformerTrainingArguments(
            output_dir=path,
            **test_args,
        ),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    assert path in trainer.config.training_args.output_dir
