import pytest
from constants import TEST_CLASSIFICATION_PATH

from chem_mrl.constants import CHEM_MRL_DIMENSIONS
from chem_mrl.losses import SelfAdjDiceLoss
from chem_mrl.schemas import BaseConfig, ClassifierConfig, WandbConfig
from chem_mrl.schemas.Enums import (
    ClassifierEvalMetricOption,
    ClassifierLossFctOption,
    DiceReductionOption,
    SchedulerOption,
)
from chem_mrl.trainers import ClassifierTrainer, TempDirTrainerExecutor


def test_classifier_trainer_instantiation():
    config = BaseConfig(
        model=ClassifierConfig(),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        wandb=WandbConfig(
            enabled=True,
            project_name="chem_mrl_test",
            run_name="test",
            use_watch=True,
            watch_log_freq=1000,
            watch_log_graph=True,
        ),
    )
    trainer = ClassifierTrainer(config)
    assert isinstance(trainer, ClassifierTrainer)
    assert isinstance(trainer.config, BaseConfig)
    assert isinstance(trainer.config.model, ClassifierConfig)


def test_classifier_test_evaluator():
    config = BaseConfig(
        model=ClassifierConfig(),
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
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        weight_decay=weight_decay,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("scheduler", SchedulerOption)
def test_classifier_scheduler_options(
    scheduler,
):
    config = BaseConfig(
        model=ClassifierConfig(),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        scheduler=scheduler,
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
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert trainer.model.truncate_dim == dimension
    assert trainer.loss_fct.smiles_embedding_dimension == dimension
    assert isinstance(result, float)


@pytest.mark.parametrize("eval_metric", ClassifierEvalMetricOption)
def test_classifier_eval_metrics(eval_metric):
    config = BaseConfig(
        model=ClassifierConfig(eval_metric=eval_metric),
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
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert trainer.loss_fct.freeze_model is True
    assert isinstance(result, float)


def test_classifier_num_labels():
    config = BaseConfig(
        model=ClassifierConfig(freeze_model=True),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    assert trainer.loss_fct.num_labels == 4  # testing dataset only has 4 classes


@pytest.mark.parametrize("dropout_p", [0.0, 0.1, 0.5, 1.0])
def test_classifier_dropout(dropout_p):
    config = BaseConfig(
        model=ClassifierConfig(dropout_p=dropout_p),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert trainer.loss_fct.dropout_p == dropout_p
    assert isinstance(result, float)


def test_dice_loss_classifier_trainer_instantiation():
    config = BaseConfig(
        model=ClassifierConfig(loss_func=ClassifierLossFctOption.selfadjdice),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        wandb=WandbConfig(
            project_name="chem_mrl_test",
            run_name="test",
            use_watch=True,
            watch_log_freq=1000,
            watch_log_graph=True,
        ),
    )
    trainer = ClassifierTrainer(config)
    assert isinstance(trainer, ClassifierTrainer)
    assert isinstance(trainer.loss_fct, SelfAdjDiceLoss)
    assert trainer.config.model.loss_func == "selfadjdice"


@pytest.mark.parametrize("dice_reduction", DiceReductionOption)
def test_dice_loss_classifier_dice_reduction_options(dice_reduction):
    config = BaseConfig(
        model=ClassifierConfig(
            loss_func=ClassifierLossFctOption.selfadjdice, dice_reduction=dice_reduction
        ),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(trainer.loss_fct, SelfAdjDiceLoss)
    assert isinstance(result, float)


@pytest.mark.parametrize("dice_gamma", [0.0, 0.5, 1.0, 2.0])
def test_dice_loss_classifier_dice_gamma_values(dice_gamma):
    config = BaseConfig(
        model=ClassifierConfig(
            loss_func=ClassifierLossFctOption.selfadjdice, dice_gamma=dice_gamma
        ),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(trainer.loss_fct, SelfAdjDiceLoss)
    assert isinstance(result, float)


@pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
def test_classifier_batch_sizes(batch_size):
    config = BaseConfig(
        model=ClassifierConfig(),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    assert trainer.train_dataloader.batch_size == batch_size
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("lr", [1e-6, 1e-4, 1e-2])
def test_classifier_learning_rates(lr):
    config = BaseConfig(
        model=ClassifierConfig(),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        lr_base=lr,
    )
    trainer = ClassifierTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "wandb_config",
    [
        WandbConfig(project_name="test", use_watch=True),
        WandbConfig(project_name="test", use_watch=False),
        WandbConfig(project_name="test", watch_log_freq=500, watch_log_graph=False),
    ],
)
def test_classifier_wandb_configurations(wandb_config):
    config = BaseConfig(
        model=ClassifierConfig(),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        wandb=wandb_config,
    )
    trainer = ClassifierTrainer(config)
    assert isinstance(trainer.config.wandb, WandbConfig)
    assert trainer.config.wandb.enabled is True


@pytest.mark.parametrize("path", ["test_output", "custom/nested/path", "model_outputs/test"])
def test_classifier_output_paths(path):
    config = BaseConfig(
        model=ClassifierConfig(),
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        model_output_path=path,
    )
    trainer = ClassifierTrainer(config)
    assert path in trainer.model_save_dir
