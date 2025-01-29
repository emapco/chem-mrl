import pytest
from constants import TEST_CLASSIFICATION_PATH

from chem_mrl.configs import ClassifierConfig, DiceLossClassifierConfig, WandbConfig
from chem_mrl.configs.types import (
    CLASSIFIER_EVAL_METRIC_OPTIONS,
    DICE_REDUCTION_OPTIONS,
    SCHEDULER_OPTIONS,
)
from chem_mrl.constants import BASE_MODEL_NAME, CHEM_MRL_DIMENSIONS
from chem_mrl.trainers import (
    CallbackTrainerExecutor,
    ClassifierTrainer,
    WandBTrainerExecutor,
)


def test_classifier_trainer_instantiation():
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        use_wandb=True,
        wandb_config=WandbConfig(
            project_name="chem_mrl_test",
            run_name="test",
            use_watch=True,
            watch_log="all",
            watch_log_freq=1000,
            watch_log_graph=True,
        ),
    )
    classifier = ClassifierTrainer(config)
    executor = WandBTrainerExecutor(trainer=classifier)
    assert isinstance(executor, WandBTrainerExecutor)
    assert isinstance(executor.trainer, ClassifierTrainer)
    assert isinstance(executor.trainer.config, ClassifierConfig)


@pytest.mark.parametrize("scheduler", SCHEDULER_OPTIONS)
def test_classifier_scheduler_options(
    scheduler,
):
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        scheduler=scheduler,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert isinstance(result, float)
    assert float != -1.0

    config = DiceLossClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        scheduler=scheduler,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert isinstance(result, float)
    assert float != -1.0


@pytest.mark.parametrize("dimension", CHEM_MRL_DIMENSIONS)
def test_classifier_classifier_hidden_dimensions(
    dimension,
):
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        classifier_hidden_dimension=dimension,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert trainer.model.truncate_dim == dimension
    assert trainer.loss_fct.smiles_embedding_dimension == dimension
    assert isinstance(result, float)
    assert float != -1.0

    config = DiceLossClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        classifier_hidden_dimension=dimension,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert trainer.model.truncate_dim == dimension
    assert trainer.loss_fct.smiles_embedding_dimension == dimension
    assert isinstance(result, float)
    assert float != -1.0


@pytest.mark.parametrize("eval_metric", CLASSIFIER_EVAL_METRIC_OPTIONS)
def test_classifier_eval_metrics(eval_metric):
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        eval_metric=eval_metric,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert isinstance(result, float)
    assert float != -1.0

    config = DiceLossClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        eval_metric=eval_metric,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert isinstance(result, float)
    assert float != -1.0


def test_classifier_freeze_internal_model():
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        freeze_model=True,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert trainer.loss_fct.freeze_model is True
    assert isinstance(result, float)
    assert float != -1.0

    config = DiceLossClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        freeze_model=True,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert trainer.loss_fct.freeze_model is True
    assert isinstance(result, float)
    assert float != -1.0


def test_classifier_num_labels():
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        freeze_model=True,
    )
    trainer = ClassifierTrainer(config)
    assert trainer.loss_fct.num_labels == 2  # testing dataset only has two classes

    config = DiceLossClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        freeze_model=True,
    )
    trainer = ClassifierTrainer(config)
    assert trainer.loss_fct.num_labels == 2  # testing dataset only has two classes


@pytest.mark.parametrize("dropout_p", [0.0, 0.1, 0.5, 1.0])
def test_classifier_dropout(dropout_p):
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        dropout_p=dropout_p,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)

    assert trainer.loss_fct.dropout_p == dropout_p
    assert isinstance(result, float)
    assert float != -1.0
    config = DiceLossClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        dropout_p=dropout_p,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert trainer.loss_fct.dropout_p == dropout_p
    assert isinstance(result, float)
    assert float != -1.0


def test_dice_loss_classifier_trainer_instantiation():
    config = DiceLossClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        use_wandb=True,
        wandb_config=WandbConfig(
            project_name="chem_mrl_test",
            run_name="test",
            use_watch=True,
            watch_log="all",
            watch_log_freq=1000,
            watch_log_graph=True,
        ),
    )
    classifier = ClassifierTrainer(config)
    executor = WandBTrainerExecutor(trainer=classifier)
    assert isinstance(executor, WandBTrainerExecutor)
    assert isinstance(executor.trainer, ClassifierTrainer)
    assert isinstance(executor.trainer.config, DiceLossClassifierConfig)


@pytest.mark.parametrize("dice_reduction", DICE_REDUCTION_OPTIONS)
def test_dice_loss_classifier_dice_reduction_options(dice_reduction):
    config = DiceLossClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        loss_func="selfadjdice",
        dice_reduction=dice_reduction,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert isinstance(result, float)
    assert float != -1.0


@pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
def test_classifier_batch_sizes(batch_size):
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        train_batch_size=batch_size,
    )
    trainer = ClassifierTrainer(config)
    assert trainer.train_dataloader.batch_size == batch_size
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert isinstance(result, float)
    assert result != -1.0


@pytest.mark.parametrize("lr", [1e-6, 1e-4, 1e-2])
def test_classifier_learning_rates(lr):
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        lr_base=lr,
    )
    trainer = ClassifierTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute(return_eval_metric=True)
    assert isinstance(result, float)
    assert result != -1.0


@pytest.mark.parametrize(
    "wandb_config",
    [
        WandbConfig(project_name="test", use_watch=True, watch_log="gradients"),
        WandbConfig(project_name="test", use_watch=False),
        WandbConfig(project_name="test", watch_log_freq=500, watch_log_graph=False),
    ],
)
def test_classifier_wandb_configurations(wandb_config):
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        use_wandb=True,
        wandb_config=wandb_config,
    )
    trainer = ClassifierTrainer(config)
    assert isinstance(trainer.config.wandb_config, WandbConfig)
    assert trainer.config.use_wandb is True


@pytest.mark.parametrize(
    "path", ["test_output", "custom/nested/path", "model_outputs/test"]
)
def test_classifier_output_paths(path):
    config = ClassifierConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CLASSIFICATION_PATH,
        val_dataset_path=TEST_CLASSIFICATION_PATH,
        model_output_path=path,
    )
    trainer = ClassifierTrainer(config)
    assert path in trainer.model_save_dir_name
