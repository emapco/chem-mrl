import pytest
from constants import TEST_CHEM_MRL_PATH

from chem_mrl.configs import Chem2dMRLConfig, ChemMRLConfig, WandbConfig
from chem_mrl.configs.types import (
    CHEM_MRL_EVAL_METRIC_OPTIONS,
    EVAL_SIMILARITY_FCT_OPTIONS,
    SCHEDULER_OPTIONS,
    TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS,
)
from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.trainers import (
    CallbackTrainerExecutor,
    ChemMRLTrainer,
    WandBTrainerExecutor,
)


def test_chem_mrl_trainer_instantiation():
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
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
    chem_mrl = ChemMRLTrainer(config)
    executor = WandBTrainerExecutor(trainer=chem_mrl)
    assert isinstance(executor, WandBTrainerExecutor)
    assert isinstance(executor.trainer, ChemMRLTrainer)
    assert isinstance(executor.trainer.config, ChemMRLConfig)


def test_chem_mrl_test_evaluator():
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        test_dataset_path=TEST_CHEM_MRL_PATH,
        n_dims_per_step=4,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0

    config = Chem2dMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        test_dataset_path=TEST_CHEM_MRL_PATH,
        n_layers_per_step=2,
        kl_div_weight=0.7,
        kl_temperature=0.5,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0


@pytest.mark.parametrize("scheduler", SCHEDULER_OPTIONS)
def test_chem_mrl_scheduler_options(
    scheduler,
):
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        scheduler=scheduler,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0

    config = Chem2dMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        scheduler=scheduler,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0


@pytest.mark.parametrize(
    "loss_func",
    [
        "tanimotosentloss",
        "cosentloss",
    ],
)
def test_chem_mrl_loss_functions(loss_func):
    # can't test tanimotosimilarityloss since it requires an additional parameter
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        loss_func=loss_func,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0

    # can't test tanimotosimilarityloss since it requires an additional parameter
    config = Chem2dMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        loss_func=loss_func,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0


@pytest.mark.parametrize("base_loss", TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS)
def test_chem_mrl_tanimoto_similarity_loss(base_loss):
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        loss_func="tanimotosimilarityloss",
        tanimoto_similarity_loss_func=base_loss,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0

    config = Chem2dMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        loss_func="tanimotosimilarityloss",
        tanimoto_similarity_loss_func=base_loss,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0


@pytest.mark.parametrize("eval_similarity", EVAL_SIMILARITY_FCT_OPTIONS)
def test_chem_mrl_eval_similarity(eval_similarity):
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        eval_similarity_fct=eval_similarity,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0

    config = Chem2dMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        eval_similarity_fct=eval_similarity,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0


@pytest.mark.parametrize("eval_metric", CHEM_MRL_EVAL_METRIC_OPTIONS)
def test_chem_mrl_eval_metrics(eval_metric):
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        eval_metric=eval_metric,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0

    config = Chem2dMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        eval_metric=eval_metric,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0


def test_chem_2d_mrl_trainer_instantiation():
    config = Chem2dMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
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
    chem_2d_mrl = ChemMRLTrainer(config)
    executor = WandBTrainerExecutor(trainer=chem_2d_mrl)
    assert isinstance(executor, WandBTrainerExecutor)
    assert isinstance(executor.trainer, ChemMRLTrainer)
    assert isinstance(executor.trainer.config, Chem2dMRLConfig)


def test_mrl_dimension_weights_validation():
    with pytest.raises(
        ValueError, match="Dimension weights must be in increasing order"
    ):
        config = ChemMRLConfig(
            model_name=BASE_MODEL_NAME,
            train_dataset_path=TEST_CHEM_MRL_PATH,
            val_dataset_path=TEST_CHEM_MRL_PATH,
            mrl_dimension_weights=(2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        )
        ChemMRLTrainer(config)
    with pytest.raises(
        ValueError, match="Dimension weights must be in increasing order"
    ):
        config = Chem2dMRLConfig(
            model_name=BASE_MODEL_NAME,
            train_dataset_path=TEST_CHEM_MRL_PATH,
            val_dataset_path=TEST_CHEM_MRL_PATH,
            mrl_dimension_weights=(2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        )
        ChemMRLTrainer(config)


def test_2d_mrl_layer_weights():
    config = Chem2dMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        last_layer_weight=2.0,
        prior_layers_weight=1.0,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0


@pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
def test_chem_mrl_batch_sizes(batch_size):
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        train_batch_size=batch_size,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    assert trainer.train_dataloader.batch_size == batch_size
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert result != -1.0


@pytest.mark.parametrize("lr", [1e-6, 1e-4, 1e-2])
def test_chem_mrl_learning_rates(lr):
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        lr_base=lr,
        return_eval_metric=True,
    )
    trainer = ChemMRLTrainer(config)
    executor = CallbackTrainerExecutor(trainer=trainer)
    result = executor.execute()
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
def test_chem_mrl_wandb_configurations(wandb_config):
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        use_wandb=True,
        wandb_config=wandb_config,
    )
    trainer = ChemMRLTrainer(config)
    assert isinstance(trainer.config.wandb_config, WandbConfig)
    assert trainer.config.use_wandb is True


@pytest.mark.parametrize(
    "path", ["test_output", "custom/nested/path", "model_outputs/test"]
)
def test_classifier_output_paths(path):
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        model_output_path=path,
    )
    trainer = ChemMRLTrainer(config)
    assert path in trainer.model_save_dir_name
