import pytest
from constants import TEST_CHEM_MRL_PATH

from chem_mrl.constants import BASE_MODEL_HIDDEN_DIM
from chem_mrl.schemas import (
    BaseConfig,
    ChemMRLConfig,
    LatentAttentionConfig,
    WandbConfig,
)
from chem_mrl.schemas.Enums import (
    ChemMrlEvalMetricOption,
    ChemMrlLossFctOption,
    EmbeddingPoolingOption,
    EvalSimilarityFctOption,
    SchedulerOption,
    TanimotoSimilarityBaseLossFctOption,
    WatchLogOption,
)
from chem_mrl.tokenizers import QuerySmilesTokenizerFast
from chem_mrl.trainers import ChemMRLTrainer, TempDirTrainerExecutor


def test_chem_mrl_trainer_instantiation():
    config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    assert isinstance(trainer, ChemMRLTrainer)
    assert isinstance(trainer.config, BaseConfig)
    assert isinstance(trainer.config.model, ChemMRLConfig)


def test_chem_mrl_test_evaluator():
    config = BaseConfig(
        model=ChemMRLConfig(n_dims_per_step=4),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        test_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("weight_decay", [0.0, 1e-8, 1e-4, 1e-2, 0.1])
def test_chem_mrl_test_weight_decay(weight_decay):
    config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        weight_decay=weight_decay,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("scheduler", SchedulerOption)
def test_chem_mrl_scheduler_options(
    scheduler,
):
    config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        scheduler=scheduler,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("pooling", EmbeddingPoolingOption)
def test_chem_mrl_pooling_options(pooling):
    config = BaseConfig(
        model=ChemMRLConfig(embedding_pooling=pooling),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
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
def test_chem_mrl_loss_functions(loss_func):
    # can't test tanimotosimilarityloss since it requires an additional parameter
    config = BaseConfig(
        model=ChemMRLConfig(
            loss_func=loss_func,
        ),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("base_loss", TanimotoSimilarityBaseLossFctOption)
def test_chem_mrl_tanimoto_similarity_loss(base_loss):
    config = BaseConfig(
        model=ChemMRLConfig(
            loss_func=ChemMrlLossFctOption.tanimotosimilarityloss,
            tanimoto_similarity_loss_func=base_loss,
        ),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("eval_similarity_fct", EvalSimilarityFctOption)
def test_chem_mrl_eval_similarity(eval_similarity_fct):
    config = BaseConfig(
        model=ChemMRLConfig(
            eval_similarity_fct=eval_similarity_fct,
        ),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("eval_metric", ChemMrlEvalMetricOption)
def test_chem_mrl_eval_metrics(eval_metric):
    config = BaseConfig(
        model=ChemMRLConfig(eval_metric=eval_metric),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


def test_chem_2d_mrl_trainer_instantiation():
    config = BaseConfig(
        model=ChemMRLConfig(
            use_2d_matryoshka=True,
        ),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    assert isinstance(trainer, ChemMRLTrainer)
    assert isinstance(trainer.config.model, ChemMRLConfig)
    assert trainer.config.model.use_2d_matryoshka is True


def test_query_chem_mrl_trainer():
    config = BaseConfig(
        model=ChemMRLConfig(
            use_query_tokenizer=True,
        ),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)
    assert isinstance(trainer, ChemMRLTrainer)
    assert isinstance(trainer.config.model, ChemMRLConfig)
    assert trainer.config.model.use_query_tokenizer is True
    assert isinstance(trainer.model.tokenizer, QuerySmilesTokenizerFast)
    assert trainer.model.get_max_seq_length() == trainer.model.tokenizer.model_max_length
    assert trainer.model.max_seq_length == trainer.model.tokenizer.model_max_length


def test_mrl_dimension_weights_validation():
    with pytest.raises(ValueError, match="Dimension weights must be in increasing order"):
        config = BaseConfig(
            model=ChemMRLConfig(mrl_dimension_weights=(2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)),
            train_dataset_path=TEST_CHEM_MRL_PATH,
            val_dataset_path=TEST_CHEM_MRL_PATH,
        )
        ChemMRLTrainer(config)


def test_mrl_latent_attention_layer():
    config = BaseConfig(
        model=ChemMRLConfig(
            latent_attention_config=LatentAttentionConfig(
                BASE_MODEL_HIDDEN_DIM,
            ),
            use_2d_matryoshka=True,
            last_layer_weight=2.0,
            prior_layers_weight=1.0,
        ),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


def test_2d_mrl_layer_weights():
    config = BaseConfig(
        model=ChemMRLConfig(
            use_2d_matryoshka=True,
            last_layer_weight=2.0,
            prior_layers_weight=1.0,
        ),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
def test_chem_mrl_batch_sizes(batch_size):
    config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    assert trainer.train_dataloader.batch_size == batch_size
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize("lr", [1e-6, 1e-4, 1e-2])
def test_chem_mrl_learning_rates(lr):
    config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        lr_base=lr,
    )
    trainer = ChemMRLTrainer(config)
    executor = TempDirTrainerExecutor(trainer)
    result = executor.execute()
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "wandb_config",
    [
        WandbConfig(project_name="test", use_watch=True, watch_log=WatchLogOption.gradients),
        WandbConfig(project_name="test", use_watch=False),
        WandbConfig(project_name="test", watch_log_freq=500, watch_log_graph=False),
    ],
)
def test_chem_mrl_wandb_configurations(wandb_config):
    config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        wandb=wandb_config,
    )
    trainer = ChemMRLTrainer(config)
    assert isinstance(trainer.config.wandb, WandbConfig)
    assert trainer.config.wandb.enabled is True


@pytest.mark.parametrize("path", ["test_output", "custom/nested/path", "model_outputs/test"])
def test_classifier_output_paths(path):
    config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        model_output_path=path,
    )
    trainer = ChemMRLTrainer(config)
    assert path in trainer.model_save_dir
