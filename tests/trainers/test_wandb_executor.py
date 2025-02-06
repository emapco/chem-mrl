from constants import TEST_CHEM_MRL_PATH, TEST_CLASSIFICATION_PATH

from chem_mrl.schemas import BaseConfig, ChemMRLConfig, ClassifierConfig, WandbConfig
from chem_mrl.trainers import ChemMRLTrainer, ClassifierTrainer, WandBTrainerExecutor


def test_chem_mrl_wandb_executor():
    config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path=TEST_CHEM_MRL_PATH,
        val_dataset_path=TEST_CHEM_MRL_PATH,
        wandb=WandbConfig(
            project_name="chem_mrl_test",
            run_name="test",
            use_watch=True,
            watch_log_freq=1000,
            watch_log_graph=True,
        ),
    )
    trainer = ChemMRLTrainer(config)
    executor = WandBTrainerExecutor(trainer=trainer)
    assert isinstance(executor, WandBTrainerExecutor)
    assert isinstance(executor.trainer, ChemMRLTrainer)
    assert isinstance(executor.trainer.config, BaseConfig)
    assert isinstance(executor.trainer.config.model, ChemMRLConfig)


def test_classifier_wandb_executor():
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
    classifier = ClassifierTrainer(config)
    executor = WandBTrainerExecutor(trainer=classifier)
    assert isinstance(executor, WandBTrainerExecutor)
    assert isinstance(executor.trainer, ClassifierTrainer)
    assert isinstance(executor.trainer.config, BaseConfig)
    assert isinstance(executor.trainer.config.model, ClassifierConfig)
