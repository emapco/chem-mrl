import os

from chem_mrl.configs import (
    Chem2dMRLConfig,
    ChemMRLConfig,
    ClassifierConfig,
    DiceLossClassifierConfig,
    WandbConfig,
)
from chem_mrl.trainers import ChemMRLTrainer, ClassifierTrainer, ExecutableTrainer

_const_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.dirname(_const_file_dir)
_data_dir = os.path.join(_project_root_dir, "data", "chem")


def test_chem_mrl_trainer_instantiation():
    config = ChemMRLConfig(
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
    trainer = ExecutableTrainer(config, trainer=chem_mrl, return_metric=True)
    assert isinstance(trainer, ExecutableTrainer)
    assert isinstance(trainer._trainer, ChemMRLTrainer)


def test_chem_2d_mrl_trainer_instantiation():
    config = Chem2dMRLConfig(
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
    trainer = ExecutableTrainer(config, trainer=chem_2d_mrl, return_metric=True)
    assert isinstance(trainer, ExecutableTrainer)
    assert isinstance(trainer._trainer, ChemMRLTrainer)


def test_classifier_trainer_instantiation():
    config = ClassifierConfig(
        use_wandb=True,
        wandb_config=WandbConfig(
            project_name="chem_mrl_test",
            run_name="test",
            use_watch=True,
            watch_log="all",
            watch_log_freq=1000,
            watch_log_graph=True,
        ),
        train_dataset_path=os.path.join(
            _data_dir, "isomer_design", "train_isomer_design.parquet"
        ),
        val_dataset_path=os.path.join(
            _data_dir, "isomer_design", "val_isomer_design.parquet"
        ),
    )
    classifier = ClassifierTrainer(config)
    trainer = ExecutableTrainer(config, trainer=classifier, return_metric=True)
    assert isinstance(trainer, ExecutableTrainer)
    assert isinstance(trainer._trainer, ClassifierTrainer)


def test_dice_loss_classifier_trainer_instantiation():
    config = DiceLossClassifierConfig(
        use_wandb=True,
        wandb_config=WandbConfig(
            project_name="chem_mrl_test",
            run_name="test",
            use_watch=True,
            watch_log="all",
            watch_log_freq=1000,
            watch_log_graph=True,
        ),
        train_dataset_path=os.path.join(
            _data_dir, "isomer_design", "train_isomer_design.parquet"
        ),
        val_dataset_path=os.path.join(
            _data_dir, "isomer_design", "val_isomer_design.parquet"
        ),
    )
    classifier = ClassifierTrainer(config)
    trainer = ExecutableTrainer(config, trainer=classifier, return_metric=True)
    assert isinstance(trainer, ExecutableTrainer)
    assert isinstance(trainer._trainer, ClassifierTrainer)
