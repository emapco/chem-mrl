import logging

import optuna

from chem_mrl.configs.Base import _scheduler_options
from chem_mrl.configs.MRL import (
    Chem2dMRLConfig,
    ChemMRLConfig,
    _tanimoto_loss_func_options,
    _tanimoto_similarity_base_loss_func_options,
)
from chem_mrl.constants import BASE_MODEL_NAME, CHEM_MRL_DATASET_KEYS, OPTUNA_DB_URI
from chem_mrl.trainers import ChemMRLTrainer, ExecutableTrainer

logger = logging.getLogger(__name__)
PROJECT_NAME = "chem-mrl-hyperparameter-tuning-2025"


def objective(
    trial: optuna.Trial,
) -> float:
    config_params = {
        "model_name": BASE_MODEL_NAME,
        "dataset_key": trial.suggest_categorical("dataset_key", CHEM_MRL_DATASET_KEYS),
        "train_batch_size": 24,
        "num_epochs": 5,
        "lr_base": 1.1190785944700813e-05,
        "scheduler": trial.suggest_categorical("scheduler", _scheduler_options),
        "warmup_steps_percent": trial.suggest_categorical(
            "warmup_steps_percent", [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
        ),
        "loss_func": trial.suggest_categorical(
            "loss_func", _tanimoto_loss_func_options
        ),
        "use_2d_matryoshka": trial.suggest_categorical(
            "use_2d_matryoshka", [True, False]
        ),
    }

    # Add tanimoto similarity loss function if needed
    if config_params["loss_func"] == "tanimotosimilarityloss":
        config_params["tanimoto_similarity_loss_func"] = trial.suggest_categorical(
            "tanimoto_similarity_loss_func", _tanimoto_similarity_base_loss_func_options
        )

    # Create appropriate config based on matryoshka setting
    if config_params["use_2d_matryoshka"]:
        config = Chem2dMRLConfig(**config_params)
    else:
        config = ChemMRLConfig(**config_params)

    trainer = ChemMRLTrainer(config)
    executable_trainer = ExecutableTrainer(config, trainer=trainer, return_metric=True)
    metric = executable_trainer.execute()

    return metric


if __name__ == "__main__":
    """Use this to generate hyperparameters to then be manually trained on using working training code."""
    study = optuna.create_study(
        storage=optuna.storages.RDBStorage(
            url=OPTUNA_DB_URI,
            heartbeat_interval=10,
            engine_kwargs={
                "pool_size": 20,
                "connect_args": {"keepalives": 1},
            },
        ),
        study_name=PROJECT_NAME,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=2),
    )
    study.optimize(
        objective,
        n_trials=256,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    logger.info("Best hyperparameters found:")
    logger.info(study.best_params)
    logger.info("Best best trials:")
    logger.info(study.best_trials)
    study.trials_dataframe().to_csv("chem-mrl-hyperparameter-tuning.csv", index=False)
