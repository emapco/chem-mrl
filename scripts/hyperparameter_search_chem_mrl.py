import logging
import os

import optuna

from chem_mrl.configs.BaseConfig import SCHEDULER_OPTIONS, WandbConfig
from chem_mrl.configs.MrlConfig import (
    CHEM_MRL_LOSS_FCT_OPTIONS,
    TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS,
    Chem2dMRLConfig,
    ChemMRLConfig,
)
from chem_mrl.constants import (
    BASE_MODEL_NAME,
    CHEM_MRL_DATASET_KEYS,
    OPTUNA_DB_URI,
    TRAIN_DS_DICT,
    VAL_DS_DICT,
)
from chem_mrl.trainers import ChemMRLTrainer, WandBTrainerExecutor

logger = logging.getLogger(__name__)
PROJECT_NAME = "chem-mrl-hyperparameter-tuning-2025"


def objective(
    trial: optuna.Trial,
) -> float:
    dataset_key = trial.suggest_categorical("dataset_key", CHEM_MRL_DATASET_KEYS)
    loss_func = trial.suggest_categorical("loss_func", CHEM_MRL_LOSS_FCT_OPTIONS)
    config_params = {
        "model_name": BASE_MODEL_NAME,
        "train_dataset_path": TRAIN_DS_DICT[dataset_key],
        "val_dataset_path": VAL_DS_DICT[dataset_key],
        "n_train_samples": 5000,
        "n_val_samples": 1500,
        "train_batch_size": 24,
        "num_epochs": 5,
        "lr_base": 1.1190785944700813e-05,
        "scheduler": trial.suggest_categorical("scheduler", SCHEDULER_OPTIONS),
        "warmup_steps_percent": trial.suggest_categorical(
            "warmup_steps_percent", [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
        ),
        "loss_func": loss_func,
        "use_2d_matryoshka": trial.suggest_categorical(
            "use_2d_matryoshka", [True, False]
        ),
        "return_eval_metric": True,
        "use_wandb": True,
        "wandb_config": WandbConfig(
            project_name=PROJECT_NAME,
            use_watch=True,
            watch_log_graph=True,
        ),
    }
    # Add tanimoto similarity loss function if needed
    if loss_func == "tanimotosimilarityloss":
        config_params["tanimoto_similarity_loss_func"] = trial.suggest_categorical(
            "tanimoto_similarity_loss_func", TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS
        )

    if config_params["use_2d_matryoshka"]:
        config = Chem2dMRLConfig(**config_params)
    else:
        config = ChemMRLConfig(**config_params)

    executable_trainer = WandBTrainerExecutor(
        trainer=ChemMRLTrainer(config), optuna_trial=trial
    )
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
