import logging

import optuna
from constants import TRAIN_ISOMER_DESIGN_DS_PATH, VAL_ISOMER_DESIGN_DS_PATH

from chem_mrl.configs.Base import _scheduler_options
from chem_mrl.configs.Classifier import (
    ClassifierConfig,
    DiceLossClassifierConfig,
    _classifier_loss_func_options,
    _dice_reduction_options,
)
from chem_mrl.constants import (
    BASE_MODEL_NAME,
    CHEM_MRL_DIMENSIONS,
    MODEL_NAMES,
    OPTUNA_DB_URI,
)
from chem_mrl.trainers import ClassifierTrainer, ExecutableTrainer

logger = logging.getLogger(__name__)
PROJECT_NAME = "chem-mrl-classification-hyperparameter-search-2025"


def objective(
    trial: optuna.Trial,
) -> float:
    model_name = trial.suggest_categorical("model_name", list(MODEL_NAMES.values()))
    loss_func = trial.suggest_categorical("loss_func", _classifier_loss_func_options)

    config_params = {
        "model_name": model_name,
        "train_batch_size": int(
            trial.suggest_float("train_batch_size", 32, 1024, step=32)
        ),
        "num_epochs": trial.suggest_int("num_epochs", 1, 3),
        "lr_base": trial.suggest_float("lr_base", 2.0e-06, 5.6e-06),
        "scheduler": trial.suggest_categorical("scheduler", _scheduler_options),
        "warmup_steps_percent": trial.suggest_float("warmup_steps_percent", 0.0, 0.06),
        "loss_func": loss_func,
        "dropout_p": trial.suggest_categorical(
            "dropout_p", [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        ),
        "classifier_hidden_dimension": 768,
        "freeze_model": trial.suggest_categorical("freeze_model", [True, False]),
        "train_dataset_path": TRAIN_ISOMER_DESIGN_DS_PATH,
        "val_dataset_path": VAL_ISOMER_DESIGN_DS_PATH,
    }
    if "seyonec" != BASE_MODEL_NAME:
        config_params.update(
            {
                "classifier_hidden_dimension": trial.suggest_categorical(
                    "classifier_hidden_dimension", CHEM_MRL_DIMENSIONS
                )
            }
        )
    if loss_func == "selfadjdice":
        config_params.update(
            {
                "dice_reduction": trial.suggest_categorical(
                    "dice_reduction", _dice_reduction_options
                ),
                "dice_gamma": trial.suggest_float("dice_gamma", 0.1, 1.0),
            }
        )
        config = DiceLossClassifierConfig(**config_params)
    else:
        config = ClassifierConfig(**config_params)

    # Set up trainer and executor
    trainer = ClassifierTrainer(config)
    executable_trainer = ExecutableTrainer(config, trainer=trainer, return_metric=True)
    metric = executable_trainer.execute()
    return metric


def generate_hyperparameters():
    study = optuna.create_study(
        storage=OPTUNA_DB_URI,
        study_name=PROJECT_NAME,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=2),
    )
    study.optimize(
        objective,
        n_trials=512,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    logger.info("Best hyperparameters found:")
    logger.info(study.best_params)
    logger.info("Best best trials:")
    logger.info(study.best_trials)
    study.trials_dataframe().to_csv(
        "chem-mrl-classification-hyperparameter-tuning.csv", index=False
    )


if __name__ == "__main__":
    generate_hyperparameters()
