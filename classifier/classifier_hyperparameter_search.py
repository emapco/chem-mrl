import logging
import math
import os

import optuna
import pandas as pd
import transformers
from apex.optimizers import FusedAdam
from constants import (
    CHEM_MRL_DIMENSIONS,
    MODEL_NAMES,
    TRAIN_ISOMER_DESIGN_DS_PATH,
    VAL_ISOMER_DESIGN_DS_PATH,
)
from evaluator import LabelAccuracyEvaluator
from load_data import load_data
from sentence_transformers import SentenceTransformer, models
from utils import get_model_save_path, get_signed_in_wandb_callback, get_train_loss

import wandb

logger = logging.getLogger(__name__)
PROJECT_NAME = "chem-mrl-classification-hyperparameter-search-2025"


def objective(
    trial: optuna.Trial,
) -> float:
    model_name = trial.suggest_categorical("model_name", list(MODEL_NAMES.values()))
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )

    # Generate param_config dictionary
    param_config = {
        "model_name": model_name,
        "seed": 42,
        "train_batch_size": int(
            trial.suggest_float("train_batch_size", 32, 1024, step=32)
        ),
        # observation: larger num of epochs models prefer warmupcosinewithhardrestarts scheduler
        "num_epochs": trial.suggest_int("num_epochs", 1, 3),
        "lr_base": trial.suggest_float(
            "lr_base", 2.0e-06, 5.6e-06
        ),  # 3.4038386108141304e-06
        "scheduler": trial.suggest_categorical(
            "scheduler",
            [
                "warmuplinear",
                "warmupcosine",
                "warmupcosinewithhardrestarts",
            ],  # warmupcosinewithhardrestarts
        ),
        "warmup_steps_percent": trial.suggest_float("warmup_steps_percent", 0.0, 0.06),
        "loss_func": trial.suggest_categorical("loss_func", ["SelfAdjDice", "SoftMax"]),
        "dropout_p": trial.suggest_categorical(
            "dropout_p", [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        ),
        "matryoshka_dim": (
            trial.suggest_categorical("matryoshka_dim", CHEM_MRL_DIMENSIONS)
            if "seyonec" not in model_name
            else 768
        ),
        "freeze_model": trial.suggest_categorical("freeze_model", [True, False]),
    }
    if param_config["loss_func"] == "SelfAdjDice":
        param_config.update(
            {
                "dice_reduction": trial.suggest_categorical(
                    "dice_reduction", ["sum", "mean"]
                ),
                "dice_gamma": trial.suggest_float("dice_gamma", 0.1, 1.0),
            }
        )

    transformers.set_seed(param_config["seed"])
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        truncate_dim=param_config["matryoshka_dim"],
    )

    train_dataloader, val_dataloader, _, num_labels = load_data(
        batch_size=param_config["train_batch_size"],
        train_file=TRAIN_ISOMER_DESIGN_DS_PATH,
        val_file=VAL_ISOMER_DESIGN_DS_PATH,
    )

    train_loss = get_train_loss(
        model=model,
        smiles_embedding_dimension=param_config["matryoshka_dim"],
        num_labels=num_labels,
        loss_func=param_config["loss_func"],
        dropout=param_config["dropout_p"],
        freeze_base_model=param_config["freeze_model"],
        dice_reduction=param_config.get("dice_reduction"),
        dice_gamma=param_config.get("dice_gamma"),
    )

    val_evaluator = LabelAccuracyEvaluator(
        dataloader=val_dataloader,
        softmax_model=train_loss,
        write_csv=True,
    )

    # Calculate training parameters
    total_number_training_points = (
        len(train_dataloader) * param_config["train_batch_size"]
    )
    # normalized weight decay for adamw optimizer - https://arxiv.org/pdf/1711.05101.pdf
    # optimized hyperparameter lambda_norm = 0.05 for AdamW optimizer
    weight_decay = 0.05 * math.sqrt(
        param_config["train_batch_size"]
        / (total_number_training_points * param_config["num_epochs"])
    )
    # scale learning rate based on sqrt of the batch size
    learning_rate = param_config["lr_base"] * math.sqrt(
        param_config["train_batch_size"]
    )
    warmup_steps = math.ceil(
        len(train_dataloader)
        * param_config["num_epochs"]
        * param_config["warmup_steps_percent"]
    )

    model_save_path = get_model_save_path(param_config)
    wandb_callback = get_signed_in_wandb_callback(train_dataloader, trial=trial)

    with wandb.init(
        project=PROJECT_NAME,
        config=param_config,
    ):
        wandb.watch(model, log="all", log_graph=True)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=val_evaluator,
            epochs=param_config["num_epochs"],
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_class=FusedAdam,
            optimizer_params={
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "adam_w_mode": True,
            },
            save_best_model=False,
            use_amp=False,
            show_progress_bar=True,
            scheduler=param_config["scheduler"],
            checkpoint_path="output",
            checkpoint_save_steps=1000000,
            checkpoint_save_total_limit=20,
            callback=wandb_callback,
        )

        # Get final metric
        eval_file_path = os.path.join(
            model_save_path, "eval/accuracy_evaluation_results.csv"
        )
        eval_results_df = pd.read_csv(eval_file_path)
        metric = float(eval_results_df.iloc[-1]["accuracy"])
        return metric
    return -1


def generate_hyperparameters():
    study = optuna.create_study(
        storage="postgresql://postgres:password@192.168.0.8:5432/postgres",
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
