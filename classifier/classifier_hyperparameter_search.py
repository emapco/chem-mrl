import math
import logging
import os
import gc

import apex
from apex.optimizers import FusedAdam
import transformers
from sentence_transformers import (
    models,
    SentenceTransformer,
)

import pandas as pd
import optuna
import wandb

from constants import (
    ISOMER_DESIGN_TRAIN_DS_PATH,
    ISOMER_DESIGN_VAL_DS_PATH,
    MODEL_NAMES,
    OUTPUT_MODEL_DIR,
)
from loss import SoftmaxLoss, SelfAdjDiceLoss
from evaluator import LabelAccuracyEvaluator
from load_data import load_data


def objective(
    trial: optuna.Trial,
) -> float:
    apex.torch.clear_autocast_cache()
    apex.torch.cuda.empty_cache()
    gc.collect()
    transformers.set_seed(42)

    # Load model components
    model_name = trial.suggest_categorical("model_name", list(MODEL_NAMES.values()))
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )

    # Generate param_config dictionary
    param_config = {
        "model_name": model_name,
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
            trial.suggest_categorical("matryoshka_dim", [768, 512, 256, 128, 64, 32])
            if "seyonec" not in model_name
            else 768
        ),
    }
    if param_config["loss_func"] == "SelfAdjDice":
        param_config.update(
            {
                "dice_reduction": trial.suggest_categorical(
                    "dice_reduction", ["sum", "mean"]
                ),
                "gamma": trial.suggest_float("gamma", 0.1, 1.0),
            }
        )

    # load data and evaluator
    train_dataloader, val_dataloader, _, num_labels = load_data(
        batch_size=param_config["train_batch_size"],
        train_file=ISOMER_DESIGN_TRAIN_DS_PATH,
        val_file=ISOMER_DESIGN_VAL_DS_PATH,
    )

    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        truncate_dim=param_config["matryoshka_dim"],
    )
    train_loss = get_loss_function(model, param_config, num_labels)

    val_evaluator = LabelAccuracyEvaluator(
        dataloader=val_dataloader,
        softmax_model=train_loss,
        write_csv=True,
    )

    # Calculate weight decay and learning rate
    total_number_training_points = (
        len(train_dataloader) * param_config["train_batch_size"]
    )
    weight_decay = 0.05 * math.sqrt(
        param_config["train_batch_size"]
        / (total_number_training_points * param_config["num_epochs"])
    )
    learning_rate = param_config["lr_base"] * math.sqrt(
        param_config["train_batch_size"]
    )

    model_save_path = get_model_save_path(param_config)
    warmup_steps = math.ceil(
        len(train_dataloader)
        * param_config["num_epochs"]
        * param_config["warmup_steps_percent"]
    )
    logging.info("Warmup-steps: {}".format(warmup_steps))

    wandb.init(
        project="chemberta-classification-hyperparameter-search-2025",
        config=param_config,
    )

    def wandb_callback(score, epoch, steps):
        eval_dict = {
            "score": score,
            "epoch": epoch,
            "steps": steps,
            **param_config,
        }
        wandb.log(eval_dict)

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
        use_amp=False,
        show_progress_bar=True,
        scheduler=param_config["scheduler"],
        checkpoint_path="output",
        callback=wandb_callback,
    )

    # Get final metric
    eval_file_path = os.path.join(
        model_save_path, "eval/accuracy_evaluation_results.csv"
    )
    eval_results_df = pd.read_csv(eval_file_path)
    metric = float(eval_results_df.iloc[-1]["accuracy"])
    return metric


def get_loss_function(model: SentenceTransformer, param_config, num_labels):
    if param_config["loss_func"] == "SoftMax":
        return SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension()
            or 768,
            num_labels=num_labels,
            dropout=param_config["dropout_p"],
        )
    else:
        return SelfAdjDiceLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension()
            or 768,
            num_labels=num_labels,
            reduction=param_config["dice_reduction"],
            dropout=param_config["dropout_p"],
            gamma=param_config["gamma"],
        )


def get_model_save_path(param_config):
    loss_parameter_str = (
        f"{param_config['dice_reduction']}-{param_config['gamma']}"
        if param_config["loss_func"] == "SelfAdjDice"
        else ""
    )

    model_save_path = os.path.join(
        OUTPUT_MODEL_DIR,
        "classifier",
        f"{param_config['model_name'].rsplit('/', 1)[1][:20]}"
        f"-{param_config['train_batch_size']}"
        f"-{param_config['num_epochs']}"
        f"-{param_config['lr_base']:6f}"
        f"-{param_config['scheduler']}-{param_config['warmup_steps_percent']:3f}"
        f"-{param_config['loss_func']}-{param_config['dropout_p']:3f}"
        f"-{param_config['matryoshka_dim']}-{loss_parameter_str}",  # noqa: E501,
    )
    print(f"\n{model_save_path}\n")
    return model_save_path


def generate_hyperparameters():
    study = optuna.create_study(
        study_name="chem-mrl-classification-hyperparameter-tuning",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=1024,  # 512, 768, 1536
        gc_after_trial=True,
        show_progress_bar=True,
    )

    print("Best hyperparameters found:")
    print(study.best_params)
    print("Best best trials:")
    print(study.best_trials)
    print("Best trial:")
    print(study.best_trial)
    study.trials_dataframe().to_csv(
        "chem-mrl-classification-hyperparameter-tuning.csv", index=False
    )


if __name__ == "__main__":
    generate_hyperparameters()
