import gc
import logging
import math
import os

import optuna
import pandas as pd
import transformers
from apex.optimizers import FusedAdam
from constants import TRAIN_DS_DICT, VAL_DS_DICT
from evaluator import EmbeddingSimilarityEvaluator, SimilarityFunction
from load_data import load_data
from sentence_transformers import SentenceTransformer, models
from utils import (
    get_base_loss,
    get_model_save_path,
    get_signed_in_wandb_callback,
    get_train_loss,
)

import wandb

logger = logging.getLogger(__name__)
PROJECT_NAME = "chem-mrl-hyperparameter-tuning-2025"


def objective(
    trial: optuna.Trial,
) -> float:
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # generate hyperparameters
    loss_func = trial.suggest_categorical(
        "loss_func", ["tanimotoloss", "tanimotosimilarityloss", "cosentloss"]
    )
    param_config = {
        "model_name": model_name,
        "dataset_key": trial.suggest_categorical(
            "dataset_key", list(TRAIN_DS_DICT.keys())
        ),
        "seed": 42,
        "train_batch_size": 64,
        # num-epochs: 2-3 likely based on chemberta hyperparameter search on wandb
        # https://wandb.ai/seyonec/huggingface/reports/seyonec-s-ChemBERTa-update-08-31--VmlldzoyMjM1NDY
        "num_epochs": 5,
        # scheduler parameters
        # "lr_base": trial.suggest_float(
        #     "lr_base", 5.0e-06, 1.0e-04
        # ),  # 1.1190785944700813e-05
        "lr_base": 1.1190785944700813e-05,
        "scheduler": trial.suggest_categorical(
            "scheduler",
            [
                "warmupconstant",
                "warmuplinear",
                "warmupcosine",
                "warmupcosinewithhardrestarts",
            ],
        ),
        "warmup_steps_percent": trial.suggest_categorical(
            "warmup_steps_percent", [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
        ),
        # loss function parameters
        "loss_func": loss_func,
        "tanimoto_similarity_loss_func": (
            trial.suggest_categorical(
                "tanimoto_similarity_loss_func",
                [
                    "mse",
                    "l1",
                    "smooth_l1",
                ],
            )
            if loss_func == "tanimotosimilarityloss"
            else None
        ),
        "use_2d_matryoshka": trial.suggest_categorical(
            "use_2d_matryoshka", [True, False]
        ),
        "last_layer_weight": 1.8708220063487997,
        "prior_layers_weight": 1.4598249321447245,
        "first_dim_weight": 1.0489590183361719,
        "second_dim_weight": 1.126163907196291,
        "third_dim_weight": 1.3807986616809407,
        "fourth_dim_weight": 1.397331091971628,
        "fifth_dim_weight": 1.6522851342433993,
        "sixth_dim_weight": 1.9858679040493405,
    }
    transformers.set_seed(param_config["seed"])

    train_dataloader, val_df = load_data(
        batch_size=param_config["train_batch_size"],
        sample_seed=param_config["seed"],
        num_of_rows_to_train_on=500000,
        num_of_rows_to_validate_on=100000,
        train_file=TRAIN_DS_DICT[param_config["dataset_key"]],
        val_file=VAL_DS_DICT[param_config["dataset_key"]],
    )

    val_evaluator = EmbeddingSimilarityEvaluator(
        val_df["smiles_a"],
        val_df["smiles_b"],
        val_df["fingerprint_similarity"],
        batch_size=param_config["train_batch_size"],
        main_similarity=SimilarityFunction.TANIMOTO,
        name="morgan-similarity",
        show_progress_bar=True,
        write_csv=True,
        precision="int8",
    )

    dimensions = [768, 512, 256, 128, 64, 32]
    matryoshka_weights: list[float] = [
        param_config["first_dim_weight"],
        param_config["second_dim_weight"],
        param_config["third_dim_weight"],
        param_config["fourth_dim_weight"],
        param_config["fifth_dim_weight"],
        param_config["sixth_dim_weight"],
    ]
    train_loss = get_train_loss(
        model,
        get_base_loss(model, loss_func, param_config["tanimoto_similarity_loss_func"]),
        param_config["use_2d_matryoshka"],
        dimensions,
        matryoshka_weights,
        param_config["last_layer_weight"],
        param_config["prior_layers_weight"],
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

    with wandb.init(project=PROJECT_NAME, config=param_config):
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
            model_save_path,
            "eval/similarity_evaluation_morgan-similarity_int8_results.csv",
        )
        eval_results_df = pd.read_csv(eval_file_path)
        metric = float(eval_results_df.iloc[-1]["spearman"])
        gc.collect()
        return metric
    return -1


def generate_hyperparameters():
    """Use this to generate hyperparameters to then be manually trained on using working training code."""
    study = optuna.create_study(
        storage=optuna.storages.RDBStorage(
            url="postgresql://postgres:password@192.168.0.8:5432/postgres",
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


if __name__ == "__main__":
    generate_hyperparameters()
