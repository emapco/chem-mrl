import math
import logging
import os
from typing import Iterable

import apex
from apex.optimizers import FusedAdam
from torch import nn
from sentence_transformers import (
    models,
    losses,
    datasets,
    SentenceTransformer,
    InputExample,
)
import transformers

import pandas as pd
import optuna
import wandb

from evaluator import EmbeddingSimilarityEvaluator, SimilarityFunction
from tanimoto_loss import TanimotoLoss, TanimotoSimilarityLoss
from load_data import load_data
from constants import (
    OUTPUT_MODEL_DIR,
    TRAIN_DS_DICT,
    TEST_DS_DICT,
)

logger = logging.getLogger()


def objective(
    trial: optuna.Trial,
    dataset_key: str,
    train_examples: Iterable[InputExample],
    val_df: pd.DataFrame,
) -> float:
    apex.torch.clear_autocast_cache()
    apex.torch.cuda.empty_cache()
    transformers.set_seed(42)

    # Load model components
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # generate hyperparameters
    loss_func = trial.suggest_categorical(
        "loss_func", ["TanimotoLoss", "TanimotoSimilarityLoss", "CoSENTLoss"]
    )
    param_config = {
        "model_name": model_name,
        "dataset_key": dataset_key,
        "train_batch_size": 8,
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
            if loss_func == "TanimotoSimilarityLoss"
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

    # create dataloaders and evaluators
    train_dataloader = datasets.NoDuplicatesDataLoader(
        train_examples, batch_size=param_config["train_batch_size"]
    )
    val_evaluator = EmbeddingSimilarityEvaluator(
        val_df["smiles_a"],  # type: ignore
        val_df["smiles_b"],  # type: ignore
        val_df["fingerprint_similarity"],  # type: ignore
        batch_size=param_config["train_batch_size"],
        main_similarity=SimilarityFunction.TANIMOTO,
        name="morgan-similarity",
        show_progress_bar=True,
        write_csv=True,
        precision="int8",
    )

    # define loss function
    base_loss = get_base_loss(
        model, loss_func, param_config["tanimoto_similarity_loss_func"]
    )
    dimensions = [768, 512, 256, 128, 64, 32]
    # more weight is given to smaller dimensions to improve downstream tasks
    # that benefit from dimensionality reduction (e.g. clustering)
    matryoshka_weights: list[float] = [
        param_config["first_dim_weight"],
        param_config["second_dim_weight"],
        param_config["third_dim_weight"],
        param_config["fourth_dim_weight"],
        param_config["fifth_dim_weight"],
        param_config["sixth_dim_weight"],
    ]

    train_loss = (
        losses.Matryoshka2dLoss(
            model,
            base_loss,
            dimensions,
            matryoshka_weights=matryoshka_weights,
            n_layers_per_step=-1,
            n_dims_per_step=-1,
            last_layer_weight=param_config["last_layer_weight"],
            prior_layers_weight=param_config["prior_layers_weight"],
        )
        if param_config["use_2d_matryoshka"]
        else losses.MatryoshkaLoss(
            model,
            base_loss,
            dimensions,
            matryoshka_weights=matryoshka_weights,
            n_dims_per_step=-1,
        )
    )

    # normalized weight decay for adamw optimizer - https://arxiv.org/pdf/1711.05101.pdf
    # optimized hyperparameter lambda_norm = 0.05 for AdamW optimizer
    total_number_training_points = (
        len(train_dataloader) * param_config["train_batch_size"]
    )
    weight_decay = 0.05 * math.sqrt(
        param_config["train_batch_size"]
        / (total_number_training_points * param_config["num_epochs"])
    )
    # scale learning rate based on sqrt of the batch size
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
        project="chem-mrl-hyperparameter-search-2025",
        config=param_config,
    )

    def wandb_callback(score, epoch, steps):
        if steps == -1:
            steps = (epoch + 1) * len(train_dataloader)
        eval_dict = {
            "score": score,
            "epoch": epoch,
            "steps": steps,
            **param_config,
        }

        wandb.log(eval_dict)
        trial.report(score, step=steps)

        if trial.should_prune():
            raise optuna.TrialPruned()

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
        callback=wandb_callback,
    )

    # Get final metric
    eval_file_path = os.path.join(
        model_save_path, "eval/similarity_evaluation_morgan-similarity_int8_results.csv"
    )
    eval_results_df = pd.read_csv(eval_file_path)
    metric = float(eval_results_df.iloc[-1]["spearman"])
    return metric


def get_base_loss(
    model: SentenceTransformer,
    loss_func: str,
    tanimoto_similarity_loss_func: str | None,
) -> nn.Module:
    LOSS_FUNCTIONS = {
        "TanimotoLoss": lambda model: TanimotoLoss(model),
        "CoSENTLoss": lambda model: losses.CoSENTLoss(model),
        "TanimotoSimilarityLoss": {
            "mse": lambda model: TanimotoSimilarityLoss(model, loss=nn.MSELoss()),
            "l1": lambda model: TanimotoSimilarityLoss(model, loss=nn.L1Loss()),
            "smooth_l1": lambda model: TanimotoSimilarityLoss(
                model, loss=nn.SmoothL1Loss()
            ),
            "huber": lambda model: TanimotoSimilarityLoss(model, loss=nn.HuberLoss()),
            "bin_cross_entropy": lambda model: TanimotoSimilarityLoss(
                model, loss=nn.BCEWithLogitsLoss()
            ),
            "kldiv": lambda model: TanimotoSimilarityLoss(
                model, loss=nn.KLDivLoss(reduction="batchmean")
            ),
            "cosine_embedding_loss": lambda model: TanimotoSimilarityLoss(
                model, loss=nn.CosineEmbeddingLoss()
            ),
        },
    }
    if loss_func in ["TanimotoLoss", "CoSENTLoss"]:
        return LOSS_FUNCTIONS[loss_func](model)

    return LOSS_FUNCTIONS["TanimotoSimilarityLoss"][tanimoto_similarity_loss_func](
        model
    )


def get_model_save_path(
    param_config: dict,
) -> str:
    model_save_path = os.path.join(
        OUTPUT_MODEL_DIR,
        f"{param_config['dataset_key']}-chem-{'2D' if param_config['use_2d_matryoshka'] else '1D'}mrl"
        f"-{param_config['train_batch_size']}-{param_config['num_epochs']}"
        f"-{param_config['lr_base']:6f}-{param_config['scheduler']}-{param_config['warmup_steps_percent']}"
        f"-{param_config['loss_func']}-{param_config['tanimoto_similarity_loss_func']}"
        f"-{param_config['last_layer_weight']:4f}-{param_config['prior_layers_weight']:4f}"
        f"-{param_config['first_dim_weight']:4f}-{param_config['second_dim_weight']:4f}-{param_config['third_dim_weight']:4f}"  # noqa: E501
        f"-{param_config['fourth_dim_weight']:4f}-{param_config['fifth_dim_weight']:4f}-{param_config['sixth_dim_weight']:4f}",  # noqa: E501
    )
    logger.info(f"\n{model_save_path}\n")
    return model_save_path


def generate_hyperparameters():
    """Use this to generate hyperparameters to then be manually trained on using working training code."""
    # expensive to load data, so load it once
    dataset_key = "morgan-similarity"
    train_examples, val_df = load_data(
        dataset_path=TRAIN_DS_DICT[dataset_key],
        val_ds_path=TEST_DS_DICT[dataset_key],
    )

    def objective_closure(trial: optuna.Trial):
        return objective(
            trial,
            dataset_key=dataset_key,
            train_examples=train_examples,
            val_df=val_df,
        )

    study = optuna.create_study(
        storage="postgresql://postgres:password@192.168.0.8:5432/postgres",
        study_name="chem-mrl-hyperparameter-tuning",
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1),
    )
    study.optimize(
        objective_closure,
        n_trials=256,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    logger.info("Best hyperparameters found:")
    logger.info(study.best_params)
    logger.info("Best best trials:")
    logger.info(study.best_trials)
    logger.info("Best trial:")
    logger.info(study.best_trial)
    study.trials_dataframe().to_csv("chem-mrl-hyperparameter-tuning.csv", index=False)


if __name__ == "__main__":
    generate_hyperparameters()
