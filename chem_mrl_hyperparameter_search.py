import math
import logging
import os

import apex
from apex.optimizers import FusedAdam
from sentence_transformers import (
    models,
    losses,
    datasets,
    SentenceTransformer,
)
import transformers
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
import pandas as pd
import optuna
import wandb

from tanimoto_loss import TanimotoLoss
from load_data import load_data


def objective(trial: optuna.Trial) -> float:
    apex.torch.clear_autocast_cache()
    apex.torch.cuda.empty_cache()
    transformers.set_seed(42)

    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 2-3 likely based on chemberta hyperparameter search on wandb
    # https://wandb.ai/seyonec/huggingface/reports/seyonec-s-ChemBERTa-update-08-31--VmlldzoyMjM1NDY
    NUM_EPOCHS = 2
    # loss parameters
    use_2d_matryoshka = True
    n_layers_per_step = trial.suggest_int("n_layers_per_step", 1, 6)
    sixth_dim_weight = 1.9858679040493405
    fifth_dim_weight = 1.6522851342433993
    fourth_dim_weight = 1.397331091971628
    third_dim_weight = 1.3807986616809407
    second_dim_weight = 1.126163907196291
    first_dim_weight = 1.0489590183361719
    last_layer_weight = 1.8708220063487997
    prior_layers_weight = 1.4598249321447245
    use_tanimoto_loss = True
    # optimizer parameters
    lr_base = 1.1190785944700813e-05
    # learning rate scheduler parameters
    warmup_steps_percent = 0
    scheduler = "warmuplinear"

    max_seq_length = word_embedding_model.max_seq_length
    train_batch_size = 24 if "ChemBERTa-77M-MLM" in model_name else 12
    # scale learning rate based on sqrt of the batch size
    LR = lr_base * math.sqrt(train_batch_size)
    model_save_path = (
        "output/"
        + model_name.split("/", 1)[1]
        + f"-{'2d-' if use_2d_matryoshka else ''}matryoshka-embeddings"
        + f"-n_layers_per_step_{n_layers_per_step}"
        + f"-{'TaniLoss' if use_tanimoto_loss else 'CoSENTLoss'}"
        + f"-lr_{lr_base}"
        + f"-last_layer_weight_{last_layer_weight}-prior_layers_weight_{prior_layers_weight}"
    )
    print(f"\n{model_save_path}\n")

    wandb.init(
        project="chem-mrl-hyperparameter-search-2d_matryoshka",
        config={
            "model_name": model_name,
            "train_batch_size": train_batch_size,
            "max_seq_length": max_seq_length,
            "num_epochs": NUM_EPOCHS,
            "use_tanimoto_loss": use_tanimoto_loss,
            "use_2d_matryoshka": use_2d_matryoshka,
            "n_layers_per_step": n_layers_per_step,
            "lr_base": lr_base,
            "warmup_steps_percent": warmup_steps_percent,
            "scheduler": scheduler,
            "first_dim_weight": first_dim_weight,
            "second_dim_weight": second_dim_weight,
            "third_dim_weight": third_dim_weight,
            "fourth_dim_weight": fourth_dim_weight,
            "fifth_dim_weight": fifth_dim_weight,
            "sixth_dim_weight": sixth_dim_weight,
            "last_layer_weight": last_layer_weight,
            "prior_layers_weight": prior_layers_weight,
        },
    )

    train_dataloader = datasets.NoDuplicatesDataLoader(
        TRAIN_EXAMPLES, batch_size=train_batch_size
    )

    # define loss function
    base_loss = TanimotoLoss(model) if use_tanimoto_loss else losses.CoSENTLoss(model)
    dimensions = (
        [384, 256, 128, 64, 32]
        if "ChemBERTa-77M-MLM" in model_name
        else [768, 512, 256, 128, 64, 32]
    )
    matryoshka_weights: list[float] = (
        # more weight is given to smaller dimensions to improve downstream tasks
        # e.g. clustering
        [1, 1, 1, 1, 1]
        if "ChemBERTa-77M-MLM" in model_name
        else [
            first_dim_weight,
            second_dim_weight,
            third_dim_weight,
            fourth_dim_weight,
            fifth_dim_weight,
            sixth_dim_weight,
        ]
    )
    train_loss = (
        losses.Matryoshka2dLoss(
            model,
            base_loss,
            dimensions,
            matryoshka_weights=matryoshka_weights,
            n_layers_per_step=n_layers_per_step,
            n_dims_per_step=-1,
            last_layer_weight=last_layer_weight,
            prior_layers_weight=prior_layers_weight,
        )
        if use_2d_matryoshka
        else losses.MatryoshkaLoss(model, base_loss, dimensions)
    )

    val_evaluator = EmbeddingSimilarityEvaluator(
        VAL_DF["smiles_a"],  # type: ignore
        VAL_DF["smiles_b"],  # type: ignore
        VAL_DF["fingerprint_similarity"],  # type: ignore
        batch_size=train_batch_size,
        main_similarity=SimilarityFunction.COSINE,
        name="morgan-similarity",
        show_progress_bar=True,
        write_csv=True,
        precision="int8",
    )

    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * warmup_steps_percent)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # normalized weight decay for adamw optimizer - https://arxiv.org/pdf/1711.05101.pdf
    # optimized hyperparameter lambda_norm = 0.05 for AdamW optimizer
    total_number_training_points = len(train_dataloader) * train_batch_size
    weight_decay = 0.05 * math.sqrt(
        train_batch_size / (total_number_training_points * NUM_EPOCHS)
    )

    def wandb_callback(score, epoch, _):
        eval_dict = {
            "cosine_pearson": score,
            "epoch": epoch,
            "lr_base": lr_base,
            "last_layer_weight": last_layer_weight,
            "prior_layers_weight": prior_layers_weight,
        }
        wandb.log(eval_dict)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],  # type: ignore
        evaluator=val_evaluator,
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        optimizer_class=FusedAdam,
        optimizer_params={
            "lr": LR,
            "weight_decay": weight_decay,
            "adam_w_mode": True,
        },
        use_amp=False,
        show_progress_bar=True,
        scheduler=scheduler,
        checkpoint_path="output",
        checkpoint_save_steps=1000000,
        callback=wandb_callback,
    )

    # load metric from evaluation output file
    eval_file_path = os.path.join(
        model_save_path, "eval/similarity_evaluation_morgan-similarity_int8_results.csv"
    )
    eval_results_df = pd.read_csv(eval_file_path)
    eval_results_df.drop(
        columns=[
            "steps",
            "dot_pearson",
            "dot_spearman",
            "manhattan_pearson",
            "manhattan_spearman",
            "euclidean_pearson",
            "euclidean_spearman",
        ],
        inplace=True,
    )
    metric = float(eval_results_df.iloc[-1]["cosine_spearman"])
    return metric


def generate_hyperparameters():
    """Bug encountered when using optuna and sentence transformers.
    The evaluation score is always NAN.
    Use this to generate hyperparameters to then be manually trained on using working training code.
    """
    global TRAIN_EXAMPLES
    global VAL_DF
    TRAIN_EXAMPLES, VAL_DF = load_data()

    study = optuna.create_study(
        study_name="chem-mrl-hyperparameter-tuning",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=20,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    print("Best hyperparameters found:")
    print(study.best_params)
    print("Best best trials:")
    print(study.best_trials)
    print("Best trial:")
    print(study.best_trial)
    study.trials_dataframe().to_csv("chem-mrl-hyperparameter-tuning.csv", index=False)


if __name__ == "__main__":
    generate_hyperparameters()
