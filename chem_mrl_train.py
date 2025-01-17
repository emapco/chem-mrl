import math
import logging
import gc

import torch
from apex.optimizers import FusedAdam
from torch.utils.data import DataLoader
from sentence_transformers import (
    models,
    losses,
    SentenceTransformer,
)
import transformers
import wandb

from tanimoto_loss import TanimotoLoss
from load_data import load_data
from evaluator import EmbeddingSimilarityEvaluator, SimilarityFunction


def train():
    continue_training = True
    current_epoch = 4
    seed = 42 + (2 * current_epoch - 1) if continue_training else 42
    transformers.set_seed(seed)

    model_name = (
        "output/ChemBERTa-zinc-base-v1-QED_functional_morgan_fingerprint-2d-matryoshka-embeddings-num_epochs_2-epoch_3"
        if continue_training
        else "seyonec/ChemBERTa-zinc-base-v1"
    )
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    NUM_EPOCHS = 1 if continue_training else 2
    # loss parameters
    use_2d_matryoshka = True
    n_layers_per_step = 2
    n_dims_per_step = -1  # all dimensions
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

    dataset = "QED_functional_morgan_fingerprint"
    max_seq_length = word_embedding_model.max_seq_length
    train_batch_size = 8
    # scale learning rate based on sqrt of the batch size
    LR = lr_base * math.sqrt(train_batch_size)
    LR = LR / current_epoch if continue_training else LR
    model_save_path = (
        (
            "output/"
            + model_name.split("/", 1)[1]
            + f"-{dataset}"
            + f"-{'2d-' if use_2d_matryoshka else ''}matryoshka-embeddings"
            # + f"-n_layers_per_step_{n_layers_per_step}"
            # + f"-{'TaniLoss' if use_tanimoto_loss else 'CoSENTLoss'}"
            # + f"-lr_{lr_base}"
            # + f"-batch_size_{train_batch_size}"
            + f"-num_epochs_{NUM_EPOCHS}"
        )
        if not continue_training
        else f"output/{model_name.split("/", 1)[1]}-epoch_{current_epoch}"
    )

    logging.info(f"\n{model_save_path}\n")

    wandb.init(
        project="chem-mrl-QED-train",
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
            "dataset": dataset,
        },
    )

    train_dataloader = DataLoader(
        TRAIN_EXAMPLES,  # type: ignore
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=1,
        pin_memory=True,
    )

    # define loss function
    base_loss = TanimotoLoss(model) if use_tanimoto_loss else losses.CoSENTLoss(model)
    dimensions = [768, 512, 256, 128, 64, 32]
    # more weight is given to smaller dimensions to improve downstream tasks
    # e.g. clustering
    matryoshka_weights: list[float] = [
        first_dim_weight,
        second_dim_weight,
        third_dim_weight,
        fourth_dim_weight,
        fifth_dim_weight,
        sixth_dim_weight,
    ]
    train_loss = (
        losses.Matryoshka2dLoss(
            model,
            base_loss,
            dimensions,
            matryoshka_weights=matryoshka_weights,
            n_layers_per_step=n_layers_per_step,
            n_dims_per_step=n_dims_per_step,
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
        batch_size=1024,
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

    def wandb_callback(score, epoch, steps):
        eval_dict = {
            "cosine_pearson": score,
            "epoch": epoch,
            "steps": steps,
            "n_layers_per_step": n_layers_per_step,
            "total_number_training_points": total_number_training_points,
        }
        wandb.log(eval_dict)
        torch.cuda.empty_cache()
        torch.clear_autocast_cache()
        gc.collect()

    eval_steps = 100000 if continue_training else 200000
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],  # type: ignore
        evaluator=val_evaluator,
        evaluation_steps=eval_steps,
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
        checkpoint_save_steps=eval_steps,
        checkpoint_save_total_limit=20,
        callback=wandb_callback,
    )


if __name__ == "__main__":
    global TRAIN_EXAMPLES
    global VAL_DF
    TRAIN_EXAMPLES, VAL_DF = load_data(
        num_of_rows_to_train_on=None,
        num_of_rows_to_validate_on=None,
        val_ds_path=None,
    )
    train()
