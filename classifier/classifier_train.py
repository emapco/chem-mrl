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
    CAT_TO_LABEL,
)
from loss import SoftmaxLoss, SelfAdjDiceLoss
from evaluator import LabelAccuracyEvaluator
from .load_data import load_data


def train() -> float:
    continue_training = True
    current_epoch = 4
    seed = 42 + (2 * current_epoch - 1) if continue_training else 42
    transformers.set_seed(seed)

    model_name = (
        "/home/manny/source/chem-mrl/output/"
        "ChemBERTa-zinc-base-v1-2d-matryoshka-embeddings"
        "-n_layers_per_step_2-TaniLoss-lr_1.1190785944700813e-05-batch_size_8"
        "-num_epochs_2-epoch_2-best-model-1900000_steps"
    )
    matryoshka_dim = 768

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model], truncate_dim=matryoshka_dim
    )

    num_epochs = 3
    warmup_steps_percent = 2
    lr_base = 3.4038386108141304e-06
    scheduler = "warmupcosinewithhardrestarts"

    dropout_p = 0.15
    loss_func = "SoftMax"
    if loss_func == "SelfAdjDice":
        dice_reduction = "mean"

    max_seq_length = word_embedding_model.max_seq_length
    train_batch_size = 160
    LR = lr_base * math.sqrt(train_batch_size)

    loss_parameter_str = (
        f"-dice_reduction_{dice_reduction}" if loss_func == "SelfAdjDice" else ""
    )
    model_save_path = (
        "output/"
        + model_name.rsplit("/", 1)[1][:20]
        + "-classification-model"
        + f"-epochs_{num_epochs}"
        + f"-mrl_dim_{matryoshka_dim}"
        + f"-lr_{lr_base}"
        + f"-batch_size_{train_batch_size}"
        + f"-warmup_steps_percent_{warmup_steps_percent}"
        + f"-scheduler_{scheduler}"
        + f"-dropout_p_{dropout_p}"
        + f"-loss_{loss_func}"
        + loss_parameter_str
    )
    print(f"\n{model_save_path}\n")

    wandb.init(
        project="chem-mrl-classification-train",
        config={
            "model_name": model_name,
            "train_batch_size": train_batch_size,
            "max_seq_length": max_seq_length,
            "matryoshka_dim": matryoshka_dim,
            "num_epochs": num_epochs,
            "lr_base": lr_base,
            "warmup_steps_percent": warmup_steps_percent,
            "scheduler": scheduler,
            "loss_func": loss_func,
            "dropout_p": dropout_p,
        },
    )

    train_dataloader, val_dataloader, _, num_labels = load_data(
        batch_size=train_batch_size,
        train_file=ISOMER_DESIGN_TRAIN_DS_PATH,
        val_file=ISOMER_DESIGN_VAL_DS_PATH,
    )

    if loss_func == "SoftMax":
        train_loss = SoftmaxLoss(
            model=model,
            # method returns truncated dim if using truncated mrl model
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=num_labels,
            dropout=dropout_p,
        )
    else:
        train_loss = SelfAdjDiceLoss(
            model=model,
            # method returns truncated dim if using truncated mrl model
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=num_labels,
            reduction=dice_reduction,
            dropout=dropout_p,
        )

    val_evaluator = LabelAccuracyEvaluator(
        dataloader=val_dataloader,
        softmax_model=train_loss,
        write_csv=True,
    )

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * warmup_steps_percent)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    total_number_training_points = len(train_dataloader) * train_batch_size
    weight_decay = 0.05 * math.sqrt(
        train_batch_size / (total_number_training_points * num_epochs)
    )

    def wandb_callback(score, epoch, _):
        eval_dict = {
            "accuracy": score,
            "epoch": epoch,
            "lr_base": lr_base,
            "model_name": model_name,
            "num_epochs": num_epochs,
            "matryoshka_dim": matryoshka_dim,
            "train_batch_size": train_batch_size,
            "warmup_steps_percent": warmup_steps_percent,
            "scheduler": scheduler,
            "loss_func": loss_func,
            "dropout_p": dropout_p if loss_func == "SoftMax" else None,
            "dice_reduction": dice_reduction if loss_func == "SelfAdjDice" else None,
        }
        wandb.log(eval_dict)
        apex.torch.clear_autocast_cache()
        apex.torch.cuda.empty_cache()
        gc.collect()

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        evaluation_steps=4,
        epochs=num_epochs,
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
        callback=wandb_callback,
    )

    eval_file_path = os.path.join(
        model_save_path, "eval/accuracy_evaluation_results.csv"
    )
    eval_results_df = pd.read_csv(eval_file_path)
    metric = float(eval_results_df.iloc[-1]["accuracy"])
    print(f"metric: {metric}")


if __name__ == "__main__":
    train()
