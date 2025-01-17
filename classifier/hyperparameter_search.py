import os
import traceback
from typing import Callable, Tuple, Any
import math
from pathlib import Path

import torch
import torch.utils.data
from torch import nn
from torch.optim import AdamW
from ray import tune, init, train
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb

from sentence_transformers import SentenceTransformer
from loss import SelfAdjDiceLoss
from util import set_seed, get_data_parallel_model, get_data_loaders, allow_cuda_tf32


def evaluate_loss(
    model: (
        nn.DataParallel[SentenceTransformerClassifier] | SentenceTransformerClassifier
    ),
    criterion: nn.CrossEntropyLoss | SelfAdjDiceLoss,
    dataloader: torch.utils.data.DataLoader[Any],
):
    model.eval()
    running_val_loss = 0.0
    count = 0
    num_labels = (
        model.module.num_labels
        if isinstance(model, nn.DataParallel)
        else model.num_labels
    )

    with torch.no_grad():
        for seq, attn_masks, labels in tqdm(dataloader):
            seq, attn_masks, labels = (
                seq.cuda(non_blocking=True),
                attn_masks.cuda(non_blocking=True),
                labels.cuda(non_blocking=True),
            )
            logits = model(seq, attn_masks)
            running_val_loss += criterion(
                logits.view(-1, num_labels), labels.view(-1)
            ).item()
            count += 1

    return running_val_loss / count


def train_model(
    model: (
        nn.DataParallel[SentenceTransformerClassifier] | SentenceTransformerClassifier
    ),
    criterion: nn.CrossEntropyLoss | SelfAdjDiceLoss,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_loader: torch.utils.data.DataLoader[Any],
    val_loader: torch.utils.data.DataLoader[Any],
    epochs: int,
    output_current_epoch_model_data: Callable[
        [
            nn.DataParallel[SentenceTransformerClassifier]
            | SentenceTransformerClassifier,
            int,
            float,
            float,
        ],
        Tuple[bool, float],
    ],
):
    try:
        val_loss = np.Inf
        num_labels = (
            model.module.num_labels
            if isinstance(model, nn.DataParallel)
            else model.num_labels
        )

        for epoch in range(epochs):
            model.train()
            running_training_loss = 0.0
            total_epoch_iter = 0

            for sent, labels in tqdm(train_loader):
                sent, labels = (
                    sent.cuda(non_blocking=True),
                    labels.cuda(non_blocking=True),
                )

                logits = model(sent)
                loss = criterion(logits.view(-1, num_labels), labels.view(-1))
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                running_training_loss += loss.item()
                total_epoch_iter += 1
                train.report({"loss": loss.item()})

            val_loss = evaluate_loss(model, criterion, val_loader)
            print(f"\nEpoch {epoch + 1} complete! Validation Loss : {val_loss}")

            avg_training_loss = running_training_loss / total_epoch_iter
            did_val_loss_increase, val_loss = output_current_epoch_model_data(
                model, epoch + 1, avg_training_loss, val_loss
            )

            if did_val_loss_increase:
                print("Validation loss increased. Stopping training.")
                return val_loss

        torch.cuda.empty_cache()
        return val_loss

    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())
        return -1


def objective(config):
    seed_val = 12345
    set_seed(seed_val)

    lr = config["lr"]
    batch_size = 80
    epochs = 2

    train_loader, val_loader, _ = get_data_loaders(batch_size)
    num_training_steps = epochs * len(train_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_data_parallel_model()

    weight_decay = 0.1
    total_number_training_points = len(train_loader) * batch_size
    weight_decay = 0.05 * math.sqrt(
        batch_size / (total_number_training_points * epochs)
    )

    criterion = SelfAdjDiceLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-6,
    )

    num_warmup_steps = int(0.06 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    output_current_epoch_model_data_fn = fn_builder_output_current_epoch_model_data(
        optimizer, lr, num_warmup_steps, weight_decay
    )
    model.to(device)

    # Initialize wandb
    wandb.init(
        project="your_project_name",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "num_warmup_steps": num_warmup_steps,
        },
    )

    val_loss = train_model(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        epochs,
        output_current_epoch_model_data_fn,
    )

    wandb.log({"val_loss": val_loss})
    wandb.finish()
    train.report({"loss": val_loss})


def fn_builder_output_current_epoch_model_data(
    optimizer, lr, num_warmup_steps, weight_decay
):
    def output_current_epoch_model_data(
        model: (
            nn.DataParallel[SentenceTransformerClassifier]
            | SentenceTransformerClassifier
        ),
        current_epoch: int,
        training_loss: float,
        val_loss: float,
    ) -> Tuple[bool, float]:
        checkpoint_dir = Path(CHECKPOINTS_DIR)
        checkpoint_dir.mkdir(exist_ok=True)
        pt_file_name = (
            f"bert-epoch-{current_epoch}-lr_{'%.6E' % lr}"
            + f"-warmup_steps_{num_warmup_steps}-weight_decay_{weight_decay}.pt"
        )
        csv_file_name = (
            f"bert-lr_{'%.6E' % lr}"
            + f"-warmup_steps_{num_warmup_steps}-weight_decay_{weight_decay}.csv"
        )
        checkpoint = train.Checkpoint.from_directory(checkpoint_dir)
        with checkpoint.as_directory() as checkpoint_ctx_dir:
            pt_path = os.path.join(checkpoint_ctx_dir, pt_file_name)
            csv_path = os.path.join(checkpoint_ctx_dir, csv_file_name)

            if isinstance(model, nn.DataParallel):
                torch.save((model.module.state_dict(), optimizer.state_dict()), pt_path)
            else:
                torch.save((model.state_dict(), optimizer.state_dict()), pt_path)
            train.report({"loss": val_loss}, checkpoint=checkpoint)

            current_epoch_metrics_df = pd.DataFrame(
                [
                    {
                        "epoch": current_epoch,
                        "training_loss": training_loss,
                        "validation_loss": val_loss,
                    }
                ]
            )
            did_val_loss_increase = False
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = pd.concat([df, current_epoch_metrics_df])
                df.to_csv(csv_path, index=False)
                did_val_loss_increase = bool(
                    df.iloc[-2]["validation_loss"] <= df.iloc[-1]["validation_loss"]
                )
                if did_val_loss_increase:
                    return did_val_loss_increase, float(df.iloc[-2]["validation_loss"])
            else:
                current_epoch_metrics_df.to_csv(csv_path, index=False)
            return did_val_loss_increase, val_loss

    return output_current_epoch_model_data


if __name__ == "__main__":
    allow_cuda_tf32()

    trial_configs = [
        {
            "lr": 2.572460e-05,
        },
    ]
    search_space = {
        "lr": hp.uniform("lr", 2e-5, 3e-5),
    }
    hyperopt_search = HyperOptSearch(
        search_space, metric="loss", mode="min", points_to_evaluate=trial_configs
    )
    init()  # Initialize Ray
    analysis_name = "web-classifier-fine-tune"
    analysis = tune.run(
        objective,
        name=analysis_name,
        resume=False,
        metric="loss",
        mode="min",
        resources_per_trial={"gpu": 2},
        num_samples=1,
        search_alg=hyperopt_search,
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        ),
    )

    best_config = analysis.get_best_config(metric="loss", mode="min")
    print("Best config: ", best_config)

    best_trial = analysis.get_best_trial(metric="loss", mode="min")
    if best_trial:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="loss", mode="min"
        )
        print("Best checkpoint: ", best_checkpoint)
