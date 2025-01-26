import argparse
import math
from contextlib import nullcontext

import transformers
import wandb
from apex.optimizers import FusedAdam
from load_data import load_data
from sentence_transformers import SentenceTransformer, models
from utils import get_model_save_path, get_signed_in_wandb_callback, get_train_loss

from chem_mrl.constants import CHEM_MRL_DIMENSIONS
from chem_mrl.evaluation import LabelAccuracyEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier model")
    # Model params
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--classifier_hidden_dimension",
        type=int,
        choices=CHEM_MRL_DIMENSIONS,
        default=768,
        help="Hidden dimension of classifier model. Must be one of the dimensions of the base MRL model.",
    )
    parser.add_argument("--train_batch_size", type=int, default=160)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--evaluation_steps", type=int, default=0)
    # Wandb params
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_api_key", default=None)
    parser.add_argument("--wandb_project_name", default="chem-mrl-classification-train")
    parser.add_argument("--wandb_run_name", default=None)
    # Optimizer and scheduler params
    parser.add_argument(
        "--lr_base",
        type=float,
        default=1.1190785944700813e-05,
        min_value=0.0,
        help="Base learning rate that will be scaled with batch size",
    )
    parser.add_argument(
        "--scheduler",
        choices=[
            "warmupconstant",
            "warmuplinear",
            "warmupcosine",
            "warmupcosinewithhardrestarts",
        ],
        default="warmupcosinewithhardrestarts",
    )
    parser.add_argument("--warmup_steps_percent", type=float, default=0.0)
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision (AMP) for training",
    )
    # Loss function params
    parser.add_argument(
        "--loss_func", choices=["SoftMax", "SelfAdjDice"], required=True
    )
    parser.add_argument(
        "--dropout_p", type=float, default=0.15, help="Dropout probability"
    )
    parser.add_argument(
        "--freeze_model",
        action="store_true",
        help="Freeze the base MRL model during training",
    )
    parser.add_argument(
        "--dice_reduction",
        default="mean",
        choices=["mean", "sum"],
    )
    parser.add_argument(
        "--dice_gamma",
        type=float,
        default=1.0,
        min=0.0,
    )
    # dataset paths
    parser.add_argument(
        "--train_dataset_path",
        help="Path to parquet train dataset.",
    )
    parser.add_argument(
        "--val_dataset_path",
        help="Path to parquet validation dataset.",
    )
    # misc.
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_output_path", default="output")
    parser.add_argument("--checkpoint_save_steps", type=int, default=1000000)
    parser.add_argument("--checkpoint_save_total_limit", type=int, default=20)

    return parser.parse_args()


def train(args):
    if args.seed is not None:
        transformers.set_seed(args.seed)

    # Load model components
    word_embedding_model = models.Transformer(args.model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        truncate_dim=args.classifier_hidden_dimension,
    )

    train_dataloader, val_dataloader, _, num_labels = load_data(
        batch_size=args.train_batch_size,
        train_file=args.train_dataset_path,
        val_file=args.val_dataset_path,
    )

    train_loss = get_train_loss(
        model=model,
        smiles_embedding_dimension=args.classifier_hidden_dimension,
        num_labels=num_labels,
        loss_func=args.loss_func,
        dropout=args.dropout_p,
        freeze_base_model=args.freeze_model,
        dice_reduction=args.dice_reduction,
        dice_gamma=args.dice_gamma,
    )

    val_evaluator = LabelAccuracyEvaluator(
        dataloader=val_dataloader,
        softmax_model=train_loss,
        write_csv=True,
    )

    # Calculate training parameters
    total_number_training_points = len(train_dataloader) * args.train_batch_size
    # normalized weight decay for adamw optimizer - https://arxiv.org/pdf/1711.05101.pdf
    # optimized hyperparameter lambda_norm = 0.05 for AdamW optimizer
    weight_decay = 0.05 * math.sqrt(
        args.train_batch_size / (total_number_training_points * args.num_epochs)
    )
    learning_rate = args.lr_base * math.sqrt(args.train_batch_size)
    warmup_steps = math.ceil(
        len(train_dataloader) * args.num_epochs * args.warmup_steps_percent
    )

    param_config = vars(args)
    model_save_path = get_model_save_path(param_config, args.model_output_path)
    wandb_callback = get_signed_in_wandb_callback(
        train_dataloader, args.use_wandb, args.wandb_api_key
    )

    with (
        nullcontext()
        if not args.use_wandb
        else wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
            config=param_config,
        )
    ):
        if args.use_wandb:
            wandb.watch(model, log="all", log_graph=True)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=val_evaluator,
            evaluation_steps=args.evaluation_steps,
            epochs=args.num_epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_class=FusedAdam,
            optimizer_params={
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "adam_w_mode": True,
            },
            save_best_model=True,
            use_amp=args.use_amp,
            show_progress_bar=True,
            scheduler=args.scheduler,
            checkpoint_path=args.model_output_path,
            checkpoint_save_steps=args.checkpoint_save_steps,
            checkpoint_save_total_limit=args.checkpoint_save_total_limit,
            callback=wandb_callback,
        )


if __name__ == "__main__":
    args = parse_args()
    train(args)
