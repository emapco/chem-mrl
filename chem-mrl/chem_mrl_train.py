import argparse
import math
from contextlib import nullcontext

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train chemical embeddings model")
    # Model params
    parser.add_argument(
        "--model_name",
        default="seyonec/ChemBERTa-zinc-base-v1",
        help="Base SMILES model name",
    )
    parser.add_argument("--train_batch_size", type=int, default=160)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--evaluation_steps", type=int, default=0)
    parser.add_argument(
        "--use_2d_matryoshka",
        action="store_true",
        help="Use 2D Matryoshka instead of 1D Matryoshka",
    )
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
        default="warmuplinear",
    )
    parser.add_argument("--warmup_steps_percent", type=float, default=0.0)
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision (AMP) for training",
    )
    # Loss function params
    parser.add_argument(
        "--loss_func",
        choices=["tanimotoloss", "tanimotosimilarityloss", "cosentloss"],
        default="tanimotoloss",
    )
    # Available loss functions are defined in utils/get_tanimoto_similarity_base_loss()
    parser.add_argument(
        "--tanimoto_similarity_loss_func",
        choices=[
            "mse",
            "l1",
            "smooth_l1",
            "huber",
            "bin_cross_entropy",
            "kldiv",
            "cosine_embedding_loss",
        ],
        help="Loss function to use for TanimotoSimilarityLoss",
    )
    # Layer weights
    parser.add_argument(
        "--last_layer_weight", type=float, default=1.8708220063487997, required=True
    )
    parser.add_argument(
        "--prior_layers_weight", type=float, default=1.4598249321447245, required=True
    )
    parser.add_argument(
        "--first_dim_weight", type=float, default=1.0489590183361719, required=True
    )
    parser.add_argument(
        "--second_dim_weight", type=float, default=1.126163907196291, required=True
    )
    parser.add_argument(
        "--third_dim_weight", type=float, default=1.3807986616809407, required=True
    )
    parser.add_argument(
        "--fourth_dim_weight", type=float, default=1.397331091971628, required=True
    )
    parser.add_argument(
        "--fifth_dim_weight", type=float, default=1.6522851342433993, required=True
    )
    parser.add_argument(
        "--sixth_dim_weight", type=float, default=1.9858679040493405, required=True
    )
    # misc.
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset_key",
        choices=list(TRAIN_DS_DICT.keys()),
        help="Key for the dataset to train and validate on. Options defined in constants.py",
        required=True,
    )
    parser.add_argument("--model_output_path", default="output")
    parser.add_argument("--checkpoint_save_steps", type=int, default=1000000)
    parser.add_argument("--checkpoint_save_total_limit", type=int, default=20)

    return parser.parse_args()


def train(args):
    if args.seed is not None:
        transformers.set_seed(args.seed)

    word_embedding_model = models.Transformer(args.model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_dataloader, val_df = load_data(
        batch_size=args.train_batch_size,
        sample_seed=args.seed,
        train_file=TRAIN_DS_DICT[args.dataset_key],
        val_file=VAL_DS_DICT[args.dataset_key],
    )

    val_evaluator = EmbeddingSimilarityEvaluator(
        val_df["smiles_a"],
        val_df["smiles_b"],
        val_df["fingerprint_similarity"],
        batch_size=args.train_batch_size,
        main_similarity=SimilarityFunction.TANIMOTO,
        name="morgan-similarity",
        show_progress_bar=True,
        write_csv=True,
        precision="int8",
    )

    dimensions = [768, 512, 256, 128, 64, 32]
    # more weight is given to smaller dimensions to improve downstream tasks
    # that benefit from dimensionality reduction (e.g. clustering)
    matryoshka_weights = [
        args.first_dim_weight,
        args.second_dim_weight,
        args.third_dim_weight,
        args.fourth_dim_weight,
        args.fifth_dim_weight,
        args.sixth_dim_weight,
    ]
    train_loss = get_train_loss(
        model,
        get_base_loss(model, args.loss_func, args.tanimoto_similarity_loss_func),
        args.use_2d_matryoshka,
        dimensions,
        matryoshka_weights,
        args.last_layer_weight,
        args.prior_layers_weight,
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
