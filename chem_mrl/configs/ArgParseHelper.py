import argparse

from chem_mrl.constants import BASE_MODEL_NAME, CHEM_MRL_DIMENSIONS

from .BaseConfig import SCHEDULER_OPTIONS, WATCH_LOG_OPTIONS
from .ClassifierConfig import (
    CLASSIFIER_EVAL_METRIC_OPTIONS,
    CLASSIFIER_LOSS_FCT_OPTIONS,
    DICE_REDUCTION_OPTIONS,
)
from .MrlConfig import (
    CHEM_MRL_EVAL_METRIC_OPTIONS,
    CHEM_MRL_LOSS_FCT_OPTIONS,
    EVAL_SIMILARITY_FCT_OPTIONS,
    TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS,
)


def add_base_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--test_dataset_path")
    parser.add_argument(
        "--num_train_samples",
        type=int,
        help="Number of training samples to load. Uses seeded sampling if a seed is set.",
    )
    parser.add_argument(
        "--num_val_samples",
        type=int,
        help="Number of evaluation samples to load. Uses seeded sampling if a seed is set.",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        help="Number of testing samples to load. Uses seeded sampling if a seed is set.",
    )
    parser.add_argument(
        "--model_name",
        default=BASE_MODEL_NAME,
        help="Name of the model to use. Either file path or a hugging-face model name.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of epochs to train"
    )
    parser.add_argument(
        "--lr_base",
        type=float,
        default=1.1190785944700813e-05,
        help="Base learning rate. Will be scaled by the batch size",
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_OPTIONS,
        default="warmuplinear",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--warmup_steps_percent",
        type=float,
        default=0.0,
        help="Number of warmup steps that the scheduler will use",
    )
    parser.add_argument(
        "--use_fused_adamw",
        action="store_true",
        help="Use cuda-optimized FusedAdamW optimizer. ~10%% faster than torch.optim.AdamW",
    )
    parser.add_argument(
        "--use_tf32",
        action="store_true",
        help="Use TensorFloat-32 for matrix multiplication and convolutions",
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Omit to not set a seed during training. Seed dataloader sampling and transformers.",
    )
    parser.add_argument(
        "--model_output_path", default="output", help="Path to save model"
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=0,
        help="Run evaluator every evaluation_steps",
    )
    parser.add_argument(
        "--checkpoint_save_steps",
        type=int,
        default=0,
        help="Save checkpoint every checkpoint_save_steps",
    )
    parser.add_argument(
        "--checkpoint_save_total_limit", type=int, default=20, help="Save total limit"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use W&B for logging. Must be enabled for other W&B features to work.",
    )

    # WandbConfig params - utilized by base config
    parser.add_argument(
        "--wandb_api_key",
        help="W&B API key. Can be omitted if W&B cli is installed and logged in",
    )
    parser.add_argument("--wandb_project_name")
    parser.add_argument("--wandb_run_name")
    parser.add_argument(
        "--wandb_use_watch", action="store_true", help="Enable W&B watch"
    )
    parser.add_argument(
        "--wandb_watch_log",
        choices=WATCH_LOG_OPTIONS,
        default="all",
        help="Specify which logs to W&B should watch",
    )
    parser.add_argument(
        "--wandb_watch_log_freq", type=int, default=1000, help="How often to log"
    )
    parser.add_argument(
        "--wandb_watch_log_graph",
        action="store_true",
        help="Specify if graphs should be logged by W&B",
    )

    return parser


def add_chem_mrl_config_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--smiles_a_column_name", default="smiles_a", help="SMILES A column name"
    )
    parser.add_argument(
        "--smiles_b_column_name", default="smiles_b", help="SMILES B column name"
    )
    parser.add_argument(
        "--label_column_name",
        default="fingerprint_similarity",
        help="Label column name",
    )
    parser.add_argument(
        "--loss_func",
        choices=CHEM_MRL_LOSS_FCT_OPTIONS,
        default="tanimotosentloss",
        help="Loss function",
    )
    parser.add_argument(
        "--tanimoto_similarity_loss_func",
        choices=TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS,
        default=None,
        help="Base loss function for tanimoto similarity loss function",
    )
    parser.add_argument(
        "--eval_similarity_fct",
        choices=EVAL_SIMILARITY_FCT_OPTIONS,
        default="tanimoto",
        help="Similarity metric to use for evaluation",
    )
    parser.add_argument(
        "--eval_metric",
        choices=CHEM_MRL_EVAL_METRIC_OPTIONS,
        default="spearman",
        help="Metric to use for evaluation",
    )
    # MRL dimension weights
    parser.add_argument("--first_dim_weight", type=float, default=1.0489590183361719)
    parser.add_argument("--second_dim_weight", type=float, default=1.126163907196291)
    parser.add_argument("--third_dim_weight", type=float, default=1.3807986616809407)
    parser.add_argument("--fourth_dim_weight", type=float, default=1.397331091971628)
    parser.add_argument("--fifth_dim_weight", type=float, default=1.6522851342433993)
    parser.add_argument("--sixth_dim_weight", type=float, default=1.9858679040493405)

    # Chem2dMRLConfig specific params
    parser.add_argument(
        "--use_2d_matryoshka", action="store_true", help="Use 2D Matryoshka"
    )
    parser.add_argument(
        "--last_layer_weight",
        type=float,
        default=1.8708220063487997,
        help="Last layer weight used by 2D Matryoshka for computing the loss",
    )
    parser.add_argument(
        "--prior_layers_weight",
        type=float,
        default=1.4598249321447245,
        help="Prior layers weight used by 2D Matryoshka for computing compute the loss",
    )

    return parser


def add_classifier_config_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--smiles_column_name", type=str, default="smiles", help="SMILES column name"
    )
    parser.add_argument(
        "--label_column_name", type=str, default="label", help="Label column name"
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        choices=CLASSIFIER_EVAL_METRIC_OPTIONS,
        default="accuracy",
        help="Metric to use for evaluation",
    )
    parser.add_argument(
        "--loss_func",
        type=str,
        choices=CLASSIFIER_LOSS_FCT_OPTIONS,
        default="softmax",
        help="Loss function",
    )
    parser.add_argument(
        "--classifier_hidden_dimension",
        type=int,
        choices=CHEM_MRL_DIMENSIONS,
        default=CHEM_MRL_DIMENSIONS[0],
        help="Classifier hidden dimension. The base SMILES model will be truncated to this dimension",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.15,
        help="Dropout probability for linear layer regularization",
    )
    parser.add_argument(
        "--freeze_model", action="store_true", help="Freeze internal base SMILES model"
    )

    # DiceLoss specific params
    parser.add_argument(
        "--dice_reduction",
        type=str,
        choices=DICE_REDUCTION_OPTIONS,
        default="mean",
        help="Dice loss reduction. Used if loss_func=selfadjdice",
    )
    parser.add_argument(
        "--dice_gamma",
        type=float,
        default=1.0,
        help="Dice loss gamma. Used if loss_func=selfadjdice",
    )

    return parser
