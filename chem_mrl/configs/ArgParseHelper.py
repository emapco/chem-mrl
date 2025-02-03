import argparse

from chem_mrl.constants import BASE_MODEL_NAME, CHEM_MRL_DIMENSIONS

from .BaseConfig import SCHEDULER_OPTIONS, WATCH_LOG_OPTIONS, WandbConfig, _BaseConfig
from .ClassifierConfig import (
    CLASSIFIER_EVAL_METRIC_OPTIONS,
    CLASSIFIER_LOSS_FCT_OPTIONS,
    DICE_REDUCTION_OPTIONS,
    ClassifierConfig,
    DiceLossClassifierConfig,
)
from .MrlConfig import (
    CHEM_MRL_EMBEDDING_POOLING_OPTIONS,
    CHEM_MRL_EVAL_METRIC_OPTIONS,
    CHEM_MRL_LOSS_FCT_OPTIONS,
    EVAL_SIMILARITY_FCT_OPTIONS,
    TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS,
    Chem2dMRLConfig,
    ChemMRLConfig,
)


def add_base_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--test_dataset_path")
    parser.add_argument(
        "--n_train_samples",
        type=int,
        help="Number of training samples to load. Uses seeded sampling if a seed is set.",
    )
    parser.add_argument(
        "--n_val_samples",
        type=int,
        help="Number of evaluation samples to load. Uses seeded sampling if a seed is set.",
    )
    parser.add_argument(
        "--n_test_samples",
        type=int,
        help="Number of testing samples to load. Uses seeded sampling if a seed is set.",
    )
    parser.add_argument(
        "--n_dataloader_workers",
        type=int,
        default=0,
        help="How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--generate_dataset_examples_at_init",
        action="store_true",
        help="If set, then all `sentence_transformers.InputExample` examples will be generated at at initialization. "
        "If not set, the `sentence_transformers.InputExample` examples are generated on the fly by the dataloader.",
    )
    parser.add_argument(
        "--model_name",
        default=BASE_MODEL_NAME,
        help="Name of the model to use. Must be either a file path or a hugging-face model name.",
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
        help="Base learning rate. Will be scaled by the square root of the batch size",
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
        help="Omit to not set a seed during training. Used to seed the dataloader sampling and the transformer.",
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
        "--return_eval_metric",
        action="store_true",
        help="Return the final evaluation metric after training",
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
        "--embedding_pooling",
        choices=CHEM_MRL_EMBEDDING_POOLING_OPTIONS,
        default="mean",
        help="Pooling layer method applied to the embeddings."
        "Pooling layer is required to generate a fixed sized SMILES embedding from a variable sized SMILES."
        "For details visit: https://sbert.net/docs/package_reference/sentence_transformer/models.html#sentence_transformers.models.Pooling",
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
        help="Base loss function for tanimoto similarity loss function (only for tanimotosimilarityloss)",
    )
    parser.add_argument(
        "--eval_similarity_fct",
        choices=EVAL_SIMILARITY_FCT_OPTIONS,
        default="tanimoto",
        help="Similarity function to use for evaluation",
    )
    parser.add_argument(
        "--eval_metric",
        choices=CHEM_MRL_EVAL_METRIC_OPTIONS,
        default="spearman",
        help="Metric to use for evaluation",
    )
    parser.add_argument(
        "--mrl_dimensions",
        nargs="+",
        default=CHEM_MRL_DIMENSIONS,
        type=int,
        help="A list of embedding dimensions to be used for the loss function. "
        "Each value must be less than equal to the base transformer's hidden dimension.",
    )
    parser.add_argument(
        "--dim_weights",
        nargs="+",
        default=[1, 1, 1, 1, 1, 1, 1, 1],
        type=float,
        help="A list of weights to be used for the loss function. "
        "The number of dimension weights must match that of the MRL dimensions.",
    )
    parser.add_argument(
        "--n_dims_per_step",
        type=int,
        default=1,
        help="The number of dimensions to use per step. If -1, then all dimensions are used. "
        "If > 0, then a random sample of n_dims_per_step dimensions are used per step.",
    )

    # Chem2dMRLConfig specific params
    parser.add_argument(
        "--use_2d_matryoshka",
        action="store_true",
        help="Use 2D Matryoshka to train over layers in addition to embedding dimensions.",
    )
    parser.add_argument(
        "--n_layers_per_step",
        type=int,
        default=1,
        help="The number of layers to use per step. If -1, then all layers are used. "
        "If > 0, then a random sample of n_layers_per_step layers are used per step. (only for 2D MRL)",
    )
    parser.add_argument(
        "--last_layer_weight",
        type=float,
        default=1.0,
        help="The weight to use for the loss of the final layer. "
        "Increase this to focus more on the performance when using all layers. (only for 2D MRL)",
    )
    parser.add_argument(
        "--prior_layers_weight",
        type=float,
        default=1.0,
        help="The weight to use for the loss of the prior layers. "
        "Increase this to focus more on the performance when using fewer layers. (only for 2D MRL)",
    )
    parser.add_argument(
        "--kl_div_weight",
        type=float,
        default=1.0,
        help="The weight to use for the KL-div loss that is used to make the prior layers match that of the last layer."
        " Increase this to focus more on the performance when using fewer layers. (only for 2D MRL)",
    )
    parser.add_argument(
        "--kl_temperature",
        type=float,
        default=0.3,
        help="The temperature to use for the KL-divergence loss. "
        "If 0, then the KL-divergence loss is not used. (only for 2D MRL)",
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
        default=CHEM_MRL_DIMENSIONS[0],
        help="Classifier hidden dimension. Must be less than equal to the ChemMRL transformer's hidden dimension."
        " Note, the base model will be truncated to this dimension.",
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


def generate_base_config(args: argparse.Namespace) -> _BaseConfig:
    wandb_config = None
    if args.use_wandb:
        wandb_config = WandbConfig(
            api_key=args.wandb_api_key,
            project_name=args.wandb_project_name,
            run_name=args.wandb_run_name,
            use_watch=args.wandb_use_watch,
            watch_log=args.wandb_watch_log,
            watch_log_freq=args.wandb_watch_log_freq,
            watch_log_graph=args.wandb_watch_log_graph,
        )

    return _BaseConfig(
        model_name=args.model_name,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        test_dataset_path=args.test_dataset_path,
        n_train_samples=args.n_train_samples,
        n_val_samples=args.n_val_samples,
        n_test_samples=args.n_test_samples,
        n_dataloader_workers=args.n_dataloader_workers,
        generate_dataset_examples_at_init=args.generate_dataset_examples_at_init,
        train_batch_size=args.train_batch_size,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb,
        wandb_config=wandb_config,
        lr_base=args.lr_base,
        scheduler=args.scheduler,
        warmup_steps_percent=args.warmup_steps_percent,
        use_fused_adamw=args.use_fused_adamw,
        use_tf32=args.use_tf32,
        use_amp=args.use_amp,
        seed=args.seed,
        model_output_path=args.model_output_path,
        evaluation_steps=args.evaluation_steps,
        checkpoint_save_steps=args.checkpoint_save_steps,
        checkpoint_save_total_limit=args.checkpoint_save_total_limit,
        return_eval_metric=args.return_eval_metric,
    )


def generate_chem_mrl_config(
    args: argparse.Namespace,
) -> ChemMRLConfig | Chem2dMRLConfig:
    base_config = generate_base_config(args)

    chem_mrl_config_params = {
        **base_config.asdict(),
        "smiles_a_column_name": args.smiles_a_column_name,
        "smiles_b_column_name": args.smiles_b_column_name,
        "label_column_name": args.label_column_name,
        "embedding_pooling": args.embedding_pooling,
        "loss_func": args.loss_func,
        "tanimoto_similarity_loss_func": args.tanimoto_similarity_loss_func,
        "eval_similarity_fct": args.eval_similarity_fct,
        "mrl_dimensions": args.mrl_dimensions,
        "use_2d_matryoshka": args.use_2d_matryoshka,
        "mrl_dimension_weights": args.dim_weights,
    }

    if args.use_2d_matryoshka:
        config = Chem2dMRLConfig(
            **chem_mrl_config_params,
            last_layer_weight=args.last_layer_weight,
            prior_layers_weight=args.prior_layers_weight,
        )
    else:
        config = ChemMRLConfig(
            **chem_mrl_config_params,
        )

    return config


def generate_classifier_config(
    args: argparse.Namespace,
) -> ClassifierConfig | DiceLossClassifierConfig:
    base_config = generate_base_config(args)

    classifier_params = {
        **base_config.asdict(),
        "smiles_column_name": args.smiles_column_name,
        "label_column_name": args.label_column_name,
        "loss_func": args.loss_func,
        "classifier_hidden_dimension": args.classifier_hidden_dimension,
        "dropout_p": args.dropout_p,
        "freeze_model": args.freeze_model,
    }

    if args.loss_func == "selfadjdice":
        config = DiceLossClassifierConfig(
            **classifier_params,
            dice_reduction=args.dice_reduction,
            dice_gamma=args.dice_gamma,
        )
    else:
        config = ClassifierConfig(**classifier_params)

    return config
