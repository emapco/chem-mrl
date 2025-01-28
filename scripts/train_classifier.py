import argparse

import transformers

from chem_mrl.configs import ClassifierConfig, DiceLossClassifierConfig, WandbConfig
from chem_mrl.configs.Base import _scheduler_options, _watch_log_options
from chem_mrl.configs.Classifier import (
    _classifier_loss_func_options,
    _dice_reduction_options,
)
from chem_mrl.constants import CHEM_MRL_DIMENSIONS, MODEL_NAME_KEYS, MODEL_NAMES
from chem_mrl.trainers import ClassifierTrainer, ExecutableTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier model")
    # Base config params
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--lr_base", type=float, default=1.1190785944700813e-05)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=_scheduler_options,
        default="warmuplinear",
    )
    parser.add_argument("--warmup_steps_percent", type=float, default=0.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_output_path", default="output")
    parser.add_argument("--evaluation_steps", type=int, default=0)
    parser.add_argument("--checkpoint_save_steps", type=int, default=1000000)
    parser.add_argument("--checkpoint_save_total_limit", type=int, default=20)

    # Wandb config params
    parser.add_argument("--wandb_api_key", type=str)
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--wandb_use_watch", action="store_true")
    parser.add_argument(
        "--wandb_watch_log",
        type=str,
        choices=_watch_log_options,
        default="all",
    )
    parser.add_argument("--wandb_watch_log_freq", type=int, default=1000)
    parser.add_argument("--wandb_watch_log_graph", action="store_true", default=True)

    # Classifier config params
    parser.add_argument(
        "--model_name", type=str, default=MODEL_NAMES[MODEL_NAME_KEYS[1]]
    )
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, required=True)
    parser.add_argument("--smiles_column_name", type=str, default="smiles")
    parser.add_argument("--label_column_name", type=str, default="label")
    parser.add_argument(
        "--loss_func",
        type=str,
        choices=_classifier_loss_func_options,
        default="softmax",
    )
    parser.add_argument(
        "--classifier_hidden_dimension",
        type=int,
        choices=CHEM_MRL_DIMENSIONS,
        default=CHEM_MRL_DIMENSIONS[0],
    )
    parser.add_argument("--dropout_p", type=float, default=0.15)
    parser.add_argument("--freeze_model", action="store_true")

    # DiceLoss specific params
    parser.add_argument(
        "--dice_reduction", type=str, choices=_dice_reduction_options, default="mean"
    )
    parser.add_argument("--dice_gamma", type=float, default=1.0)

    return parser.parse_args()


def train(args):
    if args.seed is not None:
        transformers.set_seed(args.seed)

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

    base_config_params = {
        "train_batch_size": args.train_batch_size,
        "num_epochs": args.num_epochs,
        "use_wandb": args.use_wandb,
        "wandb_config": wandb_config,
        "lr_base": args.lr_base,
        "scheduler": args.scheduler,
        "warmup_steps_percent": args.warmup_steps_percent,
        "use_amp": args.use_amp,
        "seed": args.seed,
        "model_output_path": args.model_output_path,
        "evaluation_steps": args.evaluation_steps,
        "checkpoint_save_steps": args.checkpoint_save_steps,
        "checkpoint_save_total_limit": args.checkpoint_save_total_limit,
    }

    classifier_params = {
        **base_config_params,
        "model_name": args.model_name,
        "train_dataset_path": args.train_dataset_path,
        "val_dataset_path": args.val_dataset_path,
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

    classifier = ClassifierTrainer(config)
    trainer = ExecutableTrainer(config, trainer=classifier, return_metric=True)
    trainer.execute()


if __name__ == "__main__":
    args = parse_args()
    train(args)
