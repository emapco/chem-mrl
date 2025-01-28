import argparse

from chem_mrl.configs import Chem2dMRLConfig, ChemMRLConfig, WandbConfig
from chem_mrl.configs.Base import _scheduler_options, _watch_log_options
from chem_mrl.configs.MRL import (
    _tanimoto_loss_func_options,
    _tanimoto_similarity_base_loss_func_options,
)
from chem_mrl.constants import BASE_MODEL_NAME, CHEM_MRL_DATASET_KEYS
from chem_mrl.trainers import ChemMRLTrainer, ExecutableTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train chemical embeddings model")

    # BaseConfig params
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--lr_base", type=float, default=1.1190785944700813e-05)
    parser.add_argument(
        "--scheduler", choices=_scheduler_options, default="warmuplinear"
    )
    parser.add_argument("--warmup_steps_percent", type=float, default=0.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_output_path", default="output")
    parser.add_argument("--evaluation_steps", type=int, default=0)
    parser.add_argument("--checkpoint_save_steps", type=int, default=1000000)
    parser.add_argument("--checkpoint_save_total_limit", type=int, default=20)

    # WandbConfig params
    parser.add_argument("--wandb_api_key", default=None)
    parser.add_argument("--wandb_project_name", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_use_watch", action="store_true")
    parser.add_argument("--wandb_watch_log", choices=_watch_log_options, default="all")
    parser.add_argument("--wandb_watch_log_freq", type=int, default=1000)
    parser.add_argument("--wandb_watch_log_graph", action="store_true", default=True)

    # ChemMRLConfig params
    parser.add_argument("--dataset_key", choices=CHEM_MRL_DATASET_KEYS, required=True)
    parser.add_argument("--smiles_a_column_name", default="smiles_a")
    parser.add_argument("--smiles_b_column_name", default="smiles_b")
    parser.add_argument("--label_column_name", default="fingerprint_similarity")
    parser.add_argument("--model_name", default=BASE_MODEL_NAME)
    parser.add_argument(
        "--loss_func",
        choices=_tanimoto_loss_func_options,
        default="tanimotosentloss",
    )
    parser.add_argument(
        "--tanimoto_similarity_loss_func",
        choices=_tanimoto_similarity_base_loss_func_options,
        default=None,
    )
    parser.add_argument("--use_2d_matryoshka", action="store_true")

    # Chem2dMRLConfig specific params
    parser.add_argument("--last_layer_weight", type=float, default=1.8708220063487997)
    parser.add_argument("--prior_layers_weight", type=float, default=1.4598249321447245)

    # MRL dimension weights
    parser.add_argument("--first_dim_weight", type=float, default=1.0489590183361719)
    parser.add_argument("--second_dim_weight", type=float, default=1.126163907196291)
    parser.add_argument("--third_dim_weight", type=float, default=1.3807986616809407)
    parser.add_argument("--fourth_dim_weight", type=float, default=1.397331091971628)
    parser.add_argument("--fifth_dim_weight", type=float, default=1.6522851342433993)
    parser.add_argument("--sixth_dim_weight", type=float, default=1.9858679040493405)

    return parser.parse_args()


def main():
    args = parse_args()

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

    mrl_dimension_weights = (
        args.first_dim_weight,
        args.second_dim_weight,
        args.third_dim_weight,
        args.fourth_dim_weight,
        args.fifth_dim_weight,
        args.sixth_dim_weight,
    )

    if args.use_2d_matryoshka:
        config = Chem2dMRLConfig(
            train_batch_size=args.train_batch_size,
            num_epochs=args.num_epochs,
            use_wandb=args.use_wandb,
            wandb_config=wandb_config,
            lr_base=args.lr_base,
            scheduler=args.scheduler,
            warmup_steps_percent=args.warmup_steps_percent,
            use_amp=args.use_amp,
            seed=args.seed,
            model_output_path=args.model_output_path,
            evaluation_steps=args.evaluation_steps,
            checkpoint_save_steps=args.checkpoint_save_steps,
            checkpoint_save_total_limit=args.checkpoint_save_total_limit,
            dataset_key=args.dataset_key,
            smiles_a_column_name=args.smiles_a_column_name,
            smiles_b_column_name=args.smiles_b_column_name,
            label_column_name=args.label_column_name,
            model_name=args.model_name,
            loss_func=args.loss_func,
            tanimoto_similarity_loss_func=args.tanimoto_similarity_loss_func,
            mrl_dimension_weights=mrl_dimension_weights,
            last_layer_weight=args.last_layer_weight,
            prior_layers_weight=args.prior_layers_weight,
        )
    else:
        config = ChemMRLConfig(
            train_batch_size=args.train_batch_size,
            num_epochs=args.num_epochs,
            use_wandb=args.use_wandb,
            wandb_config=wandb_config,
            lr_base=args.lr_base,
            scheduler=args.scheduler,
            warmup_steps_percent=args.warmup_steps_percent,
            use_amp=args.use_amp,
            seed=args.seed,
            model_output_path=args.model_output_path,
            evaluation_steps=args.evaluation_steps,
            checkpoint_save_steps=args.checkpoint_save_steps,
            checkpoint_save_total_limit=args.checkpoint_save_total_limit,
            dataset_key=args.dataset_key,
            smiles_a_column_name=args.smiles_a_column_name,
            smiles_b_column_name=args.smiles_b_column_name,
            label_column_name=args.label_column_name,
            model_name=args.model_name,
            loss_func=args.loss_func,
            tanimoto_similarity_loss_func=args.tanimoto_similarity_loss_func,
            mrl_dimension_weights=mrl_dimension_weights,
        )

    trainer = ChemMRLTrainer(config)
    executor = ExecutableTrainer(config, trainer=trainer, return_metric=True)
    executor.execute()


if __name__ == "__main__":
    args = parse_args()
