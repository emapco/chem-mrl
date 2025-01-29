import argparse

from chem_mrl.configs import ArgParseHelper, Chem2dMRLConfig, ChemMRLConfig, WandbConfig
from chem_mrl.trainers import ChemMRLTrainer, WandBTrainerExecutor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SMILES-based MRL embeddings model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = ArgParseHelper.add_base_config_args(parser)
    parser = ArgParseHelper.add_chem_mrl_config_args(parser)
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

    base_config_params = {
        "model_name": args.model_name,
        "train_dataset_path": args.train_dataset_path,
        "val_dataset_path": args.val_dataset_path,
        "test_dataset_path": args.test_dataset_path,
        "num_train_samples": args.num_train_samples,
        "num_val_samples": args.num_val_samples,
        "num_test_samples": args.num_test_samples,
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

    mrl_dimension_weights = (
        args.first_dim_weight,
        args.second_dim_weight,
        args.third_dim_weight,
        args.fourth_dim_weight,
        args.fifth_dim_weight,
        args.sixth_dim_weight,
    )

    chem_mrl_config_params = {
        **base_config_params,
        "smiles_a_column_name": args.smiles_a_column_name,
        "smiles_b_column_name": args.smiles_b_column_name,
        "label_column_name": args.label_column_name,
        "loss_func": args.loss_func,
        "tanimoto_similarity_loss_func": args.tanimoto_similarity_loss_func,
        "eval_similarity_fct": args.eval_similarity_fct,
        "use_2d_matryoshka": args.use_2d_matryoshka,
        "mrl_dimension_weights": mrl_dimension_weights,
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

    executor = WandBTrainerExecutor(trainer=ChemMRLTrainer(config))
    executor.execute()


if __name__ == "__main__":
    main()
