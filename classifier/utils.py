import logging
import os

from torch import nn
from sentence_transformers import SentenceTransformer
import optuna
import wandb

from constants import OUTPUT_MODEL_DIR

logger = logging.getLogger(__name__)


def get_train_loss(
    model: SentenceTransformer,
    smiles_embedding_dimension: int,
    num_labels: int,
    loss_func: str,
    dropout: float,
    dice_reduction: str | None = None,
    dice_gamma: float | None = None,
) -> nn.Module:
    from loss import SoftmaxLoss, SelfAdjDiceLoss

    if loss_func == "SoftMax":
        return SoftmaxLoss(
            model=model,
            smiles_embedding_dimension=smiles_embedding_dimension,
            num_labels=num_labels,
            dropout=dropout,
        )
    return SelfAdjDiceLoss(
        model=model,
        smiles_embedding_dimension=smiles_embedding_dimension,
        num_labels=num_labels,
        reduction=dice_reduction or "mean",
        gamma=dice_gamma or 1.0,
        dropout=dropout,
    )


def get_model_save_path(
    param_config: dict,
    output_model_dir: str = OUTPUT_MODEL_DIR,
) -> str:
    loss_parameter_str = (
        f"{param_config['dice_reduction']}-{param_config['dice_gamma']}"
        if param_config["loss_func"] == "SelfAdjDice"
        else ""
    )

    model_save_path = os.path.join(
        output_model_dir,
        "classifier",
        f"{param_config['model_name'].rsplit('/', 1)[1][:20]}"
        f"-{param_config['train_batch_size']}"
        f"-{param_config['num_epochs']}"
        f"-{param_config['lr_base']:6f}"
        f"-{param_config['scheduler']}-{param_config['warmup_steps_percent']}"
        f"-{param_config['loss_func']}-{param_config['dropout_p']:3f}"
        f"-{param_config['matryoshka_dim']}-{loss_parameter_str}",
    )
    logger.info(model_save_path)
    return model_save_path


def get_signed_in_wandb_callback(
    train_dataloader,
    use_wandb=True,
    wandb_api_key: str | None = None,
    trial: optuna.Trial | None = None,
):
    if use_wandb:
        if wandb_api_key is not None:
            wandb.login(key=wandb_api_key, verify=True)

        # assume user is authenticated either via api_key or env
        def wandb_callback(score: float, epoch: int, steps: int):
            if steps == -1:
                steps = (epoch + 1) * len(train_dataloader)
            eval_dict = {
                "score": score,
                "epoch": epoch,
                "steps": steps,
            }
            wandb.log(eval_dict)
            if trial is not None:
                trial.report(score, steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return wandb_callback

    def wandb_callback(score: float, epoch: int, steps: int):
        pass

    return wandb_callback
