import logging
import os

import optuna
from constants import OUTPUT_MODEL_DIR
from sentence_transformers import SentenceTransformer
from torch import nn

import wandb

logger = logging.getLogger(__name__)


def get_train_loss(
    model: SentenceTransformer,
    base_loss: nn.Module,
    use_2d_matryoshka: bool,
    dimensions: list[int],
    matryoshka_weights: list[float],
    last_layer_weight: float,
    prior_layers_weight: float,
):
    from sentence_transformers import losses

    if use_2d_matryoshka:
        return losses.Matryoshka2dLoss(
            model,
            base_loss,
            dimensions,
            matryoshka_weights=matryoshka_weights,
            n_layers_per_step=-1,
            n_dims_per_step=-1,
            last_layer_weight=last_layer_weight,
            prior_layers_weight=prior_layers_weight,
        )
    return losses.MatryoshkaLoss(
        model,
        base_loss,
        dimensions,
        matryoshka_weights=matryoshka_weights,
        n_dims_per_step=-1,
    )


def get_base_loss(
    model: SentenceTransformer,
    loss_func: str,
    tanimoto_similarity_loss_func: str | None,
) -> nn.Module:
    from sentence_transformers import losses
    from tanimoto_loss import TanimotoLoss, TanimotoSimilarityLoss

    LOSS_FUNCTIONS = {
        "tanimotoloss": lambda model: TanimotoLoss(model),
        "cosentloss": lambda model: losses.CoSENTLoss(model),
        "tanimotosimilarityloss": {
            "mse": lambda model: TanimotoSimilarityLoss(model, loss=nn.MSELoss()),
            "l1": lambda model: TanimotoSimilarityLoss(model, loss=nn.L1Loss()),
            "smooth_l1": lambda model: TanimotoSimilarityLoss(
                model, loss=nn.SmoothL1Loss()
            ),
            "huber": lambda model: TanimotoSimilarityLoss(model, loss=nn.HuberLoss()),
            "bin_cross_entropy": lambda model: TanimotoSimilarityLoss(
                model, loss=nn.BCEWithLogitsLoss()
            ),
            "kldiv": lambda model: TanimotoSimilarityLoss(
                model, loss=nn.KLDivLoss(reduction="batchmean")
            ),
            "cosine_embedding_loss": lambda model: TanimotoSimilarityLoss(
                model, loss=nn.CosineEmbeddingLoss()
            ),
        },
    }
    if loss_func in ["tanimotoloss", "cosentloss"]:
        return LOSS_FUNCTIONS[loss_func](model)

    return LOSS_FUNCTIONS["tanimotosimilarityloss"][tanimoto_similarity_loss_func](
        model
    )


def get_model_save_path(
    param_config: dict,
    output_model_dir: str = OUTPUT_MODEL_DIR,
) -> str:
    model_save_path = os.path.join(
        output_model_dir,
        f"{param_config['dataset_key']}-chem-{'2D' if param_config['use_2d_matryoshka'] else '1D'}mrl"
        f"-{param_config['train_batch_size']}-{param_config['num_epochs']}"
        f"-{param_config['lr_base']:6f}-{param_config['scheduler']}-{param_config['warmup_steps_percent']}"
        f"-{param_config['loss_func']}-{param_config['tanimoto_similarity_loss_func']}"
        f"-{param_config['last_layer_weight']:4f}-{param_config['prior_layers_weight']:4f}"
        f"-{param_config['first_dim_weight']:4f}-{param_config['second_dim_weight']:4f}-{param_config['third_dim_weight']:4f}"  # noqa: E501
        f"-{param_config['fourth_dim_weight']:4f}-{param_config['fifth_dim_weight']:4f}-{param_config['sixth_dim_weight']:4f}",  # noqa: E501
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
