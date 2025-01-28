import math
import os
from abc import ABC, abstractmethod
from typing import Callable

import optuna
import transformers
from apex.optimizers import FusedAdam
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from torch import nn
from torch.utils.data import DataLoader

import wandb
from chem_mrl.configs import BaseConfig


class _BaseTrainer(ABC):
    def __init__(
        self,
        config: BaseConfig,
        optuna_trial: optuna.Trial | None = None,
    ):
        self._config = config
        if self._config.seed is not None:
            transformers.set_seed(self._config.seed)
        self._wandb_callback = self._get_signed_in_wandb_callback(config, optuna_trial)

    @abstractmethod
    def _initialize_model(self):
        pass

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def _initialize_evaluator(self):
        pass

    @abstractmethod
    def _initialize_loss(self):
        pass

    @abstractmethod
    def _initialize_output_path(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @property
    @abstractmethod
    def config(self):
        pass

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def loss_fct(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def model_save_path(self) -> str:
        pass

    @property
    @abstractmethod
    def eval_file_path(self) -> str:
        pass

    @staticmethod
    def _calculate_training_params(
        train_dataloader: DataLoader,
        config: BaseConfig,
    ) -> tuple[float, float, int]:
        total_training_points = len(train_dataloader) * config.train_batch_size
        # normalized weight decay for adamw optimizer - https://arxiv.org/pdf/1711.05101.pdf
        # optimized hyperparameter lambda_norm = 0.05 for AdamW optimizer
        weight_decay = 0.05 * math.sqrt(
            config.train_batch_size / (total_training_points * config.num_epochs)
        )
        learning_rate = config.lr_base * math.sqrt(config.train_batch_size)
        warmup_steps = math.ceil(
            len(train_dataloader) * config.num_epochs * config.warmup_steps_percent
        )
        return learning_rate, weight_decay, warmup_steps

    @staticmethod
    def _get_signed_in_wandb_callback(
        config: BaseConfig,
        trial: optuna.Trial | None = None,
    ):
        if config.use_wandb:
            wandb_config = config.wandb_config
            if wandb_config is not None and wandb_config.api_key is not None:
                wandb.login(key=wandb_config.api_key, verify=True)

            # assume user is authenticated either via api_key or env
            def wandb_callback_closure(score: float, epoch: int, steps: int):
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

        else:

            def wandb_callback_closure(score: float, epoch: int, steps: int):
                pass

        return wandb_callback_closure

    @classmethod
    def _fit(
        cls,
        config: BaseConfig,
        model: SentenceTransformer,
        train_dataloader: DataLoader,
        train_loss: nn.Module,
        val_evaluator: SentenceEvaluator,
        model_save_path: str,
        wandb_callback: Callable,
    ):
        learning_rate, weight_decay, warmup_steps = cls._calculate_training_params(
            train_dataloader=train_dataloader, config=config
        )
        model_output_path = os.path.join(model_save_path, config.model_output_path)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=val_evaluator,
            evaluation_steps=config.evaluation_steps,
            epochs=config.num_epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_class=FusedAdam,
            optimizer_params={
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "adam_w_mode": True,
            },
            save_best_model=True,
            use_amp=config.use_amp,
            show_progress_bar=True,
            scheduler=config.scheduler,
            checkpoint_path=model_output_path,
            checkpoint_save_steps=config.checkpoint_save_steps,
            checkpoint_save_total_limit=config.checkpoint_save_total_limit,
            callback=wandb_callback,
        )
