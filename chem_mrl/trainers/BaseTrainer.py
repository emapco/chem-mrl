import math
import os
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

import torch
import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator

from chem_mrl.configs import BoundConfigType

BoundTrainerType = TypeVar("BoundTrainerType", bound="_BaseTrainer")


class _BaseTrainer(ABC, Generic[BoundConfigType]):
    def __init__(
        self,
        config: BoundConfigType,
    ):
        self._config = config
        if self._config.seed is not None:
            transformers.set_seed(self._config.seed)
        if self._config.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if self._config.use_fused_adamw:
            from apex.optimizers import FusedAdam

            self.__optimizer = FusedAdam
        else:

            self.__optimizer = torch.optim.AdamW

    ############################################################################
    # abstract properties
    ############################################################################

    @property
    @abstractmethod
    def config(self) -> BoundConfigType:
        pass

    @property
    @abstractmethod
    def model(self) -> SentenceTransformer:
        pass

    @property
    @abstractmethod
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        pass

    @property
    @abstractmethod
    def loss_fct(self) -> torch.nn.Module:
        pass

    @property
    @abstractmethod
    def val_evaluator(self) -> SentenceEvaluator:
        pass

    @property
    @abstractmethod
    def model_save_dir_name(self) -> str:
        pass

    @property
    @abstractmethod
    def steps_per_epoch(self) -> int:
        pass

    @property
    @abstractmethod
    def eval_metric(self) -> str:
        pass

    @property
    @abstractmethod
    def eval_file_path(self) -> str:
        pass

    ############################################################################
    # abstract methods
    ############################################################################
    @abstractmethod
    def _initialize_model(self):
        pass

    @abstractmethod
    def _initialize_data(self, train_file: str, val_file: str, test_file: str):
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

    ############################################################################
    # concrete methods
    ############################################################################

    def __calculate_training_params(self) -> tuple[float, float, int]:
        total_training_points = self.steps_per_epoch * self.config.train_batch_size
        # normalized weight decay for adamw optimizer - https://arxiv.org/pdf/1711.05101.pdf
        # optimized hyperparameter lambda_norm = 0.05 for AdamW optimizer
        weight_decay = 0.05 * math.sqrt(
            self.config.train_batch_size
            / (total_training_points * self.config.num_epochs)
        )
        learning_rate = self.config.lr_base * math.sqrt(self.config.train_batch_size)
        warmup_steps = math.ceil(
            self.steps_per_epoch
            * self.config.num_epochs
            * self.config.warmup_steps_percent
        )
        return learning_rate, weight_decay, warmup_steps

    def fit(self, eval_callback: Callable | None):
        learning_rate, weight_decay, warmup_steps = self.__calculate_training_params()
        model_output_path = os.path.join(
            self.model_save_dir_name, self._config.model_output_path
        )

        optimizer_params: dict[str, object] = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        if self.config.use_fused_adamw and not isinstance(
            self.__optimizer, torch.optim.AdamW
        ):
            # FusedAdam requires adam_w_mode flag
            optimizer_params["adam_w_mode"] = True

        self.model.fit(
            train_objectives=[(self.train_dataloader, self.loss_fct)],
            evaluator=self.val_evaluator,
            evaluation_steps=self._config.evaluation_steps,
            epochs=self._config.num_epochs,
            warmup_steps=warmup_steps,
            output_path=self.model_save_dir_name,
            optimizer_class=self.__optimizer,  # type: ignore - Library defaults to AdamW. We can safely ignore error
            optimizer_params=optimizer_params,
            save_best_model=True,
            use_amp=self._config.use_amp,
            show_progress_bar=True,
            scheduler=self._config.scheduler,
            checkpoint_path=model_output_path,
            checkpoint_save_steps=self._config.checkpoint_save_steps,
            checkpoint_save_total_limit=self._config.checkpoint_save_total_limit,
            callback=eval_callback,  # type: ignore - Library defaults to AdamW. We can safely ignore error
        )
