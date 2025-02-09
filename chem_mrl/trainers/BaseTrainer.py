import math
import os
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import pandas as pd
import torch
import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator

from chem_mrl.schemas import BaseConfig

BoundTrainerType = TypeVar("BoundTrainerType", bound="_BaseTrainer")


class _BaseTrainer(ABC):
    """Base abstract trainer class.
    Concrete trainer classes can be trained directly (via fit method) or through an executor.
    """

    def __init__(
        self,
        config: BaseConfig,
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
    def config(self) -> BaseConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> SentenceTransformer:
        raise NotImplementedError

    @property
    @abstractmethod
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    @property
    @abstractmethod
    def loss_fct(self) -> torch.nn.Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def val_evaluator(self) -> SentenceEvaluator:
        raise NotImplementedError

    @property
    @abstractmethod
    def test_evaluator(self) -> SentenceEvaluator | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_save_dir(self) -> str:
        raise NotImplementedError

    @model_save_dir.setter
    @abstractmethod
    def model_save_dir(self, value: str):
        raise NotImplementedError

    @property
    @abstractmethod
    def steps_per_epoch(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def eval_metric(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def val_eval_file_path(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def test_eval_file_path(self) -> str:
        raise NotImplementedError

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
    def _initialize_val_evaluator(self):
        pass

    @abstractmethod
    def _initialize_test_evaluator(self):
        pass

    @abstractmethod
    def _initialize_loss(self):
        raise NotImplementedError

    @abstractmethod
    def _initialize_output_path(self):
        raise NotImplementedError

    ############################################################################
    # concrete methods
    ############################################################################

    def __calculate_training_params(self) -> tuple[float, float, int]:
        total_training_points = self.steps_per_epoch * self.config.train_batch_size
        # Normalized weight decay for adamw optimizer - https://arxiv.org/pdf/1711.05101.pdf
        # optimized hyperparameter lambda_norm = 0.05 for AdamW optimizer
        # Hyperparameter search indicates a normalized weight decay outperforms
        # the default adamw weight decay
        weight_decay = 0.05 * math.sqrt(
            self.config.train_batch_size / (total_training_points * self.config.num_epochs)
        )
        learning_rate = self.config.lr_base * math.sqrt(self.config.train_batch_size)
        warmup_steps = math.ceil(
            self.steps_per_epoch * self.config.num_epochs * self.config.warmup_steps_percent
        )
        return learning_rate, weight_decay, warmup_steps

    def _write_config(self):
        config_file_name = os.path.join(self.model_save_dir, "chem_mrl_config.json")
        parsed_config = self.config.model.asdict()
        parsed_config.pop("model_name", None)

        import json

        with open(config_file_name, "w") as f:
            json.dump(parsed_config, f, indent=2)

    @staticmethod
    def _read_eval_metric(eval_file_path, eval_metric: str) -> float:
        eval_results_df = pd.read_csv(eval_file_path)
        return float(eval_results_df.iloc[-1][eval_metric])

    def train(self, eval_callback: Callable[[float, int, int], None] | None = None):
        learning_rate, weight_decay, warmup_steps = self.__calculate_training_params()
        if self.config.weight_decay is not None:
            weight_decay = self.config.weight_decay

        optimizer_params: dict[str, object] = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }

        self.model.old_fit(
            train_objectives=[(self.train_dataloader, self.loss_fct)],
            evaluator=self.val_evaluator,
            epochs=self._config.num_epochs,
            scheduler=self._config.scheduler,
            warmup_steps=warmup_steps,
            optimizer_class=self.__optimizer,  # type: ignore - Library defaults to AdamW. We can safely ignore error
            optimizer_params=optimizer_params,
            weight_decay=weight_decay,
            evaluation_steps=self._config.evaluation_steps,
            output_path=self.model_save_dir,
            save_best_model=True,
            use_amp=self._config.use_amp,
            callback=eval_callback,  # type: ignore - Library defaults to None. We can safely ignore error
            show_progress_bar=self.config.show_progress_bar,
            checkpoint_path=os.path.join(self.model_save_dir, "checkpoints"),
            checkpoint_save_steps=self._config.checkpoint_save_steps,
            checkpoint_save_total_limit=self._config.checkpoint_save_total_limit,
        )

        self._write_config()

        if self.test_evaluator is not None:
            model = SentenceTransformer(self.model_save_dir)
            self.test_evaluator(model, output_path=os.path.join(self.model_save_dir, "eval"))
            metric = self._read_eval_metric(self.test_eval_file_path, self.eval_metric)
            return metric

        metric = self._read_eval_metric(self.val_eval_file_path, self.eval_metric)
        return metric
