import logging
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

import optuna
import pandas as pd
import torch
from datasets import Dataset
from hydra.utils import instantiate
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SentenceEvaluator
from transformers.data.data_collator import DataCollator
from transformers.integrations.integration_utils import WandbCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import EvalPrediction

from chem_mrl.schemas import BaseConfig
from chem_mrl.similarity_functions import patch_sentence_transformer

logger = logging.getLogger(__name__)

BoundTrainerType = TypeVar("BoundTrainerType", bound="_BaseTrainer")

patch_sentence_transformer()


class _BaseTrainer(ABC):
    """Base abstract trainer class.
    Concrete trainer classes can be trained directly (via fit method) or through an executor.
    """

    _is_testing = False
    A_COL = "sentence_A"
    B_COL = "sentence_B"
    LABEL_COL = "label"
    CHECKPOINTS_DIR = "checkpoints"

    def __init__(
        self,
        config: BaseConfig,
    ):
        self._config = config
        if isinstance(self._config.training_args, SentenceTransformerTrainingArguments):
            self._training_args = self._config.training_args
        else:
            self._training_args: SentenceTransformerTrainingArguments = instantiate(
                self._config.training_args
            )
        self._root_output_dir = self._training_args.output_dir or "."

    ############################################################################
    # abstract properties
    ############################################################################

    @property
    @abstractmethod
    def config(self) -> BaseConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    @abstractmethod
    def eval_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> SentenceTransformer:
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
    def loss_function(self) -> torch.nn.Module:
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

    ############################################################################
    # abstract methods
    ############################################################################

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractmethod
    def _init_data(self, train_file: str, val_file: str, test_file: str):
        raise NotImplementedError

    @abstractmethod
    def _init_val_evaluator(self):
        raise NotImplementedError

    @abstractmethod
    def _init_test_evaluator(self):
        raise NotImplementedError

    @abstractmethod
    def _init_loss(self):
        raise NotImplementedError

    ############################################################################
    # concrete methods
    ############################################################################

    def __calculate_training_params(self) -> tuple[float, float]:
        total_training_points = (
            self.steps_per_epoch * self._training_args.per_device_train_batch_size
        )
        # Normalized weight decay for adamw optimizer - https://arxiv.org/pdf/1711.05101.pdf
        # optimized hyperparameter lambda_norm = 0.05 for AdamW optimizer
        # Hyperparameter search indicates a normalized weight decay outperforms
        # the default adamw weight decay
        sqrt_batch_size = math.sqrt(self._training_args.per_device_train_batch_size)
        denominator = total_training_points * self._training_args.num_train_epochs
        weight_decay = 0.05 * math.sqrt(
            self._training_args.per_device_train_batch_size / denominator
        )
        learning_rate = self._training_args.learning_rate * sqrt_batch_size
        return learning_rate, weight_decay

    def _write_chem_mrl_config(self):
        import json

        from chem_mrl import __version__

        config_file_name = os.path.join(self._root_output_dir, "config_chem_mrl.json")
        parsed_config: dict = self.config.model.asdict()
        parsed_config["__version__"] = __version__
        parsed_config = dict(sorted(parsed_config.items()))
        with open(config_file_name, "w") as f:
            json.dump(parsed_config, f, indent=4)

    def _read_eval_metric(self, file_path: str | None = None) -> float:
        if file_path is None:
            # sentence transformers adds the 'eval' directory to the file path
            # when it call the evaluator within the training loop
            file_path = os.path.join(
                self._root_output_dir,
                self.CHECKPOINTS_DIR,
                "eval",
                self.val_eval_file_path,
            )
        eval_results_df = pd.read_csv(file_path)
        return float(eval_results_df.iloc[-1][self.eval_metric])

    def train(
        self,
        trial: optuna.Trial | dict[str, Any] = None,  # type: ignore
        data_collator: DataCollator | None = None,  # type: ignore
        tokenizer: PreTrainedTokenizerBase | Callable | None = None,
        model_init: Callable[[], SentenceTransformer] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),  # type: ignore - take from sentence_transformers
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ):
        """
        Main training entry point.

        Args:
            # passed to SentenceTransformerTrainer.train()
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            # passed to SentenceTransformerTrainer()
            data_collator (`DataCollator`, *optional*):
                The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
                default to [`default_data_collator`] if no `processing_class` is provided, an instance of
                [`DataCollatorWithPadding`] otherwise if the processing_class is a feature extractor or tokenizer.
            tokenizer (`PreTrainedTokenizerBase` or `Callable`, *optional*):
                The tokenizer that will be used by this trainer to encode the data. Will default to the tokenizer
                corresponding to the first model of the model list if `tokenizer` is not provided.
            model_init (`Callable[[], PreTrainedModel]`, *optional*):
                A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
                from a new instance of the model as given by this function.

                The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
                be able to choose different architectures according to hyper parameters (such as layer count, sizes of
                inner layers, dropout probabilities etc).
            compute_loss_func (`Callable`, *optional*):
                A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
                batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618) used by [`Trainer`].
            compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
                The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
                a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
                `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
                after the last eval batch to signal that the function needs to calculate and return the global summary
                statistics rather than accumulating the batch-level statistics
            callbacks (List of [`TrainerCallback`], *optional*):
                A list of callbacks to customize the training loop. Will add those to the list of default callbacks
                detailed in [here](callback).

                If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
                A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
                model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
                A function that preprocess the logits right before caching them at each evaluation step. Must take two
                tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
                by this function will be reflected in the predictions received by `compute_metrics`.

                Note that the labels (second parameter) will be `None` if the dataset does not have them.
        """  # noqa: E501
        scaled_learning_rate, normalized_weight_decay = self.__calculate_training_params()

        self._training_args.output_dir = os.path.join(self._root_output_dir, self.CHECKPOINTS_DIR)
        if self.config.use_normalized_weight_decay:
            self._training_args.weight_decay = normalized_weight_decay
        if self.config.scale_learning_rate:
            self._training_args.learning_rate = scaled_learning_rate
        if callbacks is None and self.config.early_stopping_patience is not None:
            callbacks = [
                EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)
            ]

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=self._training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            loss=self.loss_function,
            evaluator=self.val_evaluator,
            data_collator=data_collator,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if self._is_testing:
            trainer.remove_callback(WandbCallback)

        self._write_chem_mrl_config()
        trainer.train(
            resume_from_checkpoint=self._training_args.resume_from_checkpoint, trial=trial
        )
        trainer.save_model(self._root_output_dir)

        if self.test_evaluator is not None:
            model = SentenceTransformer(self._root_output_dir)
            test_dir = os.path.join(self._root_output_dir, "test")
            self.test_evaluator(model, output_path=test_dir)
            test_path = os.path.join(test_dir, self.test_evaluator.csv_file)  # type: ignore
            metric = self._read_eval_metric(test_path)
            return metric

        metric = self._read_eval_metric()
        return metric
