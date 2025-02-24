import tempfile
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Generic, TypeVar

import optuna

import wandb
from chem_mrl.schemas import BaseConfig

from .BaseTrainer import BoundTrainerType

BoundTrainerExecutorType = TypeVar("BoundTrainerExecutorType", bound="_BaseTrainerExecutor")


class _BaseTrainerExecutor(ABC, Generic[BoundTrainerType]):
    """Base abstract executor class.
    Executors are used to execute a trainer with additional functionality.
    For example, an executor can be used to execute a trainer within a context manager.
    """

    def __init__(self, trainer: BoundTrainerType):
        self.__trainer = trainer

    @property
    def trainer(self) -> BoundTrainerType:
        return self.__trainer

    @property
    def config(self) -> BaseConfig:
        return self.__trainer.config

    @abstractmethod
    def execute(self) -> float:
        raise NotImplementedError


class TempDirTrainerExecutor(_BaseTrainerExecutor[BoundTrainerType]):
    """
    Executor that runs the trainer within a temporary directory.
    All files stored during execution are removed once the program exits.
    """

    def __init__(self, trainer: BoundTrainerType):
        super().__init__(trainer)
        self._temp_dir = tempfile.TemporaryDirectory()
        self.trainer.model_save_dir = self._temp_dir.name

    def execute(self) -> float:
        """
        Execute the trainer within the temporary directory context.
        """
        try:
            return self.trainer.train()
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """
        Cleanup temporary directory.
        """
        self._temp_dir.cleanup()

    def __del__(self):
        """
        Ensure cleanup occurs when the instance is deleted.
        """
        self.cleanup()


class WandBTrainerExecutor(_BaseTrainerExecutor[BoundTrainerType]):
    def __init__(
        self,
        trainer: BoundTrainerType,
        optuna_trial: optuna.Trial | None = None,
    ):
        super().__init__(trainer)
        self.__wandb_callback = self._signed_in_wandb_callback_factory(
            self.trainer.config, self.trainer.steps_per_epoch, optuna_trial
        )

    def execute(self) -> float:
        wandb_config = self.config.wandb
        wandb_project_name = None
        wandb_run_name = None
        if wandb_config is not None:
            wandb_project_name = wandb_config.project_name
            wandb_run_name = wandb_config.run_name

        # Do not pass unnecessary values to wandb
        parsed_config = self.config.asdict()
        parsed_config.pop("smiles_a_column_name", None)
        parsed_config.pop("smiles_b_column_name", None)
        parsed_config.pop("label_column_name", None)
        parsed_config.pop("n_dataloader_workers", None)
        parsed_config.pop("pin_memory", None)
        parsed_config.pop("generate_dataset_examples_at_init", None)
        parsed_config.pop("model_output_path", None)
        parsed_config.pop("checkpoint_save_steps", None)
        parsed_config.pop("checkpoint_save_total_limit", None)
        parsed_config.pop("show_progress_bar", None)
        parsed_config.pop("wandb", None)

        wandb_enabled = self.config.wandb is not None and self.config.wandb.enabled
        wandb_config = self.config.wandb
        with (
            wandb.init(
                project=wandb_project_name,
                name=wandb_run_name,
                config=parsed_config,
            )
            if wandb_enabled
            else nullcontext()
        ):
            if wandb_enabled and wandb_config and wandb_config.use_watch:
                wandb.watch(
                    self.trainer.model,
                    criterion=self.trainer.loss_functions,
                    log=wandb_config.watch_log.value,
                    log_freq=wandb_config.watch_log_freq,
                    log_graph=wandb_config.watch_log_graph,
                )

            metric = self.trainer.train(eval_callback=self.__wandb_callback)
            return metric

    @staticmethod
    def _signed_in_wandb_callback_factory(
        config: BaseConfig,
        steps_per_epoch: int,
        trial: optuna.Trial | None = None,
    ):
        if config.wandb and config.wandb.enabled:
            wandb_config = config.wandb
            if wandb_config is not None and wandb_config.api_key is not None:
                wandb.login(key=wandb_config.api_key, verify=True)

            # assume user is authenticated either via api_key or env
            def wandb_callback_closure(score: float, epoch: int, steps: int):
                if steps == -1:
                    steps = steps_per_epoch * (epoch + 1)

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
