from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Callable, Generic, TypeVar

import optuna
import wandb

from chem_mrl.configs import BoundConfigType

from .BaseTrainer import BoundTrainerType

BoundTrainerExecutorType = TypeVar(
    "BoundTrainerExecutorType", bound="_BaseTrainerExecutor"
)


class _BaseTrainerExecutor(ABC, Generic[BoundTrainerType, BoundConfigType]):
    """Base abstract executor class.
    Concrete executor classes should inherit from this class and implement the abstract methods and properties.

    Executors are used to execute a trainer with additional functionality.
    For example, an executor can be used to execute a trainer within a context manager.
    """

    def __init__(self, trainer: BoundTrainerType):
        self.__trainer = trainer

    @property
    def trainer(self) -> BoundTrainerType:
        return self.__trainer

    @property
    def config(self) -> BoundConfigType:
        return self.__trainer.config

    @abstractmethod
    def execute(self) -> float:
        raise NotImplementedError


class WandBTrainerExecutor(_BaseTrainerExecutor[BoundTrainerType, BoundConfigType]):
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
        wandb_config = self.config.wandb_config
        wandb_project_name = None
        wandb_run_name = None
        if wandb_config is not None:
            wandb_project_name = wandb_config.project_name
            wandb_run_name = wandb_config.run_name

        # Do not pass unnecessary values to wandb
        parsed_config = self.config.asdict()
        parsed_config.pop("use_wandb", None)
        parsed_config.pop("wandb_config", None)
        parsed_config.pop("return_eval_metric", None)
        parsed_config.pop("n_dataloader_workers", None)
        parsed_config.pop("generate_dataset_examples_at_init", None)
        parsed_config.pop("evaluation_steps", None)
        parsed_config.pop("checkpoint_save_steps", None)
        parsed_config.pop("checkpoint_save_total_limit", None)
        parsed_config.pop("model_output_path", None)

        with (
            wandb.init(
                project=wandb_project_name,
                name=wandb_run_name,
                config=parsed_config,
            )
            if self.config.use_wandb
            else nullcontext()
        ):
            if (
                self.config.use_wandb
                and wandb_config is not None
                and wandb_config.use_watch
            ):
                wandb.watch(
                    self.trainer.model,
                    criterion=self.trainer.loss_fct,
                    log=wandb_config.watch_log,
                    log_freq=wandb_config.watch_log_freq,
                    log_graph=wandb_config.watch_log_graph,
                )

            metric = self.trainer.train(eval_callback=self.__wandb_callback)
            return metric

    @staticmethod
    def _signed_in_wandb_callback_factory(
        config: BoundConfigType,
        steps_per_epoch: int,
        trial: optuna.Trial | None = None,
    ):
        if config.use_wandb:
            wandb_config = config.wandb_config
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
