import gc
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Callable, Generic, TypeVar

import optuna
import pandas as pd
import torch

import wandb
from chem_mrl.configs import BoundConfigType
from chem_mrl.trainers import BoundTrainerType

BoundTrainerExecutorType = TypeVar(
    "BoundTrainerExecutorType", bound="_BaseTrainerExecutor"
)


class _BaseTrainerExecutor(ABC, Generic[BoundTrainerType, BoundConfigType]):
    def __init__(self, trainer: BoundTrainerType):
        self.__trainer = trainer

    @property
    def trainer(self) -> BoundTrainerType:
        return self.__trainer

    @property
    def config(self) -> BoundConfigType:
        return self.__trainer.config

    @abstractmethod
    def execute(self, return_eval_metric: bool) -> float:
        pass

    @staticmethod
    def _read_eval_metric(eval_file_path, eval_metric: str) -> float:
        eval_results_df = pd.read_csv(eval_file_path)
        return float(eval_results_df.iloc[-1][eval_metric])

    @staticmethod
    def _clear_memory():
        # OOM issues in between epochs
        torch.cuda.synchronize()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        gc.collect()


class CallbackTrainerExecutor(_BaseTrainerExecutor[BoundTrainerType, BoundConfigType]):
    def __init__(
        self,
        trainer: BoundTrainerType,
        eval_callback: Callable[[float, int, int], None] | None = None,
    ):
        super().__init__(trainer)
        if eval_callback is not None and not callable(eval_callback):
            raise ValueError("eval_callback must be callable")
        self.__eval_callback = eval_callback

    def execute(self, return_eval_metric=False) -> float:
        self.trainer.fit(eval_callback=self.__eval_callback)

        if return_eval_metric:
            metric = self._read_eval_metric(
                self.trainer.eval_file_path, self.trainer.eval_metric
            )
            return metric
        return -1.0


class WandBTrainerExecutor(_BaseTrainerExecutor[BoundTrainerType, BoundConfigType]):
    def __init__(
        self,
        trainer: BoundTrainerType,
        optuna_trial: optuna.Trial | None = None,
    ):
        super().__init__(trainer)
        self.__wandb_callback = self._get_signed_in_wandb_callback(
            self.trainer.config,
            self.trainer.steps_per_epoch,
            optuna_trial,
            self._clear_memory,
        )

    def execute(self, return_eval_metric=False) -> float:
        wandb_config = self.config.wandb_config
        wandb_project_name = None
        wandb_run_name = None
        if wandb_config is not None:
            wandb_project_name = wandb_config.project_name
            wandb_run_name = wandb_config.run_name

        # Do not pass unnecessary values to wandb
        config_without_wandb = self.config.asdict()
        config_without_wandb.pop("use_wandb", None)
        config_without_wandb.pop("wandb_config", None)

        with (
            wandb.init(
                project=wandb_project_name,
                name=wandb_run_name,
                config=config_without_wandb,
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

            self.trainer.fit(eval_callback=self.__wandb_callback)

            if return_eval_metric:
                metric = self._read_eval_metric(
                    self.trainer.eval_file_path, self.trainer.eval_metric
                )
                return metric
        return -1.0

    @staticmethod
    def _get_signed_in_wandb_callback(
        config: BoundConfigType,
        steps_per_epoch: int,
        trial: optuna.Trial | None = None,
        clear_memory_callback: Callable[[], None] | None = None,
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

                # OOM issues in between epochs
                if clear_memory_callback is not None and callable(
                    clear_memory_callback
                ):
                    clear_memory_callback()

                if trial is not None:
                    trial.report(score, steps)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

        else:

            def wandb_callback_closure(score: float, epoch: int, steps: int):
                pass

        return wandb_callback_closure
