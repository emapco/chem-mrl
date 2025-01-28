import logging
from contextlib import nullcontext

import pandas as pd

import wandb
from chem_mrl.configs import BaseConfig

from .BaseTrainer import _BaseTrainer

logger = logging.getLogger(__name__)


class ExecutableTrainer:
    def __init__(self, config: BaseConfig, trainer: _BaseTrainer, return_metric=False):
        self._config = config
        self._trainer = trainer
        self._return_metric = return_metric

    def execute(self) -> float:
        wandb_config = self._config.wandb_config
        wandb_project_name = None
        wandb_run_name = None
        if wandb_config is not None:
            wandb_project_name = wandb_config.project_name
            wandb_run_name = wandb_config.run_name

        with (
            wandb.init(
                project=wandb_project_name,
                name=wandb_run_name,
                config=self._config.asdict(),
            )
            if self._config.use_wandb
            else nullcontext()
        ):
            if (
                self._config.use_wandb
                and wandb_config is not None
                and wandb_config.use_watch
            ):
                wandb.watch(
                    self._trainer.model,
                    criterion=self._trainer.loss_fct,
                    log=wandb_config.watch_log,
                    log_freq=wandb_config.watch_log_freq,
                    log_graph=wandb_config.watch_log_graph,
                )

            self._trainer.fit()

            if self._return_metric:
                metric = self._read_eval_metric(self._trainer.eval_file_path)
                return metric
        return -1

    def _read_eval_metric(self, eval_file_path):
        eval_results_df = pd.read_csv(eval_file_path)
        return float(eval_results_df.iloc[-1]["spearman"])
