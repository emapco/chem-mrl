import logging
import os

import optuna
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from chem_mrl.configs import ClassifierConfig, DiceLossClassifierConfig
from chem_mrl.datasets import PandasDataFrameDataset
from chem_mrl.evaluation import LabelAccuracyEvaluator

from .BaseTrainer import _BaseTrainer

logger = logging.getLogger(__name__)


class ClassifierTrainer(_BaseTrainer):
    def __init__(
        self,
        config: ClassifierConfig | DiceLossClassifierConfig,
        optuna_trial: optuna.Trial | None = None,
    ):
        super().__init__(config=config, optuna_trial=optuna_trial)
        self._config = config

        self._model = self._initialize_model()
        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.num_labels,
        ) = self._load_data(
            batch_size=config.train_batch_size,
            train_file=config.train_dataset_path,
            val_file=config.val_dataset_path,
        )
        self.__train_loss = self._initialize_loss()
        self.__val_evaluator = self._initialize_evaluator()
        self.__model_save_path = self._initialize_output_path()

    def _initialize_model(self):
        return SentenceTransformer(
            self._config.model_name,
            truncate_dim=self._config.classifier_hidden_dimension,
        )

    def _load_data(
        self,
        batch_size: int,
        train_file: str,
        val_file: str,
        test_file: str | None = None,
    ):
        train_df = pd.read_parquet(
            train_file,
            columns=[self.config.smiles_column_name, self.config.label_column_name],
        )
        train_dl = DataLoader(
            PandasDataFrameDataset(
                train_df,
                smiles_a_column=self.config.smiles_column_name,
                label_column=self.config.label_column_name,
            ),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            pin_memory_device="cuda",
            num_workers=12,
        )

        val_df = pd.read_parquet(
            val_file,
            columns=[self.config.smiles_column_name, self.config.label_column_name],
        )
        val_dl = DataLoader(
            PandasDataFrameDataset(
                val_df,
                smiles_a_column=self.config.smiles_column_name,
                label_column=self.config.label_column_name,
            ),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            pin_memory_device="cuda",
            num_workers=12,
        )

        if test_file:
            test_df = pd.read_parquet(
                test_file,
                columns=[
                    self.config.smiles_column_name,
                    self.config.label_column_name,
                ],
            )
            test_dl = DataLoader(
                PandasDataFrameDataset(
                    test_df,
                    smiles_a_column=self.config.smiles_column_name,
                    label_column=self.config.label_column_name,
                ),
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                pin_memory_device="cuda",
                num_workers=12,
            )
        else:
            test_dl = None

        num_labels = train_df[self.config.label_column_name].nunique()

        return train_dl, val_dl, test_dl, num_labels

    def _initialize_evaluator(self):
        return LabelAccuracyEvaluator(
            dataloader=self.val_dataloader,
            softmax_model=self.__train_loss,
            write_csv=True,
        )

    def _initialize_loss(self):
        from chem_mrl.losses import SelfAdjDiceLoss, SoftmaxLoss

        if self._config.loss_func == "softmax":
            return SoftmaxLoss(
                model=self._model,
                smiles_embedding_dimension=self._config.classifier_hidden_dimension,
                num_labels=self.num_labels,
                dropout=self._config.dropout_p,
                freeze_model=self._config.freeze_model,
            )

        assert isinstance(self._config, DiceLossClassifierConfig)
        return SelfAdjDiceLoss(
            model=self._model,
            smiles_embedding_dimension=self._config.classifier_hidden_dimension,
            num_labels=self.num_labels,
            dropout=self._config.dropout_p,
            freeze_model=self._config.freeze_model,
            reduction=self._config.dice_reduction,
            gamma=self._config.dice_gamma,
        )

    def _initialize_output_path(self):
        if isinstance(self._config, DiceLossClassifierConfig):
            loss_parameter_str = (
                f"{self._config.dice_reduction}-{self._config.dice_gamma}"
            )
        else:
            loss_parameter_str = ""

        output_path = os.path.join(
            self._config.model_output_path,
            "classifier",
            f"{self._config.model_name.rsplit('/', 1)[1][:20]}"
            f"-{self._config.train_batch_size}"
            f"-{self._config.num_epochs}"
            f"-{self._config.lr_base:6f}"
            f"-{self._config.scheduler}-{self._config.warmup_steps_percent}"
            f"-{self._config.loss_func}-{self._config.dropout_p:3f}"
            f"-{self._config.classifier_hidden_dimension}-{loss_parameter_str}",
        )
        logger.info(f"Output path: {output_path}")
        return output_path

    def fit(self):
        self._fit(
            config=self._config,
            model=self._model,
            train_dataloader=self.train_dataloader,
            train_loss=self.__train_loss,
            val_evaluator=self.__val_evaluator,
            model_save_path=self.__model_save_path,
            wandb_callback=self._wandb_callback,
        )

    @property
    def config(self) -> ClassifierConfig | DiceLossClassifierConfig:
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def loss_fct(self):
        return self.__train_loss

    @property
    def model_save_path(self):
        return self.__model_save_path

    @property
    def eval_file_path(self):
        return os.path.join(
            self.model_save_path,
            "eval",
            "accuracy_evaluation_results.csv",
        )
