import logging
import os
from dataclasses import dataclass

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from chem_mrl.configs import ClassifierConfig, DiceLossClassifierConfig
from chem_mrl.datasets import PandasDataFrameDataset
from chem_mrl.evaluation import LabelAccuracyEvaluator

from .BaseTrainer import _BaseTrainer

logger = logging.getLogger(__name__)


# TODO: Add functionality to ensure the underlying data is of type sentence_transformers.InputExample
@dataclass
class ClassifierDatasetCollection:
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: DataLoader | None
    num_classes: int

    def __post_init__(self):
        if not isinstance(self.train_dataloader, DataLoader):
            raise TypeError("train_dataloader must be a DataLoader")
        if not isinstance(self.val_dataloader, DataLoader):
            raise TypeError("val_dataloader must be a DataLoader")
        if self.test_dataloader is not None and not isinstance(
            self.test_dataloader, DataLoader
        ):
            raise TypeError("test_dataloader must be a DataLoader")
        if not isinstance(self.num_classes, int):
            raise TypeError("num_classes must be an integer")


class ClassifierTrainer(_BaseTrainer[ClassifierConfig | DiceLossClassifierConfig]):
    def __init__(
        self,
        config: ClassifierConfig | DiceLossClassifierConfig,
        classifier_dataset_collection: ClassifierDatasetCollection | None = None,
    ):
        super().__init__(config=config)
        self.__model = self._initialize_model()

        if classifier_dataset_collection is not None:
            self.__train_dataloader = classifier_dataset_collection.train_dataloader
            self.__val_dataloader = classifier_dataset_collection.val_dataloader
            self.__test_dataloader = classifier_dataset_collection.test_dataloader
            self.__num_labels = classifier_dataset_collection.num_classes
        elif (
            self._config.train_dataset_path is not None
            and self._config.val_dataset_path is not None
        ):
            (
                self.__train_dataloader,
                self.__val_dataloader,
                self.__test_dataloader,
                self.__num_labels,
            ) = self._initialize_data(
                train_file=self._config.train_dataset_path,
                val_file=self._config.val_dataset_path,
                test_file=self._config.test_dataset_path,
            )
        else:
            raise ValueError(
                "Either train_dataloader and val_dataloader must be provided, "
                "or train_dataset_path and val_dataset_path (in the config) must be provided"
            )

        self.__loss_fct = self._initialize_loss()
        self.__val_evaluator = self._initialize_val_evaluator()
        self.__test_evaluator = self._initialize_test_evaluator()
        self.__model_save_dir_name = self._initialize_output_path()

    ############################################################################
    # concrete properties
    ############################################################################

    @property
    def config(self) -> ClassifierConfig | DiceLossClassifierConfig:
        return self._config

    @property
    def model(self):
        return self.__model

    @property
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.__train_dataloader

    @property
    def loss_fct(self):
        return self.__loss_fct

    @property
    def val_evaluator(self):
        return self.__val_evaluator

    @property
    def test_evaluator(self):
        return self.__test_evaluator

    @property
    def model_save_dir_name(self):
        return self.__model_save_dir_name

    @property
    def steps_per_epoch(self):
        return len(self.__train_dataloader)

    @property
    def eval_metric(self) -> str:
        return self._config.eval_metric

    @property
    def val_eval_file_path(self):
        return os.path.join(
            self.model_save_dir_name, "eval", self.val_evaluator.csv_file
        )

    @property
    def test_eval_file_path(self):
        if self.test_evaluator is None:
            return None
        return os.path.join(
            self.model_save_dir_name, "eval", self.test_evaluator.csv_file
        )

    ############################################################################
    # concrete methods
    ############################################################################

    def _initialize_model(self):
        return SentenceTransformer(
            self._config.model_name,
            truncate_dim=self._config.classifier_hidden_dimension,
        )

    def _initialize_data(
        self,
        train_file: str,
        val_file: str,
        test_file: str | None = None,
    ):
        logging.info(f"Loading {train_file} dataset")
        train_df = pd.read_parquet(
            train_file,
            columns=[
                self._config.smiles_column_name,
                self._config.label_column_name,
            ],
        )
        train_df = train_df.astype({self._config.label_column_name: "int64"})
        if self._config.n_train_samples is not None:
            train_df = train_df.sample(
                n=self._config.n_train_samples,
                replace=False,
                random_state=self._config.seed,
                ignore_index=True,
            )

        train_dl = DataLoader(
            PandasDataFrameDataset(
                train_df,
                smiles_a_column=self._config.smiles_column_name,
                label_column=self._config.label_column_name,
                generate_dataset_examples_at_init=self._config.generate_dataset_examples_at_init,
            ),
            batch_size=self._config.train_batch_size,
            shuffle=True,
            pin_memory=True,
            pin_memory_device="cuda",
            num_workers=self._config.n_dataloader_workers,
        )

        logging.info(f"Loading {val_file} dataset")
        val_df = pd.read_parquet(
            val_file,
            columns=[
                self._config.smiles_column_name,
                self._config.label_column_name,
            ],
        )
        val_df = val_df.astype({self._config.label_column_name: "int64"})
        if self._config.n_val_samples is not None:
            val_df = val_df.sample(
                n=self._config.n_val_samples,
                replace=False,
                random_state=self._config.seed,
                ignore_index=True,
            )

        val_dl = DataLoader(
            PandasDataFrameDataset(
                val_df,
                smiles_a_column=self._config.smiles_column_name,
                label_column=self._config.label_column_name,
                generate_dataset_examples_at_init=self._config.generate_dataset_examples_at_init,
            ),
            batch_size=self._config.train_batch_size,
            shuffle=False,
            pin_memory=True,
            pin_memory_device="cuda",
            num_workers=self._config.n_dataloader_workers,
        )

        test_dl = None
        if test_file:
            logging.info(f"Loading {val_file} dataset")
            test_df = pd.read_parquet(
                test_file,
                columns=[
                    self._config.smiles_column_name,
                    self._config.label_column_name,
                ],
            )
            test_df = test_df.astype({self._config.label_column_name: "int64"})
            if self._config.n_test_samples is not None:
                test_df = test_df.sample(
                    n=self._config.n_test_samples,
                    replace=False,
                    random_state=self._config.seed,
                    ignore_index=True,
                )

            test_dl = DataLoader(
                PandasDataFrameDataset(
                    test_df,
                    smiles_a_column=self._config.smiles_column_name,
                    label_column=self._config.label_column_name,
                    generate_dataset_examples_at_init=self._config.generate_dataset_examples_at_init,
                ),
                batch_size=self._config.train_batch_size,
                shuffle=False,
                pin_memory=True,
                pin_memory_device="cuda",
                num_workers=self._config.n_dataloader_workers,
            )

        num_labels = train_df[self._config.label_column_name].nunique()

        return train_dl, val_dl, test_dl, num_labels

    def _initialize_val_evaluator(self):
        return LabelAccuracyEvaluator(
            dataloader=self.__val_dataloader,
            softmax_model=self.__loss_fct,
            write_csv=True,
            name="val",
        )

    def _initialize_test_evaluator(self):
        if self.__test_dataloader is None:
            return None
        return LabelAccuracyEvaluator(
            dataloader=self.__test_dataloader,
            softmax_model=self.__loss_fct,
            write_csv=True,
            name="test",
        )

    def _initialize_loss(self):
        from chem_mrl.losses import SelfAdjDiceLoss, SoftmaxLoss

        if self._config.loss_func == "softmax":
            return SoftmaxLoss(
                model=self.__model,
                smiles_embedding_dimension=self._config.classifier_hidden_dimension,
                num_labels=self.__num_labels,
                dropout=self._config.dropout_p,
                freeze_model=self._config.freeze_model,
            )

        assert isinstance(self._config, DiceLossClassifierConfig)
        return SelfAdjDiceLoss(
            model=self.__model,
            smiles_embedding_dimension=self._config.classifier_hidden_dimension,
            num_labels=self.__num_labels,
            dropout=self._config.dropout_p,
            freeze_model=self._config.freeze_model,
            reduction=self._config.dice_reduction,
            gamma=self._config.dice_gamma,
        )

    def _initialize_output_path(self):
        if isinstance(self._config, DiceLossClassifierConfig):
            dice_loss_suffix = (
                f"-{self._config.dice_reduction}-{self._config.dice_gamma}"
            )
        else:
            dice_loss_suffix = ""

        truncated_model_name = self._config.model_name
        try:
            truncated_model_name = self._config.model_name.rsplit("/", 1)[1]
        except IndexError:
            pass

        dir_name = (
            f"{truncated_model_name}"
            f"-{self._config.train_batch_size}"
            f"-{self._config.num_epochs}"
            f"-{self._config.lr_base:6f}"
            f"-{self._config.scheduler}-{self._config.warmup_steps_percent}"
            f"-{self._config.loss_func}-{self._config.dropout_p:3f}"
            f"-{self._config.classifier_hidden_dimension}{dice_loss_suffix}"
        )

        output_path = os.path.join(
            self._config.model_output_path,
            "classifier",
            dir_name,
        )
        logger.info(f"Output path: {output_path}")
        return output_path
