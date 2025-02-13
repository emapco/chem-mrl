import logging
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import get_device_name
from torch.utils.data import DataLoader

from chem_mrl.datasets import PandasDataFrameDataset
from chem_mrl.evaluation import LabelAccuracyEvaluator
from chem_mrl.schemas import BaseConfig, ClassifierConfig

from .BaseTrainer import _BaseTrainer

logger = logging.getLogger(__name__)


# TODO: Add checks to ensure the underlying data is of type sentence_transformers.InputExample
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
        if self.test_dataloader is not None and not isinstance(self.test_dataloader, DataLoader):
            raise TypeError("test_dataloader must be a DataLoader")
        if not isinstance(self.num_classes, int):
            raise TypeError("num_classes must be an integer")


class ClassifierTrainer(_BaseTrainer):
    def __init__(
        self,
        config: BaseConfig,
        classifier_dataset_collection: ClassifierDatasetCollection | None = None,
    ):
        super().__init__(config=config)
        if not isinstance(config.model, ClassifierConfig):
            raise TypeError("config.model must be a ClassifierConfig instance")
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
        self.__model_save_dir = self._initialize_output_path()

    ############################################################################
    # concrete properties
    ############################################################################

    @property
    def config(self):
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
    def model_save_dir(self):
        return self.__model_save_dir

    @model_save_dir.setter
    def model_save_dir(self, value: str):
        self.__model_save_dir = value

    @property
    def steps_per_epoch(self):
        return len(self.__train_dataloader)

    @property
    def eval_metric(self) -> str:
        return self._config.model.eval_metric

    @property
    def val_eval_file_path(self):
        return os.path.join(self.model_save_dir, "eval", self.val_evaluator.csv_file)

    @property
    def test_eval_file_path(self):
        if self.test_evaluator is None:
            return None
        return os.path.join(self.model_save_dir, "eval", self.test_evaluator.csv_file)

    ############################################################################
    # concrete methods
    ############################################################################

    def _initialize_model(self):
        assert isinstance(self._config.model, ClassifierConfig)
        model = SentenceTransformer(
            self._config.model.model_name,
            truncate_dim=self._config.model.classifier_hidden_dimension,
        )
        logger.info(model)
        return model

    def _initialize_data(
        self,
        train_file: str,
        val_file: str,
        test_file: str | None = None,
    ):
        logging.info(f"Loading {train_file} dataset")
        assert isinstance(self._config.model, ClassifierConfig)
        pin_device = get_device_name()

        train_df = pd.read_parquet(
            train_file,
            columns=[
                self._config.smiles_a_column_name,
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
                smiles_a_column=self._config.smiles_a_column_name,
                label_column=self._config.label_column_name,
                generate_dataset_examples_at_init=self._config.generate_dataset_examples_at_init,
            ),
            batch_size=self._config.train_batch_size,
            shuffle=True,
            pin_memory=self._config.pin_memory,
            pin_memory_device=pin_device if self._config.pin_memory else "",
            num_workers=self._config.n_dataloader_workers,
            persistent_workers=bool(self._config.n_dataloader_workers),
            multiprocessing_context=self._config.multiprocess_context,
        )

        logging.info(f"Loading {val_file} dataset")
        val_df = pd.read_parquet(
            val_file,
            columns=[
                self._config.smiles_a_column_name,
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
                smiles_a_column=self._config.smiles_a_column_name,
                label_column=self._config.label_column_name,
                generate_dataset_examples_at_init=self._config.generate_dataset_examples_at_init,
            ),
            batch_size=self._config.eval_batch_size,
            shuffle=False,
            pin_memory=self._config.pin_memory,
            pin_memory_device=pin_device if self._config.pin_memory else "",
            num_workers=self._config.n_dataloader_workers,
            persistent_workers=bool(self._config.n_dataloader_workers),
            multiprocessing_context=self._config.multiprocess_context,
        )

        test_dl = None
        if test_file:
            logging.info(f"Loading {val_file} dataset")
            test_df = pd.read_parquet(
                test_file,
                columns=[
                    self._config.smiles_a_column_name,
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
                    smiles_a_column=self._config.smiles_a_column_name,
                    label_column=self._config.label_column_name,
                    generate_dataset_examples_at_init=self._config.generate_dataset_examples_at_init,
                ),
                batch_size=self._config.eval_batch_size,
                shuffle=False,
                pin_memory=self._config.pin_memory,
                pin_memory_device=pin_device if self._config.pin_memory else "",
                num_workers=self._config.n_dataloader_workers,
                persistent_workers=bool(self._config.n_dataloader_workers),
                multiprocessing_context=self._config.multiprocess_context,
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

        assert isinstance(self._config.model, ClassifierConfig)
        if self._config.model.loss_func == "softmax":
            return SoftmaxLoss(
                model=self.__model,
                smiles_embedding_dimension=self._config.model.classifier_hidden_dimension,
                num_labels=self.__num_labels,
                dropout=self._config.model.dropout_p,
                freeze_model=self._config.model.freeze_model,
            )

        return SelfAdjDiceLoss(
            model=self.__model,
            smiles_embedding_dimension=self._config.model.classifier_hidden_dimension,
            num_labels=self.__num_labels,
            dropout=self._config.model.dropout_p,
            freeze_model=self._config.model.freeze_model,
            reduction=self._config.model.dice_reduction,
            gamma=self._config.model.dice_gamma,
        )

    def _initialize_output_path(self):
        assert isinstance(self._config.model, ClassifierConfig)

        output_path = os.path.join(
            self._config.model_output_path,
            f"classifier-{self._config.model.model_name.replace('/', '-')}"
            f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        logger.info(f"Output path: {output_path}")
        return output_path
