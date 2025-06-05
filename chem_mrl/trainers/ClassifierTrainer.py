import logging

import torch
from datasets import Dataset, Value
from sentence_transformers import SentenceTransformer

from chem_mrl.evaluation import LabelAccuracyEvaluator
from chem_mrl.load import load_dataset_with_fallback
from chem_mrl.schemas import BaseConfig, ClassifierConfig

from .BaseTrainer import _BaseTrainer

logger = logging.getLogger(__name__)


class ClassifierTrainer(_BaseTrainer):
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)
        if not isinstance(config.model, ClassifierConfig):
            raise TypeError("config.model must be a ClassifierConfig instance")
        if self._config.train_dataset_path is None or self._config.val_dataset_path is None:
            raise ValueError(
                "Either train_dataloader and val_dataloader must be provided, "
                "or train_dataset_path and val_dataset_path must be provided"
            )

        self.__model = self._init_model()
        (
            self.__train_ds,
            self.__val_ds,
            self.__test_ds,
        ) = self._init_data(
            train_file=self._config.train_dataset_path,
            val_file=self._config.val_dataset_path,
            test_file=self._config.test_dataset_path,
        )
        self.__loss_function: torch.nn.Module = self._init_loss()
        self.__val_evaluator = self._init_val_evaluator()
        self.__test_evaluator = self._init_test_evaluator()

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
    def train_dataset(self):
        return self.__train_ds

    @property
    def eval_dataset(self):
        return self.__val_ds

    @property
    def loss_function(self):
        return self.__loss_function

    @property
    def val_evaluator(self):
        return self.__val_evaluator

    @property
    def test_evaluator(self):
        return self.__test_evaluator

    @property
    def steps_per_epoch(self):
        return len(self.__train_ds)

    @property
    def eval_metric(self) -> str:
        return self._config.model.eval_metric

    @property
    def val_eval_file_path(self):
        return self.val_evaluator.csv_file

    ############################################################################
    # concrete methods
    ############################################################################

    def _init_model(self):
        assert isinstance(self._config.model, ClassifierConfig)
        model = SentenceTransformer(
            self._config.model.model_name,
            truncate_dim=self._config.model.classifier_hidden_dimension,
        )
        logger.info(model)
        return model

    def _init_data(
        self,
        train_file: str,
        val_file: str,
        test_file: str | None = None,
    ):
        assert isinstance(self._config.model, ClassifierConfig)
        assert (
            self._config.smiles_a_column_name is not None
            and self._config.smiles_a_column_name != ""
        ), "smiles_a_column_name must be specified when training a Classifier model"

        data_columns = [
            self._config.smiles_a_column_name,
            self._config.label_column_name,
        ]

        def raw_to_expected_example(batch):
            return {
                self.A_COL: batch[self._config.smiles_a_column_name],
                self.LABEL_COL: batch[self._config.label_column_name],
            }

        def process_ds(
            ds: Dataset,
            cast: str | None = None,
            sample_size: int | None = None,
        ):
            if sample_size is not None:
                ds = ds.shuffle(seed=self._training_args.data_seed).select(range(sample_size))
            if cast is not None:
                ds = ds.cast_column(self._config.label_column_name, Value(cast))
            ds = ds.map(raw_to_expected_example, batched=True, remove_columns=ds.column_names)
            return ds

        train_ds = process_ds(
            load_dataset_with_fallback(train_file, self._config.train_datasets_split, data_columns),
            cast="int64",
            sample_size=self._config.n_train_samples,
        )
        eval_ds = process_ds(
            load_dataset_with_fallback(val_file, self._config.val_datasets_split, data_columns),
            cast="int64",
            sample_size=self._config.n_val_samples,
        )
        test_ds = None
        if test_file is not None:
            test_ds = process_ds(
                load_dataset_with_fallback(
                    test_file, self._config.test_datasets_split, data_columns
                ),
                cast="int64",
                sample_size=self._config.n_test_samples,
            )

        return train_ds, eval_ds, test_ds

    def _init_val_evaluator(self):
        return LabelAccuracyEvaluator(
            dataset=self.__val_ds,
            softmax_model=self.__loss_function,
            write_csv=True,
            name="val",
            batch_size=self._config.training_args.per_device_eval_batch_size,
            smiles_column_name=self.A_COL,
            label_column_name=self.LABEL_COL,
        )

    def _init_test_evaluator(self):
        if self.__test_ds is None:
            return None
        return LabelAccuracyEvaluator(
            dataset=self.__test_ds,
            softmax_model=self.__loss_function,
            write_csv=True,
            name="test",
            batch_size=self._config.training_args.per_device_eval_batch_size,
            smiles_column_name=self.A_COL,
            label_column_name=self.LABEL_COL,
        )

    def _init_loss(self):
        from chem_mrl.losses import SelfAdjDiceLoss, SoftmaxLoss

        assert isinstance(self._config.model, ClassifierConfig)
        if self._config.model.loss_func == "softmax":
            return SoftmaxLoss(
                model=self.__model,
                smiles_embedding_dimension=self._config.model.classifier_hidden_dimension,
                num_labels=self.config.model.num_labels,
                dropout=self._config.model.dropout_p,
                freeze_model=self._config.model.freeze_model,
            )

        return SelfAdjDiceLoss(
            model=self.__model,
            smiles_embedding_dimension=self._config.model.classifier_hidden_dimension,
            num_labels=self.config.model.num_labels,
            dropout=self._config.model.dropout_p,
            freeze_model=self._config.model.freeze_model,
            reduction=self._config.model.dice_reduction,
            gamma=self._config.model.dice_gamma,
        )
