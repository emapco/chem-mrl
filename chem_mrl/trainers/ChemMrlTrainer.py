import logging
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import get_device_name
from torch import nn
from torch.utils.data import DataLoader

from chem_mrl.datasets import PandasDataFrameDataset
from chem_mrl.evaluation import EmbeddingSimilarityEvaluator
from chem_mrl.models import LatentAttentionLayer
from chem_mrl.schemas import BaseConfig, ChemMRLConfig

from .BaseTrainer import _BaseTrainer

logger = logging.getLogger(__name__)


# TODO: Add checks to ensure the underlying data is of type sentence_transformers.InputExample
@dataclass
class ChemMrlDatasetCollection:
    train_dataloader: DataLoader
    val_dataframe: pd.DataFrame
    test_dataframe: pd.DataFrame | None

    def __post_init__(self):
        if not isinstance(self.train_dataloader, DataLoader):
            raise TypeError("train_dataloader must be a DataLoader")
        if not isinstance(self.val_dataframe, pd.DataFrame):
            raise TypeError("val_dataloader must be a DataFrame")
        if self.test_dataframe is not None and not isinstance(self.test_dataframe, pd.DataFrame):
            raise TypeError("test_dataloader must be a DataFrame")


class ChemMRLTrainer(_BaseTrainer):
    def __init__(
        self,
        config: BaseConfig,
        chem_mrl_dataset_collection: ChemMrlDatasetCollection | None = None,
    ):
        super().__init__(config=config)
        if not isinstance(config.model, ChemMRLConfig):
            raise TypeError("config.model must be a ChemMRLConfig instance")
        self.__model = self._initialize_model()
        self.__model.tokenizer = self._initialize_tokenizer()
        if chem_mrl_dataset_collection is not None:
            self.__train_dataloader = chem_mrl_dataset_collection.train_dataloader
            self.__val_df = chem_mrl_dataset_collection.val_dataframe
            self.__test_df = chem_mrl_dataset_collection.test_dataframe
        elif (
            self._config.train_dataset_path is not None
            and self._config.val_dataset_path is not None
        ):
            (self.__train_dataloader, self.__val_df, self.__test_df) = self._initialize_data(
                train_file=self._config.train_dataset_path,
                val_file=self._config.val_dataset_path,
                test_file=self._config.test_dataset_path,
            )
        else:
            raise ValueError(
                "Either train_dataloader and val_dataloader must be provided, "
                "or train_dataset_path and val_dataset_path must be provided"
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
    def train_dataloader(self):
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

    def _initialize_model(self) -> SentenceTransformer:
        assert isinstance(self._config.model, ChemMRLConfig)

        base_model = models.Transformer(self._config.model.model_name)
        pooling_model = models.Pooling(
            base_model.get_word_embedding_dimension(),
            pooling_mode=self._config.model.embedding_pooling,
        )
        normalization_model = models.Normalize()

        if (
            self._config.model.latent_attention_config is not None
            and self._config.model.latent_attention_config.enable
        ):
            latent_attention_model = LatentAttentionLayer(
                self._config.model.latent_attention_config
            )
            modules = [base_model, latent_attention_model, pooling_model, normalization_model]
        else:
            modules = [base_model, pooling_model, normalization_model]

        model = SentenceTransformer(modules=modules)
        logger.info(model)
        return model

    def _initialize_tokenizer(
        self,
    ):
        assert isinstance(self._config.model, ChemMRLConfig)
        if not self._config.model.use_query_tokenizer:
            return self.__model.tokenizer

        from chem_mrl.tokenizers import QuerySmilesTokenizerFast

        return QuerySmilesTokenizerFast(max_len=self.__model.tokenizer.model_max_length)

    def _initialize_data(
        self,
        train_file: str,
        val_file: str,
        test_file: str | None = None,
    ):
        assert isinstance(self._config.model, ChemMRLConfig)
        assert (
            self._config.smiles_b_column_name is not None
            and self._config.smiles_b_column_name != ""
        ), "smiles_b_column_name must be specified when training a ChemMRL model"
        logging.info(f"Loading {train_file} dataset")
        pin_device = get_device_name()

        train_df = pd.read_parquet(
            train_file,
            columns=[
                self._config.smiles_a_column_name,
                self._config.smiles_b_column_name,
                self._config.label_column_name,
            ],
        )
        train_df = train_df.astype({self._config.label_column_name: "float32"})
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
                smiles_b_column=self._config.smiles_b_column_name,
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
                self._config.smiles_b_column_name,
                self._config.label_column_name,
            ],
        )
        # validation uses int8 tensors but keep it as a float for now
        val_df = val_df.astype({self._config.label_column_name: "float16"})
        if self._config.n_val_samples is not None:
            val_df = val_df.sample(
                n=self._config.n_val_samples,
                replace=False,
                random_state=self._config.seed,
                ignore_index=True,
            )

        test_df = None
        if test_file is not None:
            logging.info(f"Loading {test_file} dataset")
            test_df = pd.read_parquet(
                test_file,
                columns=[
                    self._config.smiles_a_column_name,
                    self._config.smiles_b_column_name,
                    self._config.label_column_name,
                ],
            )
            test_df = test_df.astype({self._config.label_column_name: "float32"})
            if self._config.n_test_samples is not None:
                test_df = test_df.sample(
                    n=self._config.n_test_samples,
                    replace=False,
                    random_state=self._config.seed,
                    ignore_index=True,
                )

        return train_dl, val_df, test_df

    def _initialize_val_evaluator(self):
        assert isinstance(self._config.model, ChemMRLConfig)
        return EmbeddingSimilarityEvaluator(
            self.__val_df[self._config.smiles_a_column_name],
            self.__val_df[self._config.smiles_b_column_name],
            self.__val_df[self._config.label_column_name],
            batch_size=self._config.eval_batch_size,
            main_similarity=self._config.model.eval_similarity_fct,
            metric=self._config.model.eval_metric,
            name="val",
            show_progress_bar=self._config.show_progress_bar,
            write_csv=True,
            precision="int8",
        )

    def _initialize_test_evaluator(self):
        if self.__test_df is None:
            return None
        assert isinstance(self._config.model, ChemMRLConfig)
        return EmbeddingSimilarityEvaluator(
            self.__test_df[self._config.smiles_a_column_name],
            self.__test_df[self._config.smiles_b_column_name],
            self.__test_df[self._config.label_column_name],
            batch_size=self._config.eval_batch_size,
            main_similarity=self._config.model.eval_similarity_fct,
            metric=self._config.model.eval_metric,
            name="test",
            show_progress_bar=self._config.show_progress_bar,
            write_csv=True,
            precision="int8",
        )

    def _initialize_loss(self):
        from chem_mrl.losses import Matryoshka2dLoss, MatryoshkaLoss

        assert isinstance(self._config.model, ChemMRLConfig)
        if self._config.model.use_2d_matryoshka:
            return Matryoshka2dLoss(
                self.__model,
                self._get_base_loss(self.__model, self._config.model),
                list(self._config.model.mrl_dimensions),
                matryoshka_weights=list(self._config.model.mrl_dimension_weights),
                n_layers_per_step=self._config.model.n_layers_per_step,
                n_dims_per_step=self._config.model.n_dims_per_step,
                last_layer_weight=self._config.model.last_layer_weight,
                prior_layers_weight=self._config.model.prior_layers_weight,
                kl_div_weight=self._config.model.kl_div_weight,
                kl_temperature=self._config.model.kl_temperature,
            )
        return MatryoshkaLoss(
            self.__model,
            self._get_base_loss(self.__model, self._config.model),
            list(self._config.model.mrl_dimensions),
            matryoshka_weights=list(self._config.model.mrl_dimension_weights),
            n_dims_per_step=self._config.model.n_dims_per_step,
        )

    def _initialize_output_path(self):
        assert isinstance(self._config.model, ChemMRLConfig)

        mrl_infix = ""
        if self._config.model.use_2d_matryoshka:
            mrl_infix = "2d"

        output_path = os.path.join(
            self._config.model_output_path,
            f"chem-{mrl_infix}mrl-{self._config.model.model_name.replace('/', '-')}"
            f"-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )

        logger.info(f"Output path: {output_path}")
        return output_path

    ############################################################################
    # private methods
    ############################################################################

    @staticmethod
    def _get_base_loss(
        model: SentenceTransformer,
        config: ChemMRLConfig,
    ) -> nn.Module:
        from sentence_transformers import losses

        from chem_mrl.losses import TanimotoSentLoss, TanimotoSimilarityLoss

        LOSS_FUNCTIONS = {
            "tanimotosentloss": lambda model: TanimotoSentLoss(model),
            "cosentloss": lambda model: losses.CoSENTLoss(model),
            "angleloss": lambda model: losses.AnglELoss(model),
            "tanimotosimilarityloss": {
                "mse": lambda model: TanimotoSimilarityLoss(model, loss=nn.MSELoss()),
                "l1": lambda model: TanimotoSimilarityLoss(model, loss=nn.L1Loss()),
                "smooth_l1": lambda model: TanimotoSimilarityLoss(model, loss=nn.SmoothL1Loss()),
                "huber": lambda model: TanimotoSimilarityLoss(model, loss=nn.HuberLoss()),
                "bin_cross_entropy": lambda model: TanimotoSimilarityLoss(
                    model, loss=nn.BCEWithLogitsLoss()
                ),
                "kldiv": lambda model: TanimotoSimilarityLoss(
                    model, loss=nn.KLDivLoss(reduction="batchmean")
                ),
                "cosine_embedding_loss": lambda model: TanimotoSimilarityLoss(
                    model, loss=nn.CosineEmbeddingLoss()
                ),
            },
        }
        if config.loss_func.value in ["tanimotosentloss", "cosentloss", "angleloss"]:
            return LOSS_FUNCTIONS[config.loss_func.value](model)

        if config.tanimoto_similarity_loss_func is None:
            raise ValueError(
                "tanimoto_similarity_loss_func must be provided "
                "when loss_func='tanimotosimilarityloss'"
            )
        return LOSS_FUNCTIONS["tanimotosimilarityloss"][config.tanimoto_similarity_loss_func.value](
            model
        )
