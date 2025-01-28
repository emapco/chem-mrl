import logging
import os
from typing import Literal

import optuna
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from torch import nn
from torch.utils.data import DataLoader

from chem_mrl.configs import Chem2dMRLConfig, ChemMRLConfig
from chem_mrl.constants import TRAIN_DS_DICT, VAL_DS_DICT
from chem_mrl.datasets import PandasDataFrameDataset
from chem_mrl.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

from .BaseTrainer import _BaseTrainer

logger = logging.getLogger(__name__)


class ChemMRLTrainer(_BaseTrainer):
    def __init__(
        self,
        config: ChemMRLConfig | Chem2dMRLConfig,
        optuna_trial: optuna.Trial | None = None,
    ):
        super().__init__(config=config, optuna_trial=optuna_trial)
        self._config = config

        self.__model = self._initialize_model()
        (self.__train_dataloader, self.__val_df) = self._load_data(
            train_file=TRAIN_DS_DICT[self._config.dataset_key],
            val_file=VAL_DS_DICT[self._config.dataset_key],
        )
        # Additional hard-coded eval properties which are used by the evaluator.
        # These attributes are used by the `eval_file_path` property
        self.__eval_name = "morgan-similarity"
        self.__eval_precision: Literal["int8"] = "int8"

        self.__val_evaluator = self._initialize_evaluator()
        self.__train_loss = self._initialize_loss()
        self.__model_save_path = self._initialize_output_path()

    def _initialize_model(self):
        word_embedding_model = models.Transformer(self._config.model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
        )
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def _load_data(
        self,
        num_of_rows_to_train_on: int | None = None,
        num_of_rows_to_validate_on: int | None = None,
        train_file: str = TRAIN_DS_DICT["fp-similarity"],
        val_file: str | None = VAL_DS_DICT["fp-similarity"],
    ):
        logging.info(f"Loading {train_file} dataset")
        train_df = pd.read_parquet(
            train_file,
            columns=[
                self._config.smiles_a_column_name,
                self._config.smiles_b_column_name,
                self._config.label_column_name,
            ],
        )
        train_df = train_df.astype({self._config.label_column_name: "float32"})
        if num_of_rows_to_train_on is not None:
            train_df = train_df.sample(
                n=num_of_rows_to_train_on,
                replace=False,
                random_state=self._config.seed,
                ignore_index=True,
            )

        train_dataloader = DataLoader(
            PandasDataFrameDataset(
                train_df,
                smiles_a_column=self._config.smiles_a_column_name,
                smiles_b_column=self._config.smiles_b_column_name,
                label_column=self._config.label_column_name,
            ),
            batch_size=self._config.train_batch_size,
            shuffle=True,
            pin_memory=True,
            pin_memory_device="cuda",
        )

        if val_file is None:
            logging.info("Splitting train_df into training and validation datasets.")
            val_df = train_df.sample(frac=0.15, random_state=self._config.seed)
            train_df.drop(val_df.index, inplace=True)
            val_df.reset_index(drop=True, inplace=True)
        else:
            logging.info(f"Loading {val_file} dataset")
            val_df = pd.read_parquet(
                val_file,
                columns=[
                    self._config.smiles_a_column_name,
                    self._config.smiles_b_column_name,
                    self._config.label_column_name,
                ],
            )
            if num_of_rows_to_validate_on is not None:
                val_df = val_df.sample(
                    n=num_of_rows_to_validate_on,
                    replace=False,
                    random_state=self._config.seed,
                    ignore_index=True,
                )

        # validation uses int8 tensors but keep it as a float for now
        val_df = val_df.astype({self._config.label_column_name: "float16"})
        return train_dataloader, val_df

    def _initialize_evaluator(self):
        return EmbeddingSimilarityEvaluator(
            self.__val_df[self._config.smiles_a_column_name],
            self.__val_df[self._config.smiles_b_column_name],
            self.__val_df[self._config.label_column_name],
            batch_size=self._config.train_batch_size,
            main_similarity=SimilarityFunction.TANIMOTO,
            name=self.__eval_name,
            show_progress_bar=True,
            write_csv=True,
            precision=self.__eval_precision,
        )

    def _initialize_loss(self):
        from sentence_transformers import losses

        if isinstance(self._config, Chem2dMRLConfig) and self._config.use_2d_matryoshka:
            return losses.Matryoshka2dLoss(
                self.__model,
                self._get_base_loss(self.__model, self._config),
                self._config.mrl_dimensions,
                matryoshka_weights=list(self._config.mrl_dimension_weights),
                n_layers_per_step=-1,
                n_dims_per_step=-1,
                last_layer_weight=self._config.last_layer_weight,
                prior_layers_weight=self._config.prior_layers_weight,
            )
        return losses.MatryoshkaLoss(
            self.__model,
            self._get_base_loss(self.__model, self._config),
            self._config.mrl_dimensions,
            matryoshka_weights=list(self._config.mrl_dimension_weights),
            n_dims_per_step=-1,
        )

    def _initialize_output_path(self):
        mrl_infix = ""
        layer_weight_infix = ""
        if isinstance(self._config, Chem2dMRLConfig) and self._config.use_2d_matryoshka:
            mrl_infix = "-2d"
            layer_weight_infix = f"{self._config.last_layer_weight:4f}-{self._config.prior_layers_weight:4f}"

        w1, w2, w3, w4, w5, w6 = self._config.mrl_dimension_weights

        output_path = os.path.join(
            self._config.model_output_path,
            f"chem-{mrl_infix}mrl",
            f"{self._config.dataset_key}-{self._config.train_batch_size}-{self._config.num_epochs}"
            f"-{self._config.lr_base:6f}-{self._config.scheduler}-{self._config.warmup_steps_percent}"
            f"-{self._config.loss_func}-{self._config.tanimoto_similarity_loss_func}-{layer_weight_infix}"
            f"-{w1:4f}-{w2:4f}-{w3:4f}"
            f"-{w4:4f}-{w5:4f}-{w6:4f}",
        )
        logger.info(f"Output path: {output_path}")
        return output_path

    def fit(self):
        self._fit(
            config=self._config,
            model=self.__model,
            train_dataloader=self.__train_dataloader,
            train_loss=self.__train_loss,
            val_evaluator=self.__val_evaluator,
            model_save_path=self.__model_save_path,
            wandb_callback=self._wandb_callback,
        )

    @property
    def config(self) -> ChemMRLConfig | Chem2dMRLConfig:
        return self._config

    @property
    def model(self):
        return self.__model

    @property
    def loss_fct(self):
        return self.__train_loss

    @property
    def model_save_path(self):
        return self.__model_save_path

    @property
    def eval_file_path(self):
        """The evaluation file path is dependent on the evaluator arguments.
        This property returns the formatted path."""
        if self.__eval_precision == "float32":
            fixed_precision_infix = ""
        else:
            fixed_precision_infix = f"_{self.__eval_precision}"

        return os.path.join(
            self.model_save_path,
            "eval",
            f"similarity_evaluation_{self.__eval_name}{fixed_precision_infix}_results.csv",
        )

    @staticmethod
    def _get_base_loss(
        model: SentenceTransformer,
        config: ChemMRLConfig | Chem2dMRLConfig,
    ) -> nn.Module:
        from sentence_transformers import losses

        from chem_mrl.losses import TanimotoSentLoss, TanimotoSimilarityLoss

        LOSS_FUNCTIONS = {
            "tanimotosentloss": lambda model: TanimotoSentLoss(model),
            "cosentloss": lambda model: losses.CoSENTLoss(model),
            "tanimotosimilarityloss": {
                "mse": lambda model: TanimotoSimilarityLoss(model, loss=nn.MSELoss()),
                "l1": lambda model: TanimotoSimilarityLoss(model, loss=nn.L1Loss()),
                "smooth_l1": lambda model: TanimotoSimilarityLoss(
                    model, loss=nn.SmoothL1Loss()
                ),
                "huber": lambda model: TanimotoSimilarityLoss(
                    model, loss=nn.HuberLoss()
                ),
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
        if config.loss_func in ["tanimotosentloss", "cosentloss"]:
            return LOSS_FUNCTIONS[config.loss_func](model)

        return LOSS_FUNCTIONS["tanimotosimilarityloss"][
            config.tanimoto_similarity_loss_func
        ](model)
