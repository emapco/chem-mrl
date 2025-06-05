import logging

from datasets import Dataset, Value
from sentence_transformers import SentenceTransformer, models
from torch import nn

from chem_mrl.evaluation import EmbeddingSimilarityEvaluator
from chem_mrl.load import load_dataset_with_fallback
from chem_mrl.models import LatentAttentionLayer
from chem_mrl.schemas import BaseConfig, ChemMRLConfig

from .BaseTrainer import _BaseTrainer

logger = logging.getLogger(__name__)


class ChemMRLTrainer(_BaseTrainer):
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)
        if not isinstance(config.model, ChemMRLConfig):
            raise TypeError("config.model must be a ChemMRLConfig instance")
        if self._config.train_dataset_path is None or self._config.val_dataset_path is None:
            raise ValueError(
                "Either train_dataloader and val_dataloader must be provided, "
                "or train_dataset_path and val_dataset_path must be provided"
            )

        self.__model: SentenceTransformer = self._init_model()
        self.__model.tokenizer = self._initialize_tokenizer()  # type: ignore
        self.__train_ds, self.__val_ds, self.__test_ds = self._init_data(
            train_file=self._config.train_dataset_path,
            val_file=self._config.val_dataset_path,
            test_file=self._config.test_dataset_path,
        )
        self.__loss_function = self._init_loss()
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

    def _init_model(self) -> SentenceTransformer:
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

        similarity_fn_name = "cosine"
        if self._config.model.loss_func in ["tanimotosentloss", "tanimotosimilarityloss"]:
            similarity_fn_name = "tanimoto"

        model = SentenceTransformer(modules=modules, similarity_fn_name=similarity_fn_name)
        logger.info(model)
        return model

    def _initialize_tokenizer(
        self,
    ):
        assert isinstance(self._config.model, ChemMRLConfig)
        if not self._config.model.use_query_tokenizer:
            return self.__model.tokenizer

        from chem_mrl.tokenizers import QuerySmilesTokenizerFast

        return QuerySmilesTokenizerFast(max_len=self.__model.tokenizer.model_max_length)  # type: ignore

    def _init_data(
        self,
        train_file: str,
        val_file: str,
        test_file: str | None = None,
    ):
        assert isinstance(self._config.model, ChemMRLConfig)
        assert (
            self._config.smiles_a_column_name is not None
            and self._config.smiles_a_column_name != ""
        ), "smiles_a_column_name must be specified when training a ChemMRL model"
        assert (
            self._config.smiles_b_column_name is not None
            and self._config.smiles_b_column_name != ""
        ), "smiles_b_column_name must be specified when training a ChemMRL model"

        data_columns = [
            self._config.smiles_a_column_name,
            self._config.smiles_b_column_name,
            self._config.label_column_name,
        ]

        def raw_to_expected_example(batch):
            return {
                self.A_COL: batch[self._config.smiles_a_column_name],
                self.B_COL: batch[self._config.smiles_b_column_name],
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
            cast="float32",
            sample_size=self._config.n_train_samples,
        )
        eval_ds = process_ds(
            load_dataset_with_fallback(val_file, self._config.val_datasets_split, data_columns),
            cast="float16",
            sample_size=self._config.n_train_samples,
        )
        test_ds = None
        if test_file is not None:
            test_ds = process_ds(
                load_dataset_with_fallback(
                    test_file, self._config.test_datasets_split, data_columns
                ),
                cast="float16",
                sample_size=self._config.n_train_samples,
            )
        return train_ds, eval_ds, test_ds

    def _init_val_evaluator(self):
        assert isinstance(self._config.model, ChemMRLConfig)
        assert (
            self._config.smiles_b_column_name is not None
            and self._config.smiles_b_column_name != ""
        ), "smiles_b_column_name must be specified when training a ChemMRL model"
        return EmbeddingSimilarityEvaluator(
            self.__val_ds[self.A_COL],
            self.__val_ds[self.B_COL],
            self.__val_ds[self.LABEL_COL],
            batch_size=self._training_args.per_device_eval_batch_size,
            main_similarity=self._config.model.eval_similarity_fct,
            metric=self._config.model.eval_metric,
            name="val",
            show_progress_bar=not self._training_args.disable_tqdm,
            write_csv=True,
            precision="int8",
        )

    def _init_test_evaluator(self):
        if self.__test_ds is None:
            return None
        assert isinstance(self._config.model, ChemMRLConfig)
        assert (
            self._config.smiles_b_column_name is not None
            and self._config.smiles_b_column_name != ""
        ), "smiles_b_column_name must be specified when training a ChemMRL model"
        return EmbeddingSimilarityEvaluator(
            self.__val_ds[self.A_COL],
            self.__val_ds[self.B_COL],
            self.__val_ds[self.LABEL_COL],
            batch_size=self._training_args.per_device_eval_batch_size,
            main_similarity=self._config.model.eval_similarity_fct,
            metric=self._config.model.eval_metric,
            name="test",
            show_progress_bar=not self._training_args.disable_tqdm,
            write_csv=True,
            precision="int8",
        )

    def _init_loss(self):
        from sentence_transformers.losses import Matryoshka2dLoss, MatryoshkaLoss

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
