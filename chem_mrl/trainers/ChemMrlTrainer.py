import logging

from datasets import Dataset
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import SentenceEvaluator
from torch import nn

from chem_mrl.evaluation import EmbeddingSimilarityEvaluator
from chem_mrl.models import LatentAttentionLayer
from chem_mrl.schemas import BaseConfig, ChemMRLConfig

from .BaseTrainer import _BaseTrainer

logger = logging.getLogger(__name__)


class ChemMRLTrainer(_BaseTrainer):
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)

        self._model_config: ChemMRLConfig = config.model
        if not isinstance(self._model_config, ChemMRLConfig):
            raise TypeError("config.model must be a ChemMRLConfig instance")

        self.__model: SentenceTransformer = self._init_model()
        self.__model.tokenizer = self._initialize_tokenizer()  # type: ignore
        self.__train_ds, self.__val_ds, self.__test_ds = self._init_data()
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
    def train_dataset(self) -> dict[str, Dataset]:
        return self.__train_ds

    @property
    def eval_dataset(self) -> dict[str, Dataset]:
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
    def eval_metric(self) -> str:
        return self._model_config.eval_metric.value

    ############################################################################
    # concrete methods
    ############################################################################

    def _init_model(self) -> SentenceTransformer:
        base_model = models.Transformer(self._model_config.model_name)
        pooling_model = models.Pooling(
            base_model.get_word_embedding_dimension(),
            pooling_mode=self._model_config.embedding_pooling,
        )
        normalization_model = models.Normalize()

        if (
            self._model_config.latent_attention_config is not None
            and self._model_config.latent_attention_config.enable
        ):
            latent_attention_model = LatentAttentionLayer(
                self._model_config.latent_attention_config
            )
            modules = [base_model, latent_attention_model, pooling_model, normalization_model]
        else:
            modules = [base_model, pooling_model, normalization_model]

        similarity_fn_name = "cosine"
        if self._model_config.loss_func in ["tanimotosentloss", "tanimotosimilarityloss"]:
            similarity_fn_name = "tanimoto"

        model = SentenceTransformer(modules=modules, similarity_fn_name=similarity_fn_name)
        logger.info(model)
        return model

    def _initialize_tokenizer(
        self,
    ):
        if not self._model_config.use_query_tokenizer:
            return self.__model.tokenizer

        from chem_mrl.tokenizers import QuerySmilesTokenizerFast

        return QuerySmilesTokenizerFast(max_len=self.__model.tokenizer.model_max_length)  # type: ignore

    def _init_val_evaluator(self):
        """
        Initialize validation evaluators for all datasets.

        Returns:
            Dictionary mapping dataset names to evaluators
        """
        evaluators: dict[str, SentenceEvaluator] = {}
        for dataset_name, val_ds in self.__val_ds.items():
            evaluators[dataset_name] = EmbeddingSimilarityEvaluator(
                val_ds["smiles_a"],
                val_ds["smiles_b"],
                val_ds["label"],
                batch_size=self._training_args.per_device_eval_batch_size,
                main_similarity=self._model_config.eval_similarity_fct,
                metric=self._model_config.eval_metric,
                name=dataset_name,
                show_progress_bar=not self._training_args.disable_tqdm,
                write_csv=True,
            )

        logger.info(f"Initialized {len(evaluators)} validation evaluators")
        return evaluators

    def _init_test_evaluator(self):
        """
        Initialize test evaluators for all datasets.

        Returns:
            Dictionary mapping dataset names to test evaluators, or None if no test datasets
        """
        evaluators: dict[str, SentenceEvaluator] = {}
        for dataset_name, test_ds in self.__test_ds.items():
            evaluators[dataset_name] = EmbeddingSimilarityEvaluator(
                test_ds["smiles_a"],
                test_ds["smiles_b"],
                test_ds["label"],
                batch_size=self._training_args.per_device_eval_batch_size,
                main_similarity=self._model_config.eval_similarity_fct,
                metric=self._model_config.eval_metric,
                name=dataset_name,
                show_progress_bar=not self._training_args.disable_tqdm,
                write_csv=True,
                precision="int8",
            )

        logger.info(f"Initialized {len(evaluators)} test evaluators")
        return evaluators

    def _init_loss(self):
        from sentence_transformers.losses import Matryoshka2dLoss, MatryoshkaLoss

        if self._model_config.use_2d_matryoshka:
            return Matryoshka2dLoss(
                self.__model,
                self._get_base_loss(self.__model, self._model_config),
                list(self._model_config.mrl_dimensions),
                matryoshka_weights=list(self._model_config.mrl_dimension_weights),
                n_layers_per_step=self._model_config.n_layers_per_step,
                n_dims_per_step=self._model_config.n_dims_per_step,
                last_layer_weight=self._model_config.last_layer_weight,
                prior_layers_weight=self._model_config.prior_layers_weight,
                kl_div_weight=self._model_config.kl_div_weight,
                kl_temperature=self._model_config.kl_temperature,
            )
        return MatryoshkaLoss(
            self.__model,
            self._get_base_loss(self.__model, self._model_config),
            list(self._model_config.mrl_dimensions),
            matryoshka_weights=list(self._model_config.mrl_dimension_weights),
            n_dims_per_step=self._model_config.n_dims_per_step,
        )

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
