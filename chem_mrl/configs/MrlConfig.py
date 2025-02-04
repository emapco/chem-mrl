from dataclasses import dataclass

from chem_mrl.constants import CHEM_MRL_DIMENSIONS

from .BaseConfig import _BaseConfig
from .types import (
    CHEM_MRL_EMBEDDING_POOLING_OPTIONS,
    CHEM_MRL_EVAL_METRIC_OPTIONS,
    CHEM_MRL_LOSS_FCT_OPTIONS,
    EVAL_SIMILARITY_FCT_OPTIONS,
    TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS,
    ChemMrlEvalMetricOptionType,
    ChemMrlLossFctOptionType,
    ChemMrlPoolingOptionType,
    EvalSimilarityMetricOptionType,
    TanimotoSimilarityBaseLossFctOptionType,
)


@dataclass(frozen=True)
class ChemMRLConfig(_BaseConfig):
    smiles_a_column_name: str = "smiles_a"
    smiles_b_column_name: str = "smiles_b"
    label_column_name: str = "similarity"
    embedding_pooling: ChemMrlPoolingOptionType = "mean"  # type: ignore
    loss_func: ChemMrlLossFctOptionType = "tanimotosentloss"  # type: ignore
    tanimoto_similarity_loss_func: TanimotoSimilarityBaseLossFctOptionType | None = None  # type: ignore
    eval_similarity_fct: EvalSimilarityMetricOptionType = "tanimoto"  # type: ignore
    eval_metric: ChemMrlEvalMetricOptionType = "spearman"  # type: ignore
    mrl_dimensions: tuple = tuple(CHEM_MRL_DIMENSIONS)
    mrl_dimension_weights: tuple = (1, 1, 1, 1, 1, 1, 1, 1)
    n_dims_per_step: int = -1
    use_2d_matryoshka: bool = False

    def __post_init__(self):
        super().__post_init__()
        # check types
        if not isinstance(self.smiles_a_column_name, str):
            raise TypeError("smiles_a_column_name must be a string")
        if not isinstance(self.smiles_b_column_name, str):
            raise TypeError("smiles_b_column_name must be a string")
        if not isinstance(self.label_column_name, str):
            raise TypeError("label_column_name must be a string")
        if not isinstance(self.embedding_pooling, str):
            raise TypeError("embedding_pooling must be a string")
        if not isinstance(self.loss_func, str):
            raise TypeError("loss_func must be a string")
        if not isinstance(self.tanimoto_similarity_loss_func, str | None):
            raise TypeError("tanimoto_similarity_loss_func must be a string or None")
        if not isinstance(self.eval_similarity_fct, str):
            raise TypeError("eval_similarity_fct must be a string")
        if not isinstance(self.eval_metric, str):
            raise TypeError("eval_metric must be a string")
        if not isinstance(self.mrl_dimensions, list | tuple):
            raise TypeError("mrl_dimensions must be a list or tuple")
        if not isinstance(self.mrl_dimension_weights, list | tuple):
            raise TypeError("mrl_dimension_weights must be a list or tuple")
        if not isinstance(self.n_dims_per_step, int):
            raise TypeError("n_dims_per_step must be an int")
        if not isinstance(self.use_2d_matryoshka, bool):
            raise TypeError("use_2d_matryoshka must be a bool")
        # check values
        if self.smiles_a_column_name == "":
            raise ValueError("smiles_a_column_name must be set")
        if self.smiles_b_column_name == "":
            raise ValueError("smiles_b_column_name must be set")
        if self.label_column_name == "":
            raise ValueError("label_column_name must be set")
        if self.embedding_pooling not in CHEM_MRL_EMBEDDING_POOLING_OPTIONS:
            raise ValueError(
                f"embedding_pooling must be one of {CHEM_MRL_EMBEDDING_POOLING_OPTIONS}"
            )
        if self.loss_func not in CHEM_MRL_LOSS_FCT_OPTIONS:
            raise ValueError(f"loss_func must be one of {CHEM_MRL_LOSS_FCT_OPTIONS}")
        if (self.tanimoto_similarity_loss_func is not None) and (
            self.tanimoto_similarity_loss_func
            not in TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS
        ):
            raise ValueError(
                f"tanimoto_similarity_loss_func must be one of {TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS}"
            )
        if self.eval_similarity_fct not in EVAL_SIMILARITY_FCT_OPTIONS:
            raise ValueError(
                f"eval_similarity_fct must be one of {EVAL_SIMILARITY_FCT_OPTIONS}"
            )
        if self.eval_metric not in CHEM_MRL_EVAL_METRIC_OPTIONS:
            raise ValueError(
                f"eval_metric must be one of {CHEM_MRL_EVAL_METRIC_OPTIONS}"
            )
        if len(self.mrl_dimension_weights) != len(self.mrl_dimensions):
            raise ValueError(
                "Number of dimension weights must match number of MRL dimensions"
            )
        if any(w <= 0 for w in self.mrl_dimension_weights):
            raise ValueError("All dimension weights must be positive")
        if not all(
            self.mrl_dimension_weights[i] <= self.mrl_dimension_weights[i + 1]
            for i in range(len(self.mrl_dimension_weights) - 1)
        ):
            raise ValueError("Dimension weights must be in increasing order")
        if self.n_dims_per_step != -1 and self.n_dims_per_step <= 0:
            raise ValueError("n_dims_per_step must be positive or -1")


@dataclass(frozen=True)
class Chem2dMRLConfig(ChemMRLConfig):
    use_2d_matryoshka: bool = True  # Explicitly enable 2D Matryoshka
    n_layers_per_step: int = -1
    last_layer_weight: float | int = 1
    prior_layers_weight: float | int = 1
    kl_div_weight: float | int = 1
    kl_temperature: float | int = 0.3

    def __post_init__(self):
        super().__post_init__()
        # check types
        if not isinstance(self.n_layers_per_step, int):
            raise TypeError("n_layers_per_step must be an int")
        if not isinstance(self.last_layer_weight, float | int):
            raise TypeError("last_layer_weight must be a float or int")
        if not isinstance(self.prior_layers_weight, float | int):
            raise TypeError("prior_layers_weight must be a float or int")
        if not isinstance(self.kl_div_weight, float | int):
            raise TypeError("kl_div_weight must be a float or int")
        if not isinstance(self.kl_temperature, float | int):
            raise TypeError("kl_temperature must be a float or int")
        # check values
        if self.use_2d_matryoshka is False:
            raise ValueError("use_2d_matryoshka must be True when training Chem2dMRL")
        if self.n_layers_per_step != -1 and self.n_layers_per_step <= 0:
            raise ValueError("n_layers_per_step must be positive or -1")
        if self.last_layer_weight <= 0:
            raise ValueError("last_layer_weight must be positive")
        if self.prior_layers_weight <= 0:
            raise ValueError("prior_layers_weight must be positive")
        if self.kl_div_weight <= 0:
            raise ValueError("kl_div_weight must be positive")
        if self.kl_temperature <= 0:
            raise ValueError("kl_temperature must be positive")
