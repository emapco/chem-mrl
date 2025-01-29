from dataclasses import dataclass

from chem_mrl.configs import BaseConfig
from chem_mrl.constants import CHEM_MRL_DIMENSIONS

from .types import (
    CHEM_MRL_EVAL_METRIC_OPTIONS,
    CHEM_MRL_LOSS_FCT_OPTIONS,
    EVAL_SIMILARITY_FCT_OPTIONS,
    TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS,
    ChemMrlEvalMetricOptionType,
    ChemMrlLossFctOptionType,
    EvalSimilarityMetricOptionType,
    TanimotoSimilarityBaseLossFctOptionType,
)


@dataclass
class ChemMRLConfig(BaseConfig):
    smiles_a_column_name: str = "smiles_a"
    smiles_b_column_name: str = "smiles_b"
    label_column_name: str = "fingerprint_similarity"
    loss_func: ChemMrlLossFctOptionType = "tanimotosentloss"  # type: ignore
    tanimoto_similarity_loss_func: TanimotoSimilarityBaseLossFctOptionType | None = None  # type: ignore
    eval_similarity_fct: EvalSimilarityMetricOptionType = "tanimoto"  # type: ignore
    eval_metric: ChemMrlEvalMetricOptionType = "spearman"  # type: ignore
    mrl_dimensions = CHEM_MRL_DIMENSIONS
    mrl_dimension_weights: tuple[float, float, float, float, float, float] = (
        1.0489590183361719,
        1.126163907196291,
        1.3807986616809407,
        1.397331091971628,
        1.6522851342433993,
        1.9858679040493405,
    )
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
        if not isinstance(self.use_2d_matryoshka, bool):
            raise TypeError("use_2d_matryoshka must be a bool")
        # check values
        if self.smiles_a_column_name == "":
            raise ValueError("smiles_a_column_name must be set")
        if self.smiles_b_column_name == "":
            raise ValueError("smiles_b_column_name must be set")
        if self.label_column_name == "":
            raise ValueError("label_column_name must be set")
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


@dataclass
class Chem2dMRLConfig(ChemMRLConfig):
    use_2d_matryoshka: bool = True  # Explicitly enable 2D Matryoshka
    last_layer_weight: float | int = 1.8708220063487997
    prior_layers_weight: float | int = 1.4598249321447245

    def __post_init__(self):
        super().__post_init__()
        # check types
        if not isinstance(self.last_layer_weight, float | int):
            raise TypeError("last_layer_weight must be a float or int")
        if not isinstance(self.prior_layers_weight, float | int):
            raise TypeError("prior_layers_weight must be a float or int")
        # check values
        if self.use_2d_matryoshka is False:
            raise ValueError("use_2d_matryoshka must be True when training Chem2dMRL")
        if self.last_layer_weight <= 0:
            raise ValueError("last_layer_weight must be positive")
        if self.prior_layers_weight <= 0:
            raise ValueError("prior_layers_weight must be positive")
