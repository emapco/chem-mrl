from dataclasses import dataclass
from typing import Literal

from chem_mrl.configs import BaseConfig
from chem_mrl.constants import (
    BASE_MODEL_NAME,
    CHEM_MRL_DATASET_KEYS,
    CHEM_MRL_DIMENSIONS,
)

_tanimoto_loss_func_option_type = Literal[
    "tanimotosentloss", "tanimotosimilarityloss", "cosentloss"
]
_tanimoto_loss_func_options: tuple[_tanimoto_loss_func_option_type, ...] = (
    "tanimotosentloss",
    "tanimotosimilarityloss",
    "cosentloss",
)
_tanimoto_similarity_base_loss_func_option_type = Literal[
    "mse",
    "l1",
    "smooth_l1",
    "huber",
    "bin_cross_entropy",
    "kldiv",
    "cosine_embedding_loss",
]
_tanimoto_similarity_base_loss_func_options: tuple[
    _tanimoto_similarity_base_loss_func_option_type, ...
] = (
    "mse",
    "l1",
    "smooth_l1",
    "huber",
    "bin_cross_entropy",
    "kldiv",
    "cosine_embedding_loss",
)


@dataclass
class ChemMRLConfig(BaseConfig):
    dataset_key: str = CHEM_MRL_DATASET_KEYS[0]
    smiles_a_column_name: str = "smiles_a"
    smiles_b_column_name: str = "smiles_b"
    label_column_name: str = "fingerprint_similarity"
    model_name: str = BASE_MODEL_NAME
    loss_func: _tanimoto_loss_func_option_type = "tanimotosentloss"
    tanimoto_similarity_loss_func: (
        _tanimoto_similarity_base_loss_func_option_type | None
    ) = None
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
        if self.dataset_key not in CHEM_MRL_DATASET_KEYS:
            raise ValueError(f"dataset_key must be one of {CHEM_MRL_DATASET_KEYS}")
        if self.smiles_a_column_name == "":
            raise ValueError("smiles_a_column_name must be set")
        if self.smiles_b_column_name == "":
            raise ValueError("smiles_b_column_name must be set")
        if self.label_column_name == "":
            raise ValueError("label_column_name must be set")
        if self.model_name == "":
            raise ValueError(f"model_name must be set (e.g. `{BASE_MODEL_NAME}`)")
        if self.loss_func not in _tanimoto_loss_func_options:
            raise ValueError(f"loss_func must be one of {_tanimoto_loss_func_options}")
        if (self.tanimoto_similarity_loss_func is not None) and (
            self.tanimoto_similarity_loss_func
            not in _tanimoto_similarity_base_loss_func_options
        ):
            raise ValueError(
                f"tanimoto_similarity_loss_func must be one of {_tanimoto_similarity_base_loss_func_options}"
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
    last_layer_weight: float = 1.8708220063487997
    prior_layers_weight: float = 1.4598249321447245

    def __post_init__(self):
        super().__post_init__()
        if self.use_2d_matryoshka is False:
            raise ValueError("use_2d_matryoshka must be True when training Chem2dMRL")
        if self.last_layer_weight <= 0:
            raise ValueError("last_layer_weight must be positive")
        if self.prior_layers_weight <= 0:
            raise ValueError("prior_layers_weight must be positive")
