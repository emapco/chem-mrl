from dataclasses import dataclass

from chem_mrl.constants import CHEM_MRL_DIMENSIONS

from .BaseConfig import _BaseConfig
from .types import (
    CLASSIFIER_EVAL_METRIC_OPTIONS,
    CLASSIFIER_LOSS_FCT_OPTIONS,
    DICE_REDUCTION_OPTIONS,
    ClassifierEvalMetricOptionType,
    ClassifierLossFctOptionType,
    DiceReductionOptionType,
)


@dataclass(frozen=True)
class ClassifierConfig(_BaseConfig):
    smiles_column_name: str = "smiles"
    label_column_name: str = "label"
    eval_metric: ClassifierEvalMetricOptionType = "accuracy"  # type: ignore
    loss_func: ClassifierLossFctOptionType = "softmax"  # type: ignore
    classifier_hidden_dimension: int = CHEM_MRL_DIMENSIONS[0]
    dropout_p: float = 0.1
    freeze_model: bool = False

    def __post_init__(self):
        super().__post_init__()
        # check types
        if not isinstance(self.smiles_column_name, str):
            raise TypeError("smiles_column_name must be a string")
        if not isinstance(self.label_column_name, str):
            raise TypeError("label_column_name must be a string")
        if not isinstance(self.eval_metric, str):
            raise TypeError("evaluation_metric must be a string")
        if not isinstance(self.loss_func, str):
            raise TypeError("loss_func must be a string")
        if not isinstance(self.classifier_hidden_dimension, int):
            raise TypeError("classifier_hidden_dimension must be an integer")
        if not isinstance(self.dropout_p, float):
            raise TypeError("dropout_p must be a float")
        if not isinstance(self.freeze_model, bool):
            raise TypeError("freeze_model must be a boolean")
        # check values
        if self.smiles_column_name == "":
            raise ValueError("smiles_column_name must be set")
        if self.label_column_name == "":
            raise ValueError("label_column_name must be set")
        if self.eval_metric not in CLASSIFIER_EVAL_METRIC_OPTIONS:
            raise ValueError(
                f"eval_metric must be one of {CLASSIFIER_EVAL_METRIC_OPTIONS}"
            )
        if self.loss_func not in CLASSIFIER_LOSS_FCT_OPTIONS:
            raise ValueError(f"loss_func must be one of {CLASSIFIER_LOSS_FCT_OPTIONS}")
        if self.classifier_hidden_dimension not in CHEM_MRL_DIMENSIONS:
            raise ValueError(
                f"classifier_hidden_dimension must be one of {CHEM_MRL_DIMENSIONS}"
            )
        if not (0 <= self.dropout_p <= 1):
            raise ValueError("dropout_p must be between 0 and 1")


@dataclass(frozen=True)
class DiceLossClassifierConfig(ClassifierConfig):
    dice_reduction: DiceReductionOptionType = "mean"  # type: ignore
    dice_gamma: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        # check types
        if not isinstance(self.dice_reduction, str):
            raise TypeError("dice_reduction must be a string")
        if not isinstance(self.dice_gamma, float | int):
            raise TypeError("dice_gamma must be a float or int")
        # check values
        if self.dice_gamma < 0:
            raise ValueError("dice_gamma must be positive")
        if self.dice_reduction not in DICE_REDUCTION_OPTIONS:
            raise ValueError("dice_reduction must be either 'mean' or 'sum'")
