from dataclasses import asdict, dataclass

from chem_mrl.constants import CHEM_MRL_MODEL_NAME, TRAINED_CHEM_MRL_DIMENSIONS

from .Enums import (
    ClassifierEvalMetricOption,
    ClassifierLossFctOption,
    DiceReductionOption,
)


@dataclass
class ClassifierConfig:
    model_name: str = CHEM_MRL_MODEL_NAME
    eval_metric: ClassifierEvalMetricOption = ClassifierEvalMetricOption.accuracy
    loss_func: ClassifierLossFctOption = ClassifierLossFctOption.softmax
    classifier_hidden_dimension: int = TRAINED_CHEM_MRL_DIMENSIONS[0]
    dropout_p: float = 0.1
    freeze_model: bool = False
    num_labels: int = 4
    dice_reduction: DiceReductionOption = DiceReductionOption.mean
    dice_gamma: float = 1.0
    asdict = asdict

    def __post_init__(self):
        # check types
        if not isinstance(self.model_name, str):
            raise TypeError("model_name must be a string")
        if not isinstance(self.eval_metric, str):
            raise TypeError("eval_metric must be a string")
        if not isinstance(self.loss_func, str):
            raise TypeError("loss_func must be a string")
        if not isinstance(self.classifier_hidden_dimension, int):
            raise TypeError("classifier_hidden_dimension must be an integer")
        if not isinstance(self.dropout_p, float):
            raise TypeError("dropout_p must be a float")
        if not isinstance(self.freeze_model, bool):
            raise TypeError("freeze_model must be a boolean")
        if not isinstance(self.num_labels, int):
            raise TypeError("num_labels must be an integer")
        if not isinstance(self.dice_reduction, str):
            raise TypeError("dice_reduction must be a string")
        if not isinstance(self.dice_gamma, float):
            raise TypeError("dice_gamma must be a float")
        # check values
        if self.model_name == "":
            raise ValueError("model_name must be set")
        if not isinstance(self.eval_metric, ClassifierEvalMetricOption):
            raise ValueError(f"eval_metric must be one of {ClassifierEvalMetricOption.to_list()}")
        if not isinstance(self.loss_func, ClassifierLossFctOption):
            raise ValueError(f"loss_func must be one of {ClassifierLossFctOption.to_list()}")
        if self.classifier_hidden_dimension < 1:
            raise ValueError("classifier_hidden_dimension must be greater than 0")
        if self.num_labels < 1:
            raise ValueError("num_labels must be greater than 0")
        if not (0 <= self.dropout_p <= 1):
            raise ValueError("dropout_p must be between 0 and 1")
        if self.dice_gamma < 0:
            raise ValueError("dice_gamma must be positive")
        if not isinstance(self.dice_reduction, DiceReductionOption):
            raise ValueError("dice_reduction must be either 'mean' or 'sum'")
