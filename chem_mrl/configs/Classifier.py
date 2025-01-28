from dataclasses import dataclass
from typing import Literal

from chem_mrl.configs import BaseConfig
from chem_mrl.constants import CHEM_MRL_DIMENSIONS, MODEL_NAME_KEYS, MODEL_NAMES

_classifier_loss_func_option_type = Literal["softmax", "selfadjdice"]
_classifier_loss_func_options: tuple[_classifier_loss_func_option_type, ...] = (
    "softmax",
    "selfadjdice",
)
_dice_reduction_option_type = Literal["mean", "sum"]
_dice_reduction_options: tuple[_dice_reduction_option_type, ...] = ("mean", "sum")


@dataclass
class ClassifierConfig(BaseConfig):
    model_name: str = MODEL_NAMES[MODEL_NAME_KEYS[1]]  #
    train_dataset_path: str = ""
    val_dataset_path: str = ""
    smiles_column_name: str = "smiles"
    label_column_name: str = "label"
    loss_func: _classifier_loss_func_option_type = "softmax"
    classifier_hidden_dimension: int = CHEM_MRL_DIMENSIONS[0]
    dropout_p: float = 0.15
    freeze_model: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.model_name == "":
            raise ValueError("model_name must be set")
        if self.train_dataset_path == "":
            raise ValueError("train_dataset_path must be set")
        if self.val_dataset_path == "":
            raise ValueError("val_dataset_path must be set")
        if self.smiles_column_name == "":
            raise ValueError("smiles_column_name must be set")
        if self.label_column_name == "":
            raise ValueError("label_column_name must be set")
        if self.loss_func not in _classifier_loss_func_options:
            raise ValueError(
                f"loss_func must be one of {_classifier_loss_func_options}"
            )
        if self.classifier_hidden_dimension not in CHEM_MRL_DIMENSIONS:
            raise ValueError(
                f"classifier_hidden_dimension must be one of {CHEM_MRL_DIMENSIONS}"
            )
        if not (0 <= self.dropout_p <= 1):
            raise ValueError("dropout_p must be between 0 and 1")


@dataclass
class DiceLossClassifierConfig(ClassifierConfig):
    dice_reduction: _dice_reduction_option_type = "mean"
    dice_gamma: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.dice_gamma < 0:
            raise ValueError("dice_gamma must be positive")
        if self.dice_reduction not in _dice_reduction_options:
            raise ValueError("dice_reduction must be either 'mean' or 'sum'")
