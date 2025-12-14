# Copyright 2025 Emmanuel Cortes. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass

from chem_mrl.constants import CHEM_MRL_MODEL_NAME, TRAINED_CHEM_MRL_DIMENSIONS

from .Enums import (
    ClassifierEvalMetricOption,
    ClassifierLossFctOption,
    DiceReductionOption,
)


@dataclass
class ClassifierConfig:
    """Configuration for the Classifier model.

    Attributes:
        model_name: Name of the model to use. Must be either a file path or a Sentence Transformer model name.
        eval_metric: Metric to use for evaluation.
        loss_func: Loss function.
        classifier_hidden_dimension: Classifier hidden dimension. Must be less than equal to the ChemMRL transformer's
            hidden dimension. Note, the base model will be truncated to this dimension.
        dropout_p: Dropout probability for linear layer regularization.
        freeze_model: Freeze internal base MRL model.
        num_labels: Number of labels.
        dice_reduction: Reduction to apply to the output (only used if loss_func=selfadjdice).
        dice_gamma: Smoothing factor for numerator and denominator (only used if loss_func=selfadjdice).
    """

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
