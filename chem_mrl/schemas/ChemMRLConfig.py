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

from dataclasses import asdict, dataclass, field

from chem_mrl.constants import BASE_MODEL_NAME, CHEM_MRL_DIMENSIONS

from .Enums import (
    ChemMrlEvalMetricOption,
    ChemMrlLossFctOption,
    EmbeddingPoolingOption,
    EvalSimilarityFctOption,
    TanimotoSimilarityBaseLossFctOption,
)
from .MaxPoolBERTConfig import MaxPoolBERTConfig


@dataclass
class ChemMRLConfig:
    model_name: str = BASE_MODEL_NAME
    embedding_pooling: EmbeddingPoolingOption = EmbeddingPoolingOption.mean
    loss_func: ChemMrlLossFctOption = ChemMrlLossFctOption.tanimotosentloss
    tanimoto_similarity_loss_func: TanimotoSimilarityBaseLossFctOption | None = None
    eval_similarity_fct: EvalSimilarityFctOption = EvalSimilarityFctOption.tanimoto
    eval_metric: ChemMrlEvalMetricOption = ChemMrlEvalMetricOption.spearman
    mrl_dimensions: tuple = tuple(CHEM_MRL_DIMENSIONS)
    mrl_dimension_weights: tuple = (1, 1, 1, 1, 1, 1, 1, 1)
    n_dims_per_step: int = -1
    use_2d_matryoshka: bool = False
    n_layers_per_step: int = -1
    last_layer_weight: float | int = 1
    prior_layers_weight: float | int = 1
    kl_div_weight: float | int = 1
    kl_temperature: float | int = 0.3
    max_pool_bert: MaxPoolBERTConfig = field(default_factory=MaxPoolBERTConfig)
    asdict = asdict

    def __post_init__(self):
        # check types
        if not isinstance(self.model_name, str):
            raise TypeError("model_name must be a string")
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
        if not isinstance(self.max_pool_bert, MaxPoolBERTConfig):
            raise TypeError("max_pool_bert must be a MaxPoolBERTConfig instance")
        # check values
        if self.model_name == "":
            raise ValueError("model_name must be set")
        if not isinstance(self.embedding_pooling, EmbeddingPoolingOption):
            raise ValueError(f"embedding_pooling must be one of {EmbeddingPoolingOption.to_list()}")
        if not isinstance(self.loss_func, ChemMrlLossFctOption):
            raise ValueError(f"loss_func must be one of {ChemMrlLossFctOption.to_list()}")
        if (self.tanimoto_similarity_loss_func is not None) and (
            not isinstance(self.tanimoto_similarity_loss_func, TanimotoSimilarityBaseLossFctOption)
        ):
            raise ValueError(
                "tanimoto_similarity_loss_func must be one of ",
                TanimotoSimilarityBaseLossFctOption.to_list(),
            )
        if not isinstance(self.eval_similarity_fct, EvalSimilarityFctOption):
            raise ValueError(f"eval_similarity_fct must be one of {EvalSimilarityFctOption.to_list()}")
        if not isinstance(self.eval_metric, ChemMrlEvalMetricOption):
            raise ValueError(f"eval_metric must be one of {ChemMrlEvalMetricOption.to_list()}")
        if len(self.mrl_dimension_weights) != len(self.mrl_dimensions):
            raise ValueError("Number of dimension weights must match number of MRL dimensions")
        if any(w <= 0 for w in self.mrl_dimension_weights):
            raise ValueError("All dimension weights must be positive")
        if not all(
            self.mrl_dimension_weights[i] <= self.mrl_dimension_weights[i + 1]
            for i in range(len(self.mrl_dimension_weights) - 1)
        ):
            raise ValueError("Dimension weights must be in increasing order")
        if self.n_dims_per_step != -1 and self.n_dims_per_step <= 0:
            raise ValueError("n_dims_per_step must be positive or -1")
        if self.n_layers_per_step != -1 and self.n_layers_per_step <= 0:
            raise ValueError("n_layers_per_step must be positive or -1")
        if self.last_layer_weight <= 0:
            raise ValueError("last_layer_weight must be positive")
        if self.prior_layers_weight <= 0:
            raise ValueError("prior_layers_weight must be positive")
        if self.kl_div_weight < 0:
            raise ValueError("kl_div_weight must be greater than or equal to zero")
        if self.kl_temperature < 0:
            raise ValueError("kl_temperature must be greater than or equal to zero")
        if self.max_pool_bert.enable and self.use_2d_matryoshka:
            raise ValueError("MaxPoolBERT is only supported for 1D MRL (use_2d_matryoshka must be False)")
