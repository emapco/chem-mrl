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

from enum import Enum


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"{value} is not a valid {cls.__name__}, please select one of {cls.to_list()}")

    def __str__(self):
        return self.value

    @classmethod
    def to_list(cls):
        return list(map(lambda c: c.value, cls))


class FieldTypeOption(ExplicitEnum):
    float64 = "float64"
    float32 = "float32"
    float16 = "float16"
    int64 = "int64"  # used for classification tasks


class EmbeddingPoolingOption(ExplicitEnum):
    mean = "mean"
    mean_sqrt_len_tokens = "mean_sqrt_len_tokens"
    weightedmean = "weightedmean"


class ChemMrlLossFctOption(ExplicitEnum):
    tanimotosentloss = "tanimotosentloss"
    tanimotosimilarityloss = "tanimotosimilarityloss"
    cosentloss = "cosentloss"
    angleloss = "angleloss"


class TanimotoSimilarityBaseLossFctOption(ExplicitEnum):
    mse = "mse"
    l1 = "l1"
    smooth_l1 = "smooth_l1"
    huber = "huber"
    bin_cross_entropy = "bin_cross_entropy"
    kldiv = "kldiv"
    cosine_embedding_loss = "cosine_embedding_loss"


class EvalSimilarityFctOption(ExplicitEnum):
    tanimoto = "tanimoto"
    cosine = "cosine"
    dot_product = "dot"
    dot = "dot"
    euclidean = "euclidean"
    manhattan = "manhattan"


class ChemMrlEvalMetricOption(ExplicitEnum):
    spearman = "spearman"
    pearson = "pearson"


class ClassifierEvalMetricOption(ExplicitEnum):
    accuracy = "accuracy"


class ClassifierLossFctOption(ExplicitEnum):
    softmax = "softmax"
    selfadjdice = "selfadjdice"


class DiceReductionOption(ExplicitEnum):
    mean = "mean"
    sum = "sum"
