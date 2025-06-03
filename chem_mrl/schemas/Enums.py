from enum import Enum


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {cls.to_list()}"
        )

    def __str__(self):
        return self.value

    @classmethod
    def to_list(cls):
        return list(map(lambda c: c.value, cls))


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
