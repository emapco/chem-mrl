from typing import Literal

#########################################################
# Config type definitions
#########################################################

# Base Config types
WATCH_LOG_OPTIONS = (
    "gradients",
    "parameters",
    "all",
)
WatchLogOptionType = Literal[*WATCH_LOG_OPTIONS]
SCHEDULER_OPTIONS = (
    "warmupconstant",
    "warmuplinear",
    "warmupcosine",
    "warmupcosinewithhardrestarts",
)
SchedulerOptionType = Literal[*SCHEDULER_OPTIONS]

# ChemMrl Config types
# # For tanimoto loss functions
CHEM_MRL_LOSS_FCT_OPTIONS = (
    "tanimotosentloss",
    "tanimotosimilarityloss",
    "cosentloss",
)
ChemMrlLossFctOptionType = Literal[*CHEM_MRL_LOSS_FCT_OPTIONS]
# # For tanimoto similarity base loss functions
TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS = (
    "mse",
    "l1",
    "smooth_l1",
    "huber",
    "bin_cross_entropy",
    "kldiv",
    "cosine_embedding_loss",
)
TanimotoSimilarityBaseLossFctOptionType = Literal[
    *TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS
]
# # For eval similarity metrics
EVAL_SIMILARITY_FCT_OPTIONS = (
    "cosine",
    "tanimoto",
)
EvalSimilarityMetricOptionType = Literal[*EVAL_SIMILARITY_FCT_OPTIONS]
CHEM_MRL_EVAL_METRIC_OPTIONS = (
    "spearman",
    "pearson",
)
ChemMrlEvalMetricOptionType = Literal[*CHEM_MRL_EVAL_METRIC_OPTIONS]

# Classifier config types
CLASSIFIER_LOSS_FCT_OPTIONS = (
    "softmax",
    "selfadjdice",
)
ClassifierLossFctOptionType = Literal[*CLASSIFIER_LOSS_FCT_OPTIONS]
CLASSIFIER_EVAL_METRIC_OPTIONS = ("accuracy",)
ClassifierEvalMetricOptionType = Literal[*CLASSIFIER_EVAL_METRIC_OPTIONS]
# For dice reduction
DICE_REDUCTION_OPTIONS = (
    "mean",
    "sum",
)
DiceReductionOptionType = Literal[*DICE_REDUCTION_OPTIONS]
