import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable
from sentence_transformers import SentenceTransformer
import logging


logger = logging.getLogger(__name__)


REDUCTION = {"mean", "sum"}


class ClassifierLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        sentence_embedding_dimension: int,
        num_labels: int,
        dropout: float = 0.15,
    ):
        super(ClassifierLoss, self).__init__()
        self.model = model
        self.classifier = nn.Linear(
            sentence_embedding_dimension, num_labels, device=model.device
        )
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(
        self, sentence_features: Iterable[Dict[str, Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sent_reps: list[Tensor] = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        # guaranteed to be single sentence embedding
        features = self._truncate_embeddings(sent_reps[0])
        self.dropout(features)
        logits: Tensor = self.classifier(features)

        return features, logits

    def _truncate_embeddings(self, embeddings: Tensor) -> Tensor:
        """Truncate embeddings if it exceeds classifier's input features. Only applicable to MRL models.
        Args:
            embeddings (Tensor): sentence embeddings

        Returns:
            Tensor: truncated embeddings
        """
        batch_size, embedding_dim = embeddings.shape
        if embedding_dim > self.classifier.in_features:
            embeddings = embeddings[:, : self.classifier.in_features]
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


# https://github.com/fursovia/self-adj-dice
# https://aclanthology.org/2020.acl-main.45.pdf
class SelfAdjDiceLoss(ClassifierLoss):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)

    This class mimics SentenceTransformers implementation which contains the linear classifier in the loss function.
    These loss functions, however, allows:
        - single sentence embedding classification (instead of just sentence-pair embedding classification like in ST implementation)
        - dropout for regularization of the linear classifier

    Args:
        alpha (float): a factor to push down the weight of easy examples
            `A close look at Eq.12 reveals that it actually mimics the idea of focal loss (FL for short) (Lin et al.,
            2017) for object detection in vision. Focal loss
            was proposed for one-stage object detector to handle foreground-background tradeoff encountered
            during training. It down-weights the loss assigned
            to well-classified examples by adding a (1 − p)**γ factor, leading the final loss to be −(1 − p)**γ * log p.
            `
            The alpha as implemented by fursovia (github user's implementation) seemly randomly (not from the equations) is actually listed as an aside where another paper/method includes a `(1-p)**alpha` factor instead of
            the self adj dice's (SAD) `(1-p)` factor. SAD/DSC modifies the Sørensen–Dice coefficient by including a `(1-p)` factor in the numerator.
            Keep alpha=1 unless keen on expanding hyperparameter search space.

        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """  # noqa

    def __init__(
        self,
        model: SentenceTransformer,
        sentence_embedding_dimension: int,
        num_labels: int,
        dropout: float = 0.15,
        alpha: float = 1.0,
        gamma: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(model, sentence_embedding_dimension, num_labels, dropout)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        sentence_features: Iterable[Dict[str, Tensor]],
        labels: Tensor,
    ) -> torch.Tensor:

        features, logits = super().forward(sentence_features)

        if labels is None:
            return features, logits

        # dice loss
        probs = torch.softmax(logits, dim=1)
        # dice paper pg 467. - `As can be seen, a negative example (yi1 = 0) does not contribute to the objective.`
        # yi1 - essentially a kronecker delta
        yi1 = 1
        # gather ensure only the positive examples contribute (yi1)
        probs = torch.gather(probs, dim=1, index=labels.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (
            probs_with_factor + yi1 + self.gamma
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")


# SoftmaxLoss implementation taken from SentenceTransformer library
# and modified for single sentence (SMILES) embedding classification
class SoftmaxLoss(ClassifierLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        sentence_embedding_dimension: int,
        num_labels: int,
        dropout: float = 0.15,
        loss_fct: Callable = nn.CrossEntropyLoss(),
    ):
        """
        This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
        model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

        :class:`MultipleNegativesRankingLoss` is an alternative loss function that often yields better results,
        as per https://arxiv.org/abs/2004.09813.

        :param model: SentenceTransformer model
        :param sentence_embedding_dimension: Dimension of your sentence embeddings
        :param num_labels: Number of different labels
        :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
        :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
        :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
        :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()

        References:
            - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks: https://arxiv.org/abs/1908.10084
            - `Training Examples > Natural Language Inference <../../examples/training/nli/README.html>`_

        Requirements:
            1. sentence pairs with a class label

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (sentence_A, sentence_B) pairs        | class  |
            +---------------------------------------+--------+
        """
        super().__init__(model, sentence_embedding_dimension, num_labels, dropout)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        features, logits = super().forward(sentence_features)

        if labels is None:
            return features, logits

        loss = self.loss_fct(logits, labels.view(-1))
        return loss
