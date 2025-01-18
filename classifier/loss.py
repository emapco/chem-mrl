import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable
from sentence_transformers import SentenceTransformer
import logging


logger = logging.getLogger(__name__)

# https://github.com/fursovia/self-adj-dice
# https://aclanthology.org/2020.acl-main.45.pdf

REDUCTION = {"mean", "sum", "none"}


class SelfAdjDiceLoss(torch.nn.Module):
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
            the self adj dice's (SAD) `(1-p)` factor. SAD/DSC modifies the Sørensen–Dice coefficientcoefficient by including a `(1-p)` factor in the numerator.
            Keep alpha=1 unless keen on expanding hyperparameter search space.

        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(
        self,
        model: SentenceTransformer,
        sentence_embedding_dimension: int,
        num_labels: int,
        concatenation_sent_rep: bool = False,
        concatenation_sent_difference: bool = False,
        concatenation_sent_multiplication: bool = False,
        alpha: float = 1.0,
        gamma: float = 1.0,
        reduction: str = "mean",
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        elif concatenation_sent_difference:
            num_vectors_concatenated += 1
        elif concatenation_sent_multiplication:
            num_vectors_concatenated += 1

        if num_vectors_concatenated != 0:
            self.classifier = nn.Linear(
                num_vectors_concatenated * sentence_embedding_dimension,
                num_labels,
                device=model.device,
            )
        else:
            self.classifier = nn.Linear(
                sentence_embedding_dimension, num_labels, device=model.device
            )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor
    ) -> torch.Tensor:
        sent_reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        features = sent_reps[0]  # at least one guaranteed
        vectors_concat = []

        rep_a, rep_b = sent_reps
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)
        elif self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))
        elif self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)
        features = self.dropout(features)
        logits = self.classifier(features)

        if not (
            self.concatenation_sent_difference
            or self.concatenation_sent_multiplication
            or self.concatenation_sent_rep
        ):
            logits = logits.view(-1, self.num_labels)

        if labels is None:
            return sent_reps, logits

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
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")


# SoftmaxLoss implementation taken from SentenceTranformer library and modified. Might be able to create a subclass
# based on the ST's Softmax class and then have custom code in child class.
class SoftmaxLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        sentence_embedding_dimension: int,
        num_labels: int,
        concatenation_sent_rep: bool = False,
        concatenation_sent_difference: bool = False,
        concatenation_sent_multiplication: bool = False,
        loss_fct: Callable = nn.CrossEntropyLoss(),
        dropout: float = 0.15,
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

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentencesDataset, losses
                from sentence_transformers.readers import InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                train_examples = [
                    InputExample(texts=['First pair, sent A',  'First pair, sent B'], label=0),
                    InputExample(texts=['Second pair, sent A', 'Second pair, sent B'], label=1),
                    InputExample(texts=['Third pair, sent A',  'Third pair, sent B'], label=0),
                    InputExample(texts=['Fourth pair, sent A', 'Fourth pair, sent B'], label=2),
                ]
                train_batch_size = 2
                train_dataset = SentencesDataset(train_examples, model)
                train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
                train_loss = losses.SoftmaxLoss(
                    model=model,
                    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                    num_labels=len(set(x.label for x in train_examples))
                )
                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        elif concatenation_sent_difference:
            num_vectors_concatenated += 1
        elif concatenation_sent_multiplication:
            num_vectors_concatenated += 1

        if num_vectors_concatenated != 0:
            self.classifier = nn.Linear(
                num_vectors_concatenated * sentence_embedding_dimension,
                num_labels,
                device=model.device,
            )
        else:
            self.classifier = nn.Linear(
                sentence_embedding_dimension, num_labels, device=model.device
            )
        self.dropout = nn.Dropout(dropout)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        sent_reps = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        features = sent_reps[0]
        vectors_concat = []

        rep_a, rep_b = sent_reps
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)
        elif self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))
        elif self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)
        features = self.dropout(features)
        logits = self.classifier(features)

        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1))
            return loss
        else:
            return sent_reps, logits
