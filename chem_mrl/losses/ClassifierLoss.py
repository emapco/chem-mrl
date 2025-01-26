from typing import Callable, Dict, Iterable

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn

REDUCTION = {"mean", "sum"}


class _ClassifierLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        smiles_embedding_dimension: int,
        num_labels: int,
        dropout: float = 0.15,
        freeze_model: bool = False,
    ):
        """
        Base class for SMILES classification loss functions.

        Parameters
        ----------
        model : SentenceTransformer
            The sentence transformer model used to generate embeddings
        smiles_embedding_dimension : int
            The dimension of the SMILE
        num_labels : int
            The number of labels to classify
        dropout : float, optional
            The dropout rate to apply to the embeddings, defaults to 0.15
        freeze_model : bool, optional
            Whether to freeze the sentence transformer model, defaults to False.
            If True, only the classifier parameters will be updated during training.
        """
        super(_ClassifierLoss, self).__init__()
        self.model = model

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        self.smiles_embedding_dimension = smiles_embedding_dimension
        self.num_labels = num_labels
        self.dropout_p = dropout
        self.classifier = nn.Linear(
            smiles_embedding_dimension, num_labels, device=model.device
        )
        self.dropout = nn.Dropout(self.dropout_p, inplace=True)

    def forward(
        self, smiles_features: Iterable[Dict[str, Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sent_reps: list[Tensor] = [
            self.model(smiles_feature)["sentence_embedding"]
            for smiles_feature in smiles_features
        ]

        # guaranteed to be single smiles (sentence) embedding
        features = self._truncate_embeddings(sent_reps[0])
        self.dropout(features)
        logits: Tensor = self.classifier(features)

        return features, logits

    def _truncate_embeddings(self, embeddings: Tensor) -> Tensor:
        """Truncate embeddings if it exceeds classifier's input features. Only applicable to MRL models.
        Args:
            embeddings (Tensor): SMILES embeddings

        Returns:
            Tensor: truncated embeddings
        """
        batch_size, embedding_dim = embeddings.shape
        if embedding_dim > self.smiles_embedding_dimension:
            embeddings = embeddings[:, : self.smiles_embedding_dimension]
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def get_config_dict(self):
        return {
            "smiles_embedding_dimension": self.smiles_embedding_dimension,
            "num_labels": self.num_labels,
            "dropout": self.dropout_p,
        }


# https://github.com/fursovia/self-adj-dice
# https://aclanthology.org/2020.acl-main.45.pdf
class SelfAdjDiceLoss(_ClassifierLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        smiles_embedding_dimension: int,
        num_labels: int,
        dropout: float = 0.15,
        freeze_model: bool = False,
        alpha: float = 1.0,
        gamma: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        """
        This class implements a Self-adjusting Dice Loss for multi-class classification tasks, particularly suited for handling data imbalance.
        It extends the SentenceTransformer implementation by supporting single SMILES embedding classification and dropout regularization.

        Parameters
        ----------
        model : SentenceTransformer
            The sentence transformer model used to generate embeddings
        smiles_embedding_dimension : int
            The dimension of the SMILES embedding
        num_labels : int
            The number of labels to classify
        dropout : float, optional
            The dropout rate for regularization, defaults to 0.15
        feeze_model : bool, optional
            Whether to freeze the sentence transformer model, defaults to False.
        alpha : float, optional
            Factor to down-weight easy examples, defaults to 1.0
            `A close look at Eq.12 reveals that it actually mimics the idea of focal loss (FL for short) (Lin et al.,
                2017) for object detection in vision. Focal loss
                was proposed for one-stage object detector to handle foreground-background tradeoff encountered
                during training. It down-weights the loss assigned
                to well-classified examples by adding a (1 − p)**γ factor, leading the final loss to be −(1 − p)**γ * log p.
                `
                The alpha as implemented by fursovia (github user's implementation) seemly randomly (not from the equations) is actually listed as an aside where another paper/method includes a `(1-p)**alpha` factor instead of
                the self adj dice's (SAD) `(1-p)` factor. SAD/DSC modifies the Sørensen–Dice coefficient by including a `(1-p)` factor in the numerator.
                Keep alpha=1 unless keen on expanding hyperparameter search space.
        gamma : float, optional
            Smoothing factor for numerator and denominator, defaults to 1.0
            a factor added to both the nominator and the denominator for smoothing purposes
        reduction : str, optional
            Specifies the reduction to apply to the output: ``'mean'`` | ``'sum'``
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output
                ``'sum'``: the output will be summed.

        References:
            "Dice Loss for Data-imbalanced NLP Tasks" paper
            Implementation inspired by: https://github.com/fursovia/self-adj-dice

        Inputs:
            +--------------------------------+--------+
            | Texts                          | Labels |
            +================================+========+
            | SMILES string                  | class  |
            +--------------------------------+--------+
        """  # noqa
        super().__init__(
            model, smiles_embedding_dimension, num_labels, dropout, freeze_model
        )
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        smiles_features: Iterable[Dict[str, Tensor]],
        labels: Tensor,
    ) -> torch.Tensor:

        features, logits = super().forward(smiles_features)

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

    def get_config_dict(self):
        return {
            **super().get_config_dict(),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "reduction": self.reduction,
        }


class SoftmaxLoss(_ClassifierLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        smiles_embedding_dimension: int,
        num_labels: int,
        dropout: float = 0.15,
        freeze_model: bool = False,
        loss_fct: Callable = nn.CrossEntropyLoss(),
    ):
        """
        This class implements the softmax loss function for SMILES classification.
        By default, it uses the cross entropy loss function but can be changed to any other loss function.
        It is designed to be used with the SentenceTransformer model.

        Parameters
        ----------
        model : SentenceTransformer
            The sentence transformer model used to generate embeddings
        smiles_embedding_dimension : int
            The dimension of the SMILES embedding
        num_labels : int
            The number of labels to classify
        dropout : float, optional
            The dropout rate to apply to the embeddings, defaults to 0.15
        freeze_model : bool, optional
            Whether to freeze the model weights, defaults to False
        loss : nn.Module, optional
            The base loss function to compute the final loss value, defaults to nn.MSELoss()

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | SMILES string                         | class  |
            +---------------------------------------+--------+
        """
        super().__init__(
            model, smiles_embedding_dimension, num_labels, dropout, freeze_model
        )
        self.loss_fct = loss_fct

    def forward(self, smiles_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        features, logits = super().forward(smiles_features)

        if labels is None:
            return features, logits

        loss = self.loss_fct(logits, labels.view(-1))
        return loss

    def get_config_dict(self):
        return {
            **super().get_config_dict(),
            "loss_fct": type(self.loss_fct).__name__,
        }
