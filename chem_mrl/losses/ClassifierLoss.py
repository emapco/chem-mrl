from collections.abc import Callable, Iterable

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
            The dropout rate to apply to the embeddings, defaults to 0.15.
            If set to 0, no dropout will be applied.
        freeze_model : bool, optional
            Whether to freeze the sentence transformer model, defaults to False.
            If True, only the classifier parameters will be updated during training.
        """
        super().__init__()
        self.__model = model
        self.__freeze_model = freeze_model

        if self.__freeze_model:
            for param in self.__model.parameters():
                param.requires_grad = False

        self.__smiles_embedding_dimension = smiles_embedding_dimension
        self.__num_labels = num_labels
        self.__dropout_p = dropout
        # strategy pattern - determine whether to apply dropout or no-op at runtime
        if dropout > 0:
            self.__dropout = nn.Dropout(self.__dropout_p)
        else:
            self.__dropout = nn.Identity()  # no-op
        self.__net = nn.Sequential(
            self.__dropout,
            nn.Linear(smiles_embedding_dimension, smiles_embedding_dimension),
            nn.Tanh(),
            self.__dropout,
            nn.Linear(smiles_embedding_dimension, num_labels),
        )

    @property
    def freeze_model(self):
        return self.__freeze_model

    @property
    def smiles_embedding_dimension(self):
        return self.__smiles_embedding_dimension

    @property
    def num_labels(self):
        return self.__num_labels

    @property
    def dropout_p(self):
        return self.__dropout_p

    def preprocess(
        self, smiles_features: Iterable[dict[str, Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sent_reps: list[Tensor] = [
            self.__model(smiles_feature)["sentence_embedding"] for smiles_feature in smiles_features
        ]
        # guaranteed to be a single smiles (sentence) embedding
        features = self._truncate_embeddings(sent_reps[0])
        logits = self.__net(features)
        return features, logits

    def _truncate_embeddings(self, embeddings: Tensor) -> Tensor:
        """Truncate embeddings if it exceeds classifier's input features.
        Only applicable to MRL models.

        Args:
            embeddings (Tensor): SMILES embeddings

        Returns:
            Tensor: truncated embeddings
        """
        batch_size, embedding_dim = embeddings.shape
        if embedding_dim > self.__smiles_embedding_dimension:
            embeddings = embeddings[:, : self.__smiles_embedding_dimension]
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def get_config_dict(self):
        return {
            "smiles_embedding_dimension": self.__smiles_embedding_dimension,
            "num_labels": self.__num_labels,
            "dropout": self.__dropout_p,
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
            `A close look at Eq.12 reveals that it actually mimics the idea of focal loss
            (FL for short) (Lin et al., 2017) for object detection in vision. Focal loss
            was proposed for one-stage object detector to handle foreground-background
            tradeoff encountered during training. It down-weights the loss assigned
            to well-classified examples by adding a (1 − p)**γ factor,
            leading the final loss to be −(1 − p)**γ * log p.
            `
            The alpha as implemented by fursovia (github user's implementation) seemly randomly (not from the equations)
            is actually listed as an aside where another paper/method includes a `(1-p)**alpha` factor instead of
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
        """  # noqa: E501
        super().__init__(model, smiles_embedding_dimension, num_labels, dropout, freeze_model)
        self.__alpha = alpha
        self.__gamma = gamma
        self.__reduction = reduction

    def forward(self, smiles_features: Iterable[dict[str, Tensor]], labels: Tensor | None):
        features, logits = self.preprocess(smiles_features)

        if labels is None:
            return features, logits

        # dice loss
        probs = torch.softmax(logits, dim=1)
        # dice paper pg 467.
        # `As can be seen, a negative example (yi1 = 0) does not contribute to the objective.`
        # yi1 - essentially a kronecker delta
        yi1 = 1
        # gather ensure only the positive examples contribute (yi1)
        probs = torch.gather(probs, dim=1, index=labels.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.__alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.__gamma) / (probs_with_factor + yi1 + self.__gamma)

        if self.__reduction == "mean":
            return loss.mean()
        elif self.__reduction == "sum":
            return loss.sum()
        else:
            raise NotImplementedError(f"Reduction `{self.__reduction}` is not supported.")

    def get_config_dict(self):
        return {
            **super().get_config_dict(),
            "alpha": self.__alpha,
            "gamma": self.__gamma,
            "reduction": self.__reduction,
        }


class SoftmaxLoss(_ClassifierLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        smiles_embedding_dimension: int,
        num_labels: int,
        dropout: float = 0.15,
        freeze_model: bool = False,
        loss_fct: Callable = nn.CrossEntropyLoss(),  # noqa: B008
    ):
        """
        This class implements the softmax loss function for SMILES classification.
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
        super().__init__(model, smiles_embedding_dimension, num_labels, dropout, freeze_model)
        self.__loss_fct = loss_fct

    def forward(self, smiles_features: Iterable[dict[str, Tensor]], labels: Tensor | None):
        features, logits = self.preprocess(smiles_features)

        if labels is None:
            return features, logits

        loss = self.__loss_fct(logits, labels.view(-1))
        return loss

    def get_config_dict(self):
        return {
            **super().get_config_dict(),
            "loss_fct": type(self.__loss_fct).__name__,
        }
