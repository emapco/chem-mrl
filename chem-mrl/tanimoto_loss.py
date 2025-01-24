from typing import Dict, Iterable

import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.losses import CoSENTLoss
from torch import Tensor, nn, tensor


def pairwise_tanimoto_similarity(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the Tanimoto similarity between two numpy arrays x and y.

    Defined in 10.1186 (Tanimoto coefficient) as:
    T(x,y) = <x,y> / (x^2 + y^2 - <x,y>)

    References
    ----------
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2
    https://arxiv.org/pdf/2302.05666.pdf

    Parameters
    ----------
    x : Tensor
        A tensor of shape (n_samples, n_features)
    y : Tensor
        A tensor of shape (n_samples, n_features)

    Returns
    -------
    similarity : Tensor
        A tensor of shape (n_samples, n_samples)
        The Tanimoto similarity between x and y.
    """
    if not isinstance(x, Tensor):
        x = tensor(x)

    if not isinstance(y, Tensor):
        y = tensor(y)

    dot_product = util.pairwise_dot_score(x, y)
    denominator = x.pow(2).sum(dim=-1) + y.pow(2).sum(dim=-1) - dot_product

    return dot_product / denominator.clamp(min=1e-9)


class TanimotoLoss(CoSENTLoss):
    def __init__(
        self,
        model: sentence_transformers.SentenceTransformer,
        scale: float = 20.0,
    ):
        """
        This class implements a variation of CoSENTLoss where instead of incorporating cosine similarity it uses tanimoto similarity.

        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(k,l)-s(i,j))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition.

        Parameters
        ----------
        model : SentenceTransformer
            The sentence transformer model used to generate embeddings
        scale : float, optional
            Scaling factor (inverse temperature) applied to similarity scores, defaults to 20.0

        Requirements:
            - SMILES pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Relations:
            - Extends CoSENTLoss by replacing pairwise_cos_sim with pairwise_tanimoto_similarity

        Inputs:
            +---------------------------+------------------------+
            | Texts                     | Labels                 |
            +===========================+========================+
            | (smiles_A, smiles2) pairs | float similarity score |
            +---------------------------+------------------------+
        """  # noqa: E501
        super().__init__(
            model,
            scale,
            similarity_fct=pairwise_tanimoto_similarity,
        )


class TanimotoSimilarityLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module = nn.MSELoss(),
    ):
        """
        This class implements a loss function that measures the difference between predicted Tanimoto similarities
        of smiles embeddings and their expected similarity scores. It uses a SentenceTransformer model to generate
        embeddings and computes pairwise Tanimoto similarities between them.

        Parameters
        ----------
        model : SentenceTransformer
            The sentence transformer model used to generate embeddings
        loss : nn.Module, optional
            The base loss function to compute the final loss value, defaults to nn.MSELoss()

        Inputs:
            +---------------------------+------------------------+
            | Texts                     | Labels                 |
            +===========================+========================+
            | (smiles_A, smiles2) pairs | float similarity score |
            +---------------------------+------------------------+
        """
        super().__init__()
        self.model = model
        self.loss_fct = loss
        self.similarity_fct = pairwise_tanimoto_similarity

    def forward(self, smiles_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings: list[Tensor] = [
            self.model(smiles_feature)["sentence_embedding"]
            for smiles_feature in smiles_features
        ]

        # if isinstance(self.loss, nn.CosineEmbeddingLoss):
        #     loss = self.loss(embeddings[0], embeddings[1], labels)
        #     return loss

        similarities = self.similarity_fct(embeddings[0], embeddings[1])
        loss = self.loss_fct(similarities, labels.view(-1))
        return loss

    def get_config_dict(self):
        return {
            "loss_fct": type(self.loss_fct).__name__,
            "similarity_fct": self.similarity_fct.__name__,
        }
