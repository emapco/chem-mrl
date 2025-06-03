from collections.abc import Callable, Iterable

import sentence_transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss
from torch import Tensor, nn

from chem_mrl.similarity_functions import pairwise_tanimoto_similarity


class TanimotoSentLoss(CoSENTLoss):
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
        loss: Callable = nn.MSELoss(),  # noqa: B008
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
        """  # noqa: E501
        super().__init__()
        self.__model = model
        self.__loss_fct = loss
        self.__similarity_fct = pairwise_tanimoto_similarity
        # strategy pattern - determine which forward to call at runtime
        if isinstance(self.__loss_fct, nn.CosineEmbeddingLoss):
            self.__forward = self._bypass_similarity_fct_forward
        else:
            self.__forward = self._compute_similarity_fct_forward

    def forward(self, smiles_features: Iterable[dict[str, Tensor]], labels: Tensor):
        embeddings: list[Tensor] = [
            self.__model(smiles_feature)["sentence_embedding"] for smiles_feature in smiles_features
        ]
        return self.__forward(embeddings, labels)

    def _compute_similarity_fct_forward(self, embeddings: list[Tensor], labels: Tensor):
        similarities = self.__similarity_fct(embeddings[0], embeddings[1])
        return self.__loss_fct(similarities, labels.view(-1))

    def _bypass_similarity_fct_forward(self, embeddings: list[Tensor], labels: Tensor):
        """nn.CosineEmbeddingLoss does not use similarity_fct, so we bypass it"""
        return self.__loss_fct(embeddings[0], embeddings[1], labels.view(-1))

    def get_config_dict(self):
        return {
            "loss_fct": type(self.__loss_fct).__name__,
            "similarity_fct": self.__similarity_fct.__name__,
        }
