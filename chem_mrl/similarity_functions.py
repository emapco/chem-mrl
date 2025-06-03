from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Literal

import numpy as np
from numpy import ndarray
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import (
    cos_sim,
    dot_score,
    euclidean_sim,
    manhattan_sim,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)
from sklearn.metrics.pairwise import (
    check_paired_arrays,
    row_norms,
)
from sklearn.preprocessing import normalize
from torch import Tensor


def patch_sentence_transformer():
    class SentenceTransformerOverride(SentenceTransformer):
        @property
        def similarity_fn_name(
            self,
        ) -> Literal["tanimoto", "cosine", "dot", "euclidean", "manhattan"]:
            """Return the name of the similarity function used by :meth:`SentenceTransformer.similarity`
            and :meth:`SentenceTransformer.similarity_pairwise`.

            Returns:
                Optional[str]: The name of the similarity function.
                Can be None if not set, in which case it will default to "cosine" when first called.

            Example:
                >>> model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
                >>> model.similarity_fn_name
                'dot'
            """  # noqa: E501
            if self._similarity_fn_name is None:
                self.similarity_fn_name = SimilarityFunction.COSINE
            return self._similarity_fn_name

        @similarity_fn_name.setter
        def similarity_fn_name(
            self,
            value: Literal["cosine", "dot", "euclidean", "manhattan"] | SimilarityFunction,
        ) -> None:
            if isinstance(value, SimilarityFunction):
                value = value.value
            self._similarity_fn_name = value

            if value is not None:
                self._similarity = SimilarityFunction.to_similarity_fn(value)
                self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(value)

    new_prop = SentenceTransformerOverride.__dict__["similarity_fn_name"]

    SentenceTransformer.similarity_fn_name = property(
        fget=new_prop.fget,
        fset=new_prop.fset,
        fdel=new_prop.fdel,
        doc=new_prop.__doc__,
    )


# Used by loss classes
def pairwise_tanimoto_similarity(a: list | ndarray | Tensor, b: list | ndarray | Tensor) -> Tensor:
    """
    Computes the Tanimoto similarity between two numpy arrays x and y.

    Tanimoto coefficient as defined in 10.1186/s13321-015-0069-3 for continuous variables:
    T(X,Y) = <X,Y> / (Σx^2 + Σy^2 - <X,Y>)

    References
    ----------
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2
    https://arxiv.org/pdf/2302.05666.pdf - Other intersection over union (IoU) metrics

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
    a = util._convert_to_tensor(a)
    b = util._convert_to_tensor(b)
    dot_product = util.pairwise_dot_score(a, b)
    denominator = a.pow(2).sum(dim=-1) + b.pow(2).sum(dim=-1) - dot_product

    return dot_product / denominator.clamp(min=1e-9)


def tanimoto_similarity(a: list | ndarray | Tensor, b: list | ndarray | Tensor) -> Tensor:
    """
    Computes the Tanimoto similarity between two numpy arrays x and y.

    Tanimoto coefficient as defined in 10.1186/s13321-015-0069-3 for continuous variables:
    T(X,Y) = <X,Y> / (Σx^2 + Σy^2 - <X,Y>)

    References
    ----------
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2
    https://arxiv.org/pdf/2302.05666.pdf - Other intersection over union (IoU) metrics

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
    a = util._convert_to_batch_tensor(a)
    b = util._convert_to_batch_tensor(b)
    dot_product = util.dot_score(a, b)
    denominator = a.pow(2).sum(dim=-1) + b.pow(2).sum(dim=-1) - dot_product

    return dot_product / denominator.clamp(min=1e-9)


# Used by evaluation classes
def paired_cosine_distances(X, Y):
    """
    Compute the paired cosine distances between X and Y.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples, n_features)
        An array where each row is a sample and each column is a feature.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`, where `distances[i]` is the
        distance between `X[i]` and `Y[i]`.

    Notes
    -----
    The cosine distance is equivalent to the half the squared
    euclidean distance if each sample is normalized to unit norm.
    """
    X, Y = check_paired_arrays(X, Y)
    X = normalize(X).astype(np.float32, copy=False)
    Y = normalize(Y).astype(np.float32, copy=False)
    return (0.5 * row_norms(X - Y, squared=True)).astype(np.float32)


def paired_tanimoto_similarity(X, Y):
    """
    Compute the paired Tanimoto similarity between X and Y.

    Tanimoto coefficient as defined in 10.1186/s13321-015-0069-3 for continuous variables:
    T(X,Y) = <X,Y> / (Σx^2 + Σy^2 - <X,Y>)

    References
    ----------
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3/tables/2
    https://arxiv.org/pdf/2302.05666.pdf - Other intersection over union (IoU) metrics

    Parameters
    ----------
    X : {array-like} of shape (n_samples, n_features)
        First array of samples
    Y : {array-like} of shape (n_samples, n_features)
        Second array of samples

    Returns
    -------
    similarity : ndarray of shape (n_samples,)
        Tanimoto similarity between paired rows of X and Y
    """
    X, Y = check_paired_arrays(X, Y)
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)
    dot_product = np.sum(X * Y, axis=1)
    np.multiply(X, X, out=X)  # X is now X²
    np.multiply(Y, Y, out=Y)  # Y is now Y²
    X = np.sum(X, axis=1)
    Y = np.sum(Y, axis=1)
    denominator = X + Y - dot_product
    return (dot_product / np.maximum(denominator, 1e-9)).astype(np.float32)


class SimilarityFunction(Enum):
    """
    Enum class for supported similarity functions. The following functions are supported:

    - ``SimilarityFunction.TANIMOTO`` (``"tanimoto"``): Tanimoto similarity
    - ``SimilarityFunction.COSINE`` (``"cosine"``): Cosine similarity
    - ``SimilarityFunction.DOT_PRODUCT`` (``"dot"``, ``dot_product``): Dot product similarity
    - ``SimilarityFunction.EUCLIDEAN`` (``"euclidean"``): Euclidean distance
    - ``SimilarityFunction.MANHATTAN`` (``"manhattan"``): Manhattan distance
    """

    TANIMOTO = "tanimoto"
    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    DOT = "dot"  # Alias for DOT_PRODUCT
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

    @staticmethod
    def to_similarity_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[[Tensor | ndarray, Tensor | ndarray], Tensor]:
        """
        Converts a similarity function name or enum value to the corresponding similarity function.

        Args:
            similarity_function (Union[str, SimilarityFunction]): The name or enum value of the similarity function.

        Returns:
            Callable[[Union[Tensor, ndarray], Union[Tensor, ndarray]], Tensor]: The corresponding similarity function.

        Raises:
            ValueError: If the provided function is not supported.

        Example:
            >>> similarity_fn = SimilarityFunction.to_similarity_fn("cosine")
            >>> similarity_scores = similarity_fn(embeddings1, embeddings2)
            >>> similarity_scores
            tensor([[0.3952, 0.0554],
                    [0.0992, 0.1570]])
        """  # noqa: E501
        similarity_function = SimilarityFunction(similarity_function)

        if similarity_function == SimilarityFunction.COSINE:
            return cos_sim
        if similarity_function == SimilarityFunction.DOT_PRODUCT:
            return dot_score
        if similarity_function == SimilarityFunction.MANHATTAN:
            return manhattan_sim
        if similarity_function == SimilarityFunction.EUCLIDEAN:
            return euclidean_sim
        if similarity_function == SimilarityFunction.TANIMOTO:
            return tanimoto_similarity

        raise ValueError(
            f"The provided function {similarity_function} is not supported. "
            f"Use one of the supported values: {SimilarityFunction.possible_values()}."
        )

    @staticmethod
    def to_similarity_pairwise_fn(
        similarity_function: str | SimilarityFunction,
    ) -> Callable[[Tensor | ndarray, Tensor | ndarray], Tensor]:
        """
        Converts a similarity function into a pairwise similarity function.

        The pairwise similarity function returns the diagonal vector from the similarity matrix, i.e. it only
        computes the similarity(a[i], b[i]) for each i in the range of the input tensors, rather than
        computing the similarity between all pairs of a and b.

        Args:
            similarity_function (Union[str, SimilarityFunction]): The name or enum value of the similarity function.

        Returns:
            Callable[[Union[Tensor, ndarray], Union[Tensor, ndarray]], Tensor]: The pairwise similarity function.

        Raises:
            ValueError: If the provided similarity function is not supported.

        Example:
            >>> pairwise_fn = SimilarityFunction.to_similarity_pairwise_fn("cosine")
            >>> similarity_scores = pairwise_fn(embeddings1, embeddings2)
            >>> similarity_scores
            tensor([0.3952, 0.1570])
        """  # noqa: E501
        similarity_function = SimilarityFunction(similarity_function)

        if similarity_function == SimilarityFunction.COSINE:
            return pairwise_cos_sim  # type: ignore
        if similarity_function == SimilarityFunction.DOT_PRODUCT:
            return pairwise_dot_score  # type: ignore
        if similarity_function == SimilarityFunction.MANHATTAN:
            return pairwise_manhattan_sim
        if similarity_function == SimilarityFunction.EUCLIDEAN:
            return pairwise_euclidean_sim
        if similarity_function == SimilarityFunction.TANIMOTO:
            return pairwise_tanimoto_similarity

        raise ValueError(
            f"The provided function {similarity_function} is not supported. "
            f"Use one of the supported values: {SimilarityFunction.possible_values()}."
        )

    @staticmethod
    def possible_values() -> list[str]:
        """
        Returns a list of possible values for the SimilarityFunction enum.

        Returns:
            list: A list of possible values for the SimilarityFunction enum.

        Example:
            >>> possible_values = SimilarityFunction.possible_values()
            >>> possible_values
            ['cosine', 'dot', 'euclidean', 'manhattan']
        """
        return [m.value for m in SimilarityFunction]
