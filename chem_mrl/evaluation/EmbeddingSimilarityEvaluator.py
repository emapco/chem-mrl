import logging
from contextlib import nullcontext
from enum import Enum
from typing import Iterable, Literal

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics.pairwise import check_paired_arrays, row_norms
from sklearn.preprocessing import normalize

from .utils import _write_results_to_csv

logger = logging.getLogger(__name__)


class SimilarityFunction(Enum):
    COSINE = 0
    TANIMOTO = 1


class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as tanimoto similarity.
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(
        self,
        smiles1: Iterable[str],
        smiles2: Iterable[str],
        scores: Iterable[float],
        batch_size: int = 16,
        main_similarity: SimilarityFunction = SimilarityFunction.TANIMOTO,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: (
            Literal["float32", "int8", "uint8", "binary", "ubinary"] | None
        ) = None,
        truncate_dim: int | None = None,
    ):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between a pair of SMILES.

        :param smiles1:  List with the first SMILES in a pair
        :param smiles2: List with the second SMILES in a pair
        :param scores: Similarity score between smiles[i] and smiles[i]
        :param write_csv: Write results to a CSV file
        :param precision: The precision to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or
            "ubinary". Defaults to None.
        :param truncate_dim: The dimension to truncate SMILES embeddings to. `None` uses the model's current
            truncation dimension. Defaults to None.
        """
        if precision is None:
            precision = "float32"

        self.smiles1 = smiles1
        self.smiles2 = smiles2
        self.labels = scores
        self.write_csv = write_csv
        self.precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = (
            precision
        )
        self.truncate_dim = truncate_dim

        assert len(self.smiles1) == len(self.smiles2)  # type: ignore
        assert len(self.smiles1) == len(self.labels)  # type: ignore

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = (
            "similarity_evaluation"
            + ("_" + name if name else "")
            + ("_" + precision if precision else "")
            + "_results.csv"
        )

        self.csv_headers = [
            "epoch",
            "steps",
            "pearson",
            "spearman",
        ]

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = ".",
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f"after epoch {epoch}"
            else:
                out_txt = f"in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f"(truncated to {self.truncate_dim})"

        logger.info(
            f"Custom EmbeddingSimilarityEvaluator: Evaluating the model on the {self.name} dataset {out_txt}:"
        )

        with (
            nullcontext()
            if self.truncate_dim is None
            else model.truncate_sentence_embeddings(self.truncate_dim)
        ):
            logger.info("Encoding smiles 1 validation data.")
            embeddings1 = model.encode(
                self.smiles1,  # type: ignore
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )
            logger.info("Encoding smiles 2 validation data.")
            embeddings2 = model.encode(
                self.smiles2,  # type: ignore
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )

        # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
        if self.precision == "binary":
            embeddings1 = (embeddings1 + 128).astype(np.uint8)
            embeddings2 = (embeddings2 + 128).astype(np.uint8)
        if self.precision in ("ubinary", "binary"):
            embeddings1 = np.unpackbits(embeddings1, axis=1)
            embeddings2 = np.unpackbits(embeddings2, axis=1)

        if self.main_similarity == SimilarityFunction.TANIMOTO:
            main_similarity_scores = paired_tanimoto_similarity(
                embeddings1, embeddings2
            )
            main_similarity_name = "Tanimoto-Similarity"
        else:
            main_similarity_scores = 1 - (
                paired_cosine_distances(embeddings1, embeddings2)
            )
            main_similarity_name = "Cosine-Similarity"

        del embeddings1, embeddings2

        eval_pearson, _ = pearsonr(self.labels, main_similarity_scores)
        eval_spearman, _ = spearmanr(self.labels, main_similarity_scores)
        del main_similarity_scores

        logger.info(
            "{} :\tPearson: {:.5f}\tSpearman: {:.5f}".format(
                main_similarity_name, eval_pearson, eval_spearman
            )
        )

        _write_results_to_csv(
            self.write_csv,
            self.csv_file,
            self.csv_headers,
            output_path,
            results=[
                epoch,
                steps,
                eval_pearson,
                eval_spearman,
            ],
        )

        eval_pearson = float(eval_spearman)  # type: ignore
        assert isinstance(eval_pearson, float)
        return eval_pearson


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
    return (0.5 * row_norms(X - Y, squared=True)).astype(np.float16)


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
    return (dot_product / np.maximum(denominator, 1e-9)).astype(np.float16)
