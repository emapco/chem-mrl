from contextlib import nullcontext
from enum import Enum
from typing import List, Literal
import logging
import os
import gc
import csv
import time

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    SentenceEvaluator,
)

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import check_paired_arrays, row_norms
from sklearn.preprocessing import normalize
import numpy as np


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
        sentences1: List[str],
        sentences2: List[str],
        scores: List[float],
        batch_size: int = 16,
        main_similarity: SimilarityFunction | None = None,
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

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        :param precision: The precision to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or
            "ubinary". Defaults to None.
        :param truncate_dim: The dimension to truncate sentence embeddings to. `None` uses the model's current
            truncation dimension. Defaults to None.
        """
        if precision is None:
            precision = "float32"

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = scores
        self.write_csv = write_csv
        self.precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = (
            precision
        )
        self.truncate_dim = truncate_dim

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)

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
            "cosine_pearson",
            "cosine_spearman",
        ]

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(
            f"Custom EmbeddingSimilarityEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:"
        )

        with (
            nullcontext()
            if self.truncate_dim is None
            else model.truncate_sentence_embeddings(self.truncate_dim)
        ):
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Encoding sentence 1 validation data.")
            embeddings1 = model.encode(
                self.sentences1,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )
            logger.info("Encoding sentence 2 validation data.")
            embeddings2 = model.encode(
                self.sentences2,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )
            torch.cuda.empty_cache()
            gc.collect()
        # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
        if self.precision == "binary":
            embeddings1 = (embeddings1 + 128).astype(np.uint8)
            embeddings2 = (embeddings2 + 128).astype(np.uint8)
        if self.precision in ("ubinary", "binary"):
            embeddings1 = np.unpackbits(embeddings1, axis=1)
            embeddings2 = np.unpackbits(embeddings2, axis=1)

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        del embeddings1, embeddings2
        gc.collect()
        time.sleep(15)

        eval_pearson_cosine, _ = pearsonr(self.labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(self.labels, cosine_scores)
        del cosine_scores
        gc.collect()

        logger.info(
            "Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                eval_pearson_cosine, eval_spearman_cosine
            )
        )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(
                csv_path,
                newline="",
                mode="a" if output_file_exists else "w",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                        eval_pearson_cosine,
                        eval_spearman_cosine,
                    ]
                )

        gc.collect()
        if self.main_similarity == SimilarityFunction.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity is None:
            return max(eval_spearman_cosine)
        else:
            raise ValueError("Unknown main_similarity value")


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

    Examples
    --------
    >>> from sklearn.metrics.pairwise import paired_cosine_distances
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> paired_cosine_distances(X, Y)
    array([0.5       , 0.18...])
    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        X, Y = check_paired_arrays(X, Y)
        X = X.astype(np.float16, copy=False)
        Y = Y.astype(np.float16, copy=False)

    if isinstance(X, np.ndarray):
        X = normalize(X).astype(np.float16, copy=False)
    if isinstance(X, np.ndarray):
        Y = normalize(Y).astype(np.float16, copy=False)

    return 0.5 * row_norms(X - Y, squared=True)
