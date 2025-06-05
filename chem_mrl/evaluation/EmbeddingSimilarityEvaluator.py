import logging
import os
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Literal

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics.pairwise import (
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from chem_mrl.schemas.Enums import ChemMrlEvalMetricOption, EvalSimilarityFctOption
from chem_mrl.similarity_functions import (
    SimilarityFunction,
    paired_cosine_distances,
    paired_tanimoto_similarity,
)

from .utils import _write_results_to_csv

logger = logging.getLogger(__name__)


class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating
    the Spearman and Pearson rank correlation in comparison to the gold standard labels.
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
        main_similarity: EvalSimilarityFctOption = EvalSimilarityFctOption.tanimoto,
        metric: ChemMrlEvalMetricOption = ChemMrlEvalMetricOption.spearman,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] | None = None,
        truncate_dim: int | None = None,
    ):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between a pair of SMILES.

        :param smiles1:  List with the first SMILES in a pair
        :param smiles2: List with the second SMILES in a pair
        :param scores: Similarity score between smiles[i] and smiles[i]
        :param write_csv: Write results to a CSV file
        :param precision: The precision to use for the embeddings.
            Can be "float32", "int8", "uint8", "binary", or "ubinary". Defaults to None.
        :param truncate_dim: The dimension to truncate SMILES embeddings to.
            `None` uses the model's current truncation dimension. Defaults to None.
        """
        if precision is None:
            precision = "float32"

        self.smiles1 = smiles1
        self.smiles2 = smiles2
        self.labels = scores
        self.write_csv = write_csv
        self.precision = precision
        self.truncate_dim = truncate_dim

        assert len(self.smiles1) == len(self.smiles2)  # type: ignore
        assert len(self.smiles1) == len(self.labels)  # type: ignore

        self.metric = metric
        self.main_similarity = SimilarityFunction(main_similarity)
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is False:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = os.path.join(
            (
                "similarity_evaluation"
                + ("_" + name if name else "")
                + ("_" + precision if precision else "")
                + "_results.csv"
            ),
        )
        self.csv_headers = ["epoch", "steps", metric]

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = ".",
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
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
            "EmbeddingSimilarityEvaluator: "
            f"Evaluating the model on the {self.name} dataset {out_txt}:"
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
                precision=self.precision,  # type: ignore
                normalize_embeddings=bool(self.precision),
            )
            logger.info("Encoding smiles 2 validation data.")
            embeddings2 = model.encode(
                self.smiles2,  # type: ignore
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,  # type: ignore
                normalize_embeddings=bool(self.precision),
            )

        # Binary and ubinary embeddings are packed,
        # so we need to unpack them for the distance metrics
        if self.precision == "binary":
            embeddings1 = (embeddings1 + 128).astype(np.uint8)
            embeddings2 = (embeddings2 + 128).astype(np.uint8)
        if self.precision in ("ubinary", "binary"):
            embeddings1 = np.unpackbits(embeddings1, axis=1)
            embeddings2 = np.unpackbits(embeddings2, axis=1)

        similarity_functions = {
            "tanimoto": lambda x, y: paired_tanimoto_similarity(x, y),
            "cosine": lambda x, y: 1 - paired_cosine_distances(x, y),
            "manhattan": lambda x, y: -paired_manhattan_distances(x, y),
            "euclidean": lambda x, y: -paired_euclidean_distances(x, y),
            "dot": lambda x, y: [np.dot(emb1, emb2) for emb1, emb2 in zip(x, y, strict=False)],
        }
        main_similarity_scores = similarity_functions[self.main_similarity.value](
            embeddings1, embeddings2
        )
        del embeddings1, embeddings2

        metric_functions = {
            ChemMrlEvalMetricOption.pearson: lambda x, y: pearsonr(x, y)[0],
            ChemMrlEvalMetricOption.spearman: lambda x, y: spearmanr(x, y)[0],
        }
        eval_metric: float = metric_functions[self.metric](self.labels, main_similarity_scores)
        logger.info(
            f"{self.main_similarity.value.capitalize()}-Similarity :"
            f"\t{self.metric.capitalize()}: {eval_metric:.5f}\n"
        )
        del main_similarity_scores

        _write_results_to_csv(
            self.write_csv,
            self.csv_file,
            self.csv_headers,
            output_path,
            results=[
                epoch,
                steps,
                eval_metric,
            ],
        )

        self.primary_metric = self.metric
        metrics = self.prefix_name_to_metrics({self.metric: eval_metric}, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    @property
    def description(self) -> str:
        return "Semantic Similarity"

    def get_config_dict(self):
        config_dict = {}
        config_dict_candidate_keys = ["truncate_dim", "precision"]
        for key in config_dict_candidate_keys:
            if getattr(self, key) is not None:
                config_dict[key] = getattr(self, key)
        return config_dict
