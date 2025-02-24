import logging
import os
from time import perf_counter

import pandas as pd
from sqlalchemy import create_engine, text

from chem_mrl.constants import (
    BASE_MODEL_HIDDEN_DIM,
    BASE_MODEL_NAME,
    CHEM_MRL_DIMENSIONS,
    TEST_FP_SIZES,
)
from chem_mrl.molecular_embedder import ChemMRL
from chem_mrl.molecular_fingerprinter import MorganFingerprinter
from chem_mrl.util import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class PgVectorBenchmark:
    def __init__(
        self,
        psql_connect_uri: str,
        output_path: str,
        knn_k: int = 50,
        use_functional_fp: bool = False,
    ):
        self.knn_k = knn_k
        self.engine = create_engine(psql_connect_uri)
        self.output_path = output_path
        self.truth_dim = max(TEST_FP_SIZES)
        self.use_functional_fp = use_functional_fp

    def execute_knn_query(
        self,
        table_name: str,
        query_embedding: list[float],
        embedding_dim: int,
    ) -> tuple[list[str], float]:
        """
        Execute a KNN query using halfvec embeddings in a PostgreSQL table.

        Args:
            engine: SQLAlchemy engine instance.
            table_name: Name of the table containing embeddings.
            query_embedding: Query embedding as a list of floats.
            embedding_dim: Dimension of the embeddings.
            k: Number of nearest neighbors to retrieve.

        Returns:
            A tuple containing a list of (id, similarity) results and query execution time.
        """

        query = f"""
        WITH query_embedding AS (
            SELECT :query_embedding AS query
        )
        SELECT zinc_id
        FROM {table_name}
        ORDER BY embedding <=> (SELECT query::halfvec(:dim) FROM query_embedding)
        LIMIT :k
        """

        start_time = perf_counter()
        with self.engine.connect() as conn:
            result = conn.execute(
                text(query),
                {
                    "query_embedding": query_embedding,
                    "dim": embedding_dim,
                    "k": self.knn_k,
                },
            )
            results = [row[0] for row in result]
        query_duration = perf_counter() - start_time

        return results, query_duration

    def calculate_accuracy(self, ground_truth: list[str], predicted: list[str]) -> float:
        """Calculate accuracy (common entries in top-k)"""
        common = set(ground_truth) & set(predicted)
        accuracy = len(common) / len(ground_truth)
        return accuracy

    def test_morgan_fingerprints(
        self,
        morgan_fp: MorganFingerprinter,
        smiles: str,
        ground_truth: list[str],
        dim: int,
    ):
        if self.use_functional_fp:
            morgan_embedding = morgan_fp.get_functional_fingerprint_numpy(smiles)
        else:
            morgan_embedding = morgan_fp.get_fingerprint_numpy(smiles)
        if isinstance(morgan_embedding, float):
            return None

        morgan_results, morgan_time = self.execute_knn_query(
            f"test_{dim}",
            morgan_embedding.tolist(),  # type: ignore
            dim,
        )
        accuracy = self.calculate_accuracy(ground_truth, morgan_results)

        return {
            "model": "morgan",
            "dimension": dim,
            "accuracy": accuracy,
            "query_duration": morgan_time,
        }

    def test_transformer_embeddings(
        self,
        smiles_embedder: ChemMRL,
        table_name: str,
        model_name: str,
        smiles: str,
        ground_truth: list[str],
        dim: int,
    ):
        transformer_embedding = smiles_embedder.get_embedding(smiles)
        transformer_results, transformer_time = self.execute_knn_query(
            table_name,
            transformer_embedding.tolist(),  # type: ignore
            dim,
        )
        accuracy = self.calculate_accuracy(ground_truth, transformer_results)

        return {
            "model": model_name,
            "dimension": dim,
            "accuracy": accuracy,
            "query_duration": transformer_time,
        }

    def generate_ground_truth_result(
        self,
        smiles: str,
        ground_truth_fp: MorganFingerprinter,
    ) -> list[str] | float:
        """
        Generate ground truth data for testing."""
        if self.use_functional_fp:
            ground_truth_embedding = ground_truth_fp.get_functional_fingerprint_numpy(smiles)
        else:
            ground_truth_embedding = ground_truth_fp.get_fingerprint_numpy(smiles)

        if isinstance(ground_truth_embedding, float):
            return ground_truth_embedding

        ground_truth_results, _ = self.execute_knn_query(
            f"test_{self.truth_dim}",
            ground_truth_embedding.tolist(),  # type: ignore
            self.truth_dim,
        )
        return ground_truth_results

    def run_benchmark(
        self,
        test_queries: pd.DataFrame,
        model_name: str,
        chem_mrl_dimensions: list[int] = CHEM_MRL_DIMENSIONS,
        base_model_name: str = BASE_MODEL_NAME,
        base_model_hidden_dim: int = BASE_MODEL_HIDDEN_DIM,
        smiles_column_name: str = "smiles",
    ):
        logger.info("Starting benchmark...")
        results_data = []

        # compute the ground_truth for all rows first
        ground_truth_fp = MorganFingerprinter(radius=2, fp_size=self.truth_dim)
        ground_truth_queries = test_queries.copy()
        logger.info("Generating ground truth results")
        ground_truth_queries["ground_truth"] = ground_truth_queries[smiles_column_name].apply(
            self.generate_ground_truth_result, ground_truth_fp=ground_truth_fp
        )
        ground_truth_queries = ground_truth_queries.dropna(
            subset=["ground_truth"], ignore_index=True
        )
        logger.info(f"Total rows: {len(ground_truth_queries)}")

        for dim in chem_mrl_dimensions:
            logger.info(f"Processing dimension {dim}")

            morgan_fp = MorganFingerprinter(radius=2, fp_size=dim)
            mrl_embedder = ChemMRL(model_name=model_name, embedding_size=dim)
            base_embedder = None
            if dim == base_model_hidden_dim:
                base_embedder = ChemMRL(model_name=base_model_name, embedding_size=dim)

            for idx, row in ground_truth_queries.iterrows():
                if idx % 100 == 0:  # type: ignore
                    logger.info(f"Processing query {idx + 1}/{len(ground_truth_queries)}")  # type: ignore

                results_data.append(
                    self.test_morgan_fingerprints(
                        morgan_fp=morgan_fp,
                        smiles=row[smiles_column_name],
                        ground_truth=row["ground_truth"],
                        dim=dim,
                    )
                )
                results_data.append(
                    self.test_transformer_embeddings(
                        smiles_embedder=mrl_embedder,
                        table_name=f"cme_{dim}",
                        model_name="chem-mrl",
                        smiles=row[smiles_column_name],
                        ground_truth=row["ground_truth"],
                        dim=dim,
                    )
                )
                if dim == BASE_MODEL_HIDDEN_DIM:
                    assert base_embedder is not None
                    results_data.append(
                        self.test_transformer_embeddings(
                            smiles_embedder=base_embedder,
                            table_name=f"base_{dim}",
                            model_name=base_model_name,
                            smiles=row[smiles_column_name],
                            ground_truth=row["ground_truth"],
                            dim=dim,
                        )
                    )

        results_df = pd.DataFrame(results_data)
        summary_stats = results_df.groupby(["model", "dimension"]).describe()

        results_df.to_csv(
            os.path.join(self.output_path, "benchmark_detailed_results.csv"),
            index=False,
        )
        summary_stats.to_csv(os.path.join(self.output_path, "benchmark_summary_stats.csv"))
        return results_df, summary_stats
