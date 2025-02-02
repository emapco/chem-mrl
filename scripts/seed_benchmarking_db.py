import argparse
import logging
import os
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context

import pandas as pd
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from chem_mrl.constants import (
    BASE_MODEL_DIMENSIONS,
    BASE_MODEL_NAME,
    CHEM_MRL_DIMENSIONS,
    EMBEDDING_MODEL_HIDDEN_DIM,
    OUTPUT_DATA_DIR,
    TEST_FP_SIZES,
)
from chem_mrl.device_manager import CudaDeviceManager
from chem_mrl.molecular_embedder import ChemMRL
from chem_mrl.molecular_fingerprinter import MorganFingerprinter

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@dataclass
class DBSeederConfig:
    file_path: str
    total_rows: int
    fp_sizes: list[int]
    num_processes: int
    db_uri: str
    batch_size: int = 100_000
    embedder_batch_size: int = 2048
    embedding_col_name: str = "embedding"


class BenchmarkDataSeeder(ABC):
    """
    Abstract base class for seeding chemical embedding benchmark data into a PostgreSQL database.
    Provides common functionality for data loading and database connection management.
    Derived classes must implement generate() and seed() methods for specific embedding types.
    """

    def __init__(self, config: DBSeederConfig):
        self._config = config

    @property
    def config(self):
        return self._config

    @staticmethod
    def _get_pooled_engine(db_uri: str, pool_size: int = 2):
        return create_engine(
            db_uri,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=10,
            pool_pre_ping=True,
        )

    @staticmethod
    def _load_chemical_data(
        file_path: str, skip_rows: int | None = None, batch_size: int = 100_000
    ):
        df = pd.read_csv(
            file_path,
            sep=" ",
            header=None,
            nrows=batch_size,
            skiprows=skip_rows or 0,
        )
        df.columns = ["smiles", "zinc_id"]
        return df

    @abstractmethod
    def generate(self, fp_size: int, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def seed(self):
        raise NotImplementedError


class MorganFingerprintSeeder(BenchmarkDataSeeder):
    """Morgan fingerprint implementation of the BenchmarkDataSeeder class.
    Handles generation and seeding of Morgan fingerprint embeddings
    into PostgreSQL database tables. Uses CPU-based parallel processing
    for efficient batch processing of chemical SMILES data."""

    @staticmethod
    def generate(config: DBSeederConfig, fp_size: int):
        logging.info(f"Generating morgan fingerprints of dimension {fp_size}")
        engine = BenchmarkDataSeeder._get_pooled_engine(config.db_uri, pool_size=2)
        try:
            fingerprinter = MorganFingerprinter(fp_size=fp_size)
            for offset in range(0, config.total_rows, config.batch_size):
                test_df = BenchmarkDataSeeder._load_chemical_data(
                    config.file_path, skip_rows=offset, batch_size=config.batch_size
                )
                test_df[config.embedding_col_name] = test_df["smiles"].apply(
                    fingerprinter.get_functional_fingerprint_numpy  # type: ignore
                )
                test_df.dropna(
                    subset=[config.embedding_col_name],
                    inplace=True,
                    ignore_index=True,
                )
                test_df.to_sql(
                    f"test_{fp_size}",
                    engine,
                    if_exists="append",
                    index=False,
                    dtype={config.embedding_col_name: Vector(dim=fp_size)},  # type: ignore
                )
        finally:
            engine.dispose()
        logging.info(f"Finished generating fingerprints of dimension {fp_size}")

    def seed(self):
        with ProcessPoolExecutor(
            max_workers=self._config.num_processes,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            futures = [
                executor.submit(self.generate, self._config, fp_size)
                for fp_size in self._config.fp_sizes
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    logging.error(f"An error occurred: {str(e)}")


class TransformerEmbeddingSeeder(BenchmarkDataSeeder):
    """Transformer-based embedding implementation of the BenchmarkDataSeeder class.
    Manages generation and seeding of transformer model embeddings
    into PostgreSQL database tables.

    Handles GPU device allocation and parallel
    processing for efficient batch processing of chemical SMILES data.
    Falls back to CPU processing if GPU device allocation fails.
    """

    def __init__(self, config: DBSeederConfig, model_name: str):
        super().__init__(config)
        self._model_name = model_name
        self._device_manager = CudaDeviceManager(
            max_cpu_processes_fallback=config.num_processes
        )

    @staticmethod
    def generate(config: DBSeederConfig, model_name: str, fp_size: int, device: str):
        logging.info(f"Generating performance embeddings of dimension {fp_size}")
        engine = BenchmarkDataSeeder._get_pooled_engine(config.db_uri, pool_size=2)
        try:
            embedder = ChemMRL(
                model_name=model_name,
                fp_size=fp_size,
                use_half_precision=True,
                device=device,
                batch_size=config.embedder_batch_size,
                normalize_embeddings=fp_size < EMBEDDING_MODEL_HIDDEN_DIM,
            )
            for offset in range(0, config.total_rows, config.batch_size):
                test_df = BenchmarkDataSeeder._load_chemical_data(
                    config.file_path, skip_rows=offset, batch_size=config.batch_size
                )
                smiles_embeddings = embedder.get_embeddings(test_df["smiles"])
                test_df[config.embedding_col_name] = list(smiles_embeddings)
                del smiles_embeddings

                table_name = (
                    f"base_{fp_size}"
                    if model_name == BASE_MODEL_NAME
                    else f"cme_{fp_size}"
                )

                test_df.to_sql(
                    table_name,
                    engine,
                    if_exists="append",
                    index=False,
                    dtype={
                        config.embedding_col_name: Vector(dim=fp_size)  # type: ignore
                    },
                )
            del embedder
        finally:
            engine.dispose()
            return device

    def seed(self):
        max_concurrent = self._device_manager.num_processes
        with ProcessPoolExecutor(
            max_workers=max_concurrent,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            running_futures = {}
            initial_tasks, remaining_tasks = (
                self._config.fp_sizes[:max_concurrent],
                self._config.fp_sizes[max_concurrent:],
            )

            for fp_size in initial_tasks:
                device = self._device_manager.get_device()
                future = executor.submit(
                    self.generate, self._config, self._model_name, fp_size, device
                )
                running_futures[future] = fp_size

            while running_futures:
                for future in as_completed(running_futures):
                    fp_size = running_futures.pop(future)
                    try:
                        device = future.result()
                        self._device_manager.release_device(device)

                        if remaining_tasks:
                            next_fp_size = remaining_tasks.pop(0)
                            next_device = self._device_manager.get_device()
                            next_future = executor.submit(
                                self.generate,
                                self._config,
                                self._model_name,
                                next_fp_size,
                                next_device,
                            )
                            running_futures[next_future] = next_fp_size

                    except Exception as e:
                        traceback.print_exc()
                        logging.error(
                            f"An error occurred with fp_size {fp_size}: {str(e)}"
                        )


def parse_args(mode_choice: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed embeddings or fingerprints tables."
    )
    parser.add_argument(
        "--mode",
        choices=mode_choice,
        required=True,
        help="Specify whether to seed test fingerprint, chem_mrl embedding, or base embedding table(s).",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=os.path.join(OUTPUT_DATA_DIR, "zinc20", "smiles_all_00.txt"),
        help="Path to the input file. Format: two-column header-less space-delimited file. "
        "First column is SMILES, second column is a unique identifier.",
    )
    parser.add_argument(
        "--total_rows",
        type=int,
        default=1_000_000,
        help="Total number of rows to process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100_000,
        help="Number of rows to process at a time. Note that each process will load this amount of rows at a time.",
    )
    parser.add_argument(
        "--embedder_batch_size",
        type=int,
        default=4096,
        help="Specify the number of embeddings to generate at a time using the embedding transformer model.",
    )
    parser.add_argument(
        "--num_cpu_processes",
        type=int,
        default=4,
        help="Number of CPU processes to use.",
    )
    parser.add_argument(
        "--postgres_uri",
        type=str,
        default="postgresql://postgres:password@127.0.0.1:5431/postgres",
        help="URI to the postgres database with pgvector>=0.7 extension.",
    )
    return parser.parse_args()


def main():
    modes = ["test", "chem_mrl", "base"]
    args = parse_args(modes)
    config_map = {
        "test": (TEST_FP_SIZES, MorganFingerprintSeeder),
        "chem_mrl": (
            CHEM_MRL_DIMENSIONS,
            lambda c: TransformerEmbeddingSeeder(c, "chem_mrl"),
        ),
        "base": (
            BASE_MODEL_DIMENSIONS,
            lambda c: TransformerEmbeddingSeeder(c, BASE_MODEL_NAME),
        ),
    }

    fp_sizes, seeder_class = config_map[args.mode]

    config = DBSeederConfig(
        file_path=args.file_path,
        total_rows=args.total_rows,
        fp_sizes=fp_sizes,
        num_processes=args.num_cpu_processes,
        db_uri=args.postgres_uri,
        batch_size=args.batch_size,
        embedder_batch_size=args.embedder_batch_size,
    )

    seeder = seeder_class(config)
    assert isinstance(seeder, (MorganFingerprintSeeder, TransformerEmbeddingSeeder))

    seeder.seed()


if __name__ == "__main__":
    main()
