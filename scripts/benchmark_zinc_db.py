import os
from argparse import ArgumentParser

import pandas as pd

from chem_mrl.benchmark import PgVectorBenchmark
from chem_mrl.constants import OUTPUT_DATA_DIR


def parse_args():
    parser = ArgumentParser(description="Parse arguments for ZINC20 DB benchmark.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(OUTPUT_DATA_DIR, "zinc20", "smiles_all_99.txt"),
        help="Path to a ZINC20 dataset file. Dataset files can be found here: https://files.docking.org/zinc20-ML/",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to the output directory where benchmark results will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling."
    )
    parser.add_argument(
        "--num_rows", type=int, default=500, help="Number of rows to sample."
    )
    parser.add_argument(
        "--psql_connection_uri",
        type=str,
        default="postgresql://postgres:password@127.0.0.1:5431/postgres",
        help="PostgreSQL connection URI string.",
    )
    parser.add_argument(
        "--knn_k", type=int, default=50, help="Number of neighbors for k-NN search."
    )

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()

    test_queries = pd.read_csv(ARGS.dataset_path, sep=" ", header=None)
    test_queries = test_queries.sample(ARGS.num_rows, random_state=ARGS.seed)
    test_queries.columns = ["smiles", "zinc_id"]
    test_queries = test_queries.drop(columns=["zinc_id"])

    benchmarker = PgVectorBenchmark(
        psql_connect_uri=ARGS.psql_connection_uri,
        output_path=ARGS.output_path,
        knn_k=ARGS.knn_k,
    )
    detailed_results, summary_stats = benchmarker.run_benchmark(
        model_name="chem_mrl", test_queries=test_queries, smiles_column_name="smiles"
    )

    header = "Benchmark Results Summary:"
    print(f"\n{header}")
    print("=" * len(header))
    print(summary_stats)
