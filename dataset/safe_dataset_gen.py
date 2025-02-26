import gc
import os

import pandas as pd
from pandarallel import pandarallel
from util import init_logging, parallel_safe_encode

from chem_mrl.constants import OUTPUT_DATA_DIR

logger = init_logging(__name__)
pandarallel.initialize(progress_bar=True)


def get_files_to_process():
    processed_smiles_dir = os.path.join(OUTPUT_DATA_DIR, "processed")
    processed_safe_dir = os.path.join(OUTPUT_DATA_DIR, "safe")
    dataset_subset_substrings = [
        "pubchem_dataset_less_than_128_tokens",
        "druglike_QED_36M_less_than_128_tokens",
        "full_ds_less_than_128_tokens",
        "druglike_QED-Pfizer_13M_less_than_128_tokens",
    ]

    input_files = os.listdir(processed_smiles_dir)
    safe_ds_files = [file for file in input_files if "safe" in file]
    completed_files = os.listdir(processed_safe_dir) + safe_ds_files
    files_to_gen = [
        file
        for file in input_files
        if any(substring in file for substring in dataset_subset_substrings)
        and not any(substring in file for substring in completed_files)
    ]
    files_to_gen.sort(reverse=True)
    return files_to_gen, processed_smiles_dir, processed_safe_dir


def main():
    files_to_gen, processed_smiles_dir, processed_safe_dir = get_files_to_process()
    for file in files_to_gen:
        in_path = os.path.join(processed_smiles_dir, file)
        out_path = os.path.join(processed_safe_dir, file)
        logger.info(f"Processing {file}")

        df = pd.read_parquet(in_path, columns=["smiles"])
        df.drop_duplicates(keep="first", inplace=True, ignore_index=True)
        df["safe"] = df["smiles"].parallel_apply(parallel_safe_encode)
        df.dropna(subset=["safe"], inplace=True, ignore_index=True)
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
        # free memory to prevent WSL OOM error
        del df
        gc.collect()


if __name__ == "__main__":
    main()
