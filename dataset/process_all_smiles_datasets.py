import logging
import os
import sys

# import pandas as cudf  # cpu
import cudf  # gpu
from pandarallel import pandarallel
from pandas import DataFrame
from rdkit import RDLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
RDLogger.DisableLog("rdApp.*")  # type: ignore - DisableLog is an exported function


def get_canonical_smiles(smiles: str) -> str | None:
    from rdkit.Chem import MolFromSmiles, MolToSmiles
    from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles

    mol = MolFromSmiles(smiles)
    if mol is None:
        try:
            mol = MolFromSmiles(StandardizeSmiles(smiles))  # may raise error
            if mol is None:
                return None
        except Exception:
            return None
    smiles = MolToSmiles(mol, canonical=True)
    return smiles


def load_parquet_file(file_path: str) -> cudf.DataFrame:
    df = cudf.read_parquet(file_path)

    if any(col.endswith(" ") for col in df.columns):
        column_mapping = {col: col.rstrip() for col in df.columns if col.endswith(" ")}
        df.rename(columns=column_mapping, inplace=True)

    # The zinc20 dataset file is too large so only sample a portion of the dataset
    if "zinc20" in file_path:
        df = df.sample(n=1400000)

    if "URL" in df.columns:
        df.rename(columns={"URL": "url"}, inplace=True)

    for col in ["source", "url"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def get_valid_files(output_dir: str) -> list[str]:
    excluded_keywords = [
        "zinc20",  # too large
        "fp_sim",  # fingerprint datasets
        "druglike_QED-Pfizer_13M",  # use full QED_36M instead
        "full_ds",  # final combined smiles dataset
        "canonical",  # already processed
    ]
    files = [f for f in os.listdir(output_dir) if f.endswith(".parquet")]
    files.sort()
    return [f for f in files if not any(keyword in f for keyword in excluded_keywords)]


def clean_dataframe(df: cudf.DataFrame) -> DataFrame:
    df.drop_duplicates(subset=["smiles"], keep="first", inplace=True, ignore_index=True)

    # Clean invalid values
    df.loc[df["inchi"].notnull() & (df["inchi"].str.startswith("InChI=") is False), "inchi"] = None

    for col in ["name", "formula", "smiles"]:
        df.loc[df[col] == "N/A", col] = None

    df["source"] = df["source"].astype("category")
    df["url"] = df["url"].astype("category")

    # Remove invalid rows and columns that are no longer needed
    df.drop(columns=["name", "formula", "inchi", "url"], inplace=True)
    df.dropna(subset=["smiles"], inplace=True)

    # Sort dataframe based on smiles string length for postprocessing
    cutoff = 3
    df["smiles_length"] = df["smiles"].str.len()
    df = df[df["smiles"].str.len() > cutoff].sort_values(
        "smiles_length", ascending=True, ignore_index=True
    )
    df.drop(columns=["smiles_length"], inplace=True)

    # Convert to pandas dataframe if using cudf library
    # Need to apply function on strings - not currently supported by cudf
    if hasattr(df, "to_pandas"):
        if callable(df.to_pandas):
            pd_df = df.to_pandas()
    assert isinstance(pd_df, DataFrame)
    del df

    # Use pandarallel to parallelize get_canonical_smiles and then remove invalid rows
    pandarallel.initialize(progress_bar=True)
    logging.info("Canonicalizing SMILES strings")
    pd_df["canonical_smiles"] = pd_df["smiles"].parallel_apply(get_canonical_smiles)
    pd_df.dropna(subset=["canonical_smiles"], inplace=True)
    pd_df.drop_duplicates(
        subset=["canonical_smiles"], keep="first", inplace=True, ignore_index=True
    )
    return pd_df


def process_all_chemistry_datasets(output_dir: str) -> DataFrame:
    dfs = []
    valid_files = get_valid_files(output_dir)

    for file in valid_files:
        logging.info(f"Processing {file}")
        df = load_parquet_file(os.path.join(output_dir, file))
        dfs.append(df)
        logging.info(f"Loaded df size: {len(df)}")

    combined_df = cudf.concat(dfs, ignore_index=True)
    del dfs

    return clean_dataframe(combined_df)


def main():
    curr_file_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_file_dir)
    processed_output_dir = os.path.join(parent_dir, "data", "chem")
    preprocessed_output_dir = os.path.join(processed_output_dir, "preprocessed")

    df = process_all_chemistry_datasets(preprocessed_output_dir)
    logging.info(f"Final df size: {len(df)}")

    output_path = os.path.join(processed_output_dir, "full_ds_canonical.parquet")
    logging.info(f"Saving dataset to {output_path}")
    df.to_parquet(
        output_path,
        engine="fastparquet",
        compression="zstd",
        index=False,
    )


if __name__ == "__main__":
    main()
