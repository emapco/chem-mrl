import os
from typing import List

import cudf  # gpu

# import pandas as cudf   # cpu


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


def get_valid_files(output_dir: str) -> List[str]:
    excluded_keywords = [
        "zinc20",  # too large
        "fp_similarity",  # fingerprint datasets
        "druglike_QED-Pfizer_13M",  # use full QED_36M instead
        "isomer_design",  # used for classifier example
        "full_ds",  # final combined smiles dataset
    ]
    files = [f for f in os.listdir(output_dir) if f.endswith(".parquet")]
    files.sort()
    return [f for f in files if not any(keyword in f for keyword in excluded_keywords)]


def clean_dataframe(df: cudf.DataFrame) -> cudf.DataFrame:
    # Clean invalid values
    df.loc[
        df["inchi"].notnull() & (df["inchi"].str.startswith("InChI=") is False), "inchi"
    ] = None

    for col in ["name", "formula", "smiles"]:
        df.loc[df[col] == "N/A", col] = None

    df["source"] = df["source"].astype("category")
    df["url"] = df["url"].astype("category")

    # Process SMILES
    df = (
        df.dropna(subset=["smiles"])
        .drop_duplicates(subset=["smiles"], keep="first")
        .drop(columns=["name", "formula", "inchi", "url"])
        .reset_index(drop=True)
    )

    # Sort dataframe based on smiles string length for postprocessing
    df["smiles_length"] = df["smiles"].str.len()
    df = df[df["smiles"].str.len() > 3].sort_values(
        "smiles_length", ascending=True, ignore_index=True
    )
    df.drop(columns=["smiles_length"], inplace=True)
    return df


def process_all_chemistry_datasets(output_dir: str) -> cudf.DataFrame:
    dataframes = []
    valid_files = get_valid_files(output_dir)

    for file in valid_files:
        print(f"Processing {file}")
        df = load_parquet_file(os.path.join(output_dir, file))
        dataframes.append(df)
        print(f"Loaded df size: {len(df)}")

    combined_df = cudf.concat(dataframes, ignore_index=True)
    del dataframes

    combined_df.drop_duplicates(subset=["smiles"], keep="first", inplace=True)

    return clean_dataframe(combined_df)


def main():
    curr_file_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_file_dir)
    output_dir = os.path.join(parent_dir, "data", "chem")
    df = process_all_chemistry_datasets(output_dir)
    print(f"Final df size: {len(df)}")

    # get pandas dataframe if using cudf library
    if hasattr(df, "to_pandas"):
        if callable(df.to_pandas):
            df = df.to_pandas()

    output_path = os.path.join(output_dir, "full_ds.parquet")
    print(f"Saving dataset to {output_path}")
    df.to_parquet(  # type: ignore
        output_path,
        engine="fastparquet",
        compression="zstd",
        index=False,
    )


if __name__ == "__main__":
    main()
