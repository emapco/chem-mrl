import logging
import os

import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import (
    LoggingHandler,
    InputExample,
)
import pandas as pd

from constants import (
    TRAIN_DS_DICT,
    VAL_DS_DICT,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class PandasDataFrameDataset(Dataset):
    """
    PyTorch Dataset class for similarity data from Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return InputExample(
            texts=[row.smiles_a, row.smiles_b], label=row.fingerprint_similarity
        )


def load_data(
    batch_size: int = 16,
    sample_seed: int = 42,
    num_of_rows_to_train_on: int | None = None,
    num_of_rows_to_validate_on: int | None = None,
    train_file: str = TRAIN_DS_DICT["morgan-similarity"],
    val_file: str | None = VAL_DS_DICT["morgan-similarity"],
):
    logging.info(f"Loading {train_file} dataset")
    train_df = pd.read_parquet(
        train_file, columns=["smiles_a", "smiles_b", "fingerprint_similarity"]
    )
    train_df = train_df.astype({"fingerprint_similarity": "float32"})
    if num_of_rows_to_train_on is not None:
        train_df = train_df.sample(
            n=num_of_rows_to_train_on,
            replace=False,
            random_state=sample_seed,
        )
        train_df.reset_index(drop=True, inplace=True)

    train_dataloader = DataLoader(
        PandasDataFrameDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device="cuda",
    )

    if val_file is None:
        logging.info("Splitting train_df into training and validation datasets.")
        val_df = train_df.sample(frac=0.15, random_state=sample_seed)
        train_df.drop(val_df.index, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
    else:
        logging.info(f"Loading {val_file} dataset")
        val_df = pd.read_parquet(
            val_file,
            columns=["smiles_a", "smiles_b", "fingerprint_similarity"],
        )
        if num_of_rows_to_validate_on is not None:
            val_df = val_df.sample(
                n=num_of_rows_to_validate_on,
                replace=False,
                random_state=sample_seed,
            )
            val_df.reset_index(drop=True, inplace=True)

    # validation uses int8 tensors but keep it as a float for now
    val_df = val_df.astype({"fingerprint_similarity": "float16"})
    return train_dataloader, val_df
