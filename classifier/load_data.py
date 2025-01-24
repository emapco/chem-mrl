import logging
import os

import pandas as pd
import torch
from constants import CAT_TO_LABEL
from sentence_transformers import InputExample, LoggingHandler
from torch.utils.data import DataLoader, Dataset

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
    PyTorch Dataset class for a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        df.rename(columns={"smiles": "texts", "category": "label"}, inplace=True)
        df["label"] = df["label"].map(CAT_TO_LABEL)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return InputExample(texts=row["texts"], label=row["label"])


def load_data(
    batch_size: int, train_file: str, val_file: str, test_file: str | None = None
):
    # Load train data
    train_df = pd.read_parquet(train_file, columns=["smiles", "category"])
    train_dl = DataLoader(
        PandasDataFrameDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=12,
    )

    # Load validation data
    val_df = pd.read_parquet(val_file, columns=["smiles", "category"])
    val_dl = DataLoader(
        PandasDataFrameDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=12,
    )

    # Load test data if provided
    if test_file:
        test_df = pd.read_parquet(test_file, columns=["smiles", "category"])
        test_dl = DataLoader(
            PandasDataFrameDataset(test_df),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            pin_memory_device="cuda",
            num_workers=12,
        )
    else:
        test_dl = None

    # Get the number of unique labels
    num_labels = train_df["label"].nunique()

    return train_dl, val_dl, test_dl, num_labels
