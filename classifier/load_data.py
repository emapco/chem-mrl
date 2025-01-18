import logging
import os

import apex
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import (
    InputExample,
    LoggingHandler,
)

import pandas as pd

from constants import CAT_TO_LABEL


os.environ["TOKENIZERS_PARALLELISM"] = "true"
apex.torch.backends.cuda.matmul.allow_tf32 = True
apex.torch.backends.cudnn.allow_tf32 = True
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
        df = df.rename(columns={"smiles": "texts", "category": "label"})
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
    train_ds = PandasDataFrameDataset(train_df)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=16,
    )

    # Load validation data
    val_df = pd.read_parquet(val_file, columns=["smiles", "category"])
    val_ds = PandasDataFrameDataset(val_df)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=16,
    )

    # Load test data if provided
    if test_file:
        test_df = pd.read_parquet(test_file, columns=["smiles", "category"])
        test_ds = PandasDataFrameDataset(test_df)
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            pin_memory_device="cuda",
            num_workers=16,
        )
    else:
        test_dl = None

    # Get the number of unique labels
    num_labels = train_df["category"].nunique()

    return train_dl, val_dl, test_dl, num_labels
