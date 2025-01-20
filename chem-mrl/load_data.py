import logging
import os
import gc

import torch
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


def load_data(
    num_of_rows_to_train_on: int | None = 750000,
    num_of_rows_to_validate_on: int | None = 150000,
    dataset_path: str = TRAIN_DS_DICT["morgan-similarity"],
    val_ds_path: str | None = VAL_DS_DICT["morgan-similarity"],
):
    sample_seed = 42
    logging.info(f"Loading {dataset_path} dataset")
    train_df = pd.read_parquet(
        dataset_path, columns=["smiles_a", "smiles_b", "fingerprint_similarity"]
    )
    if num_of_rows_to_train_on is not None:
        train_df = train_df.sample(
            n=num_of_rows_to_train_on, replace=False, random_state=sample_seed
        ).reset_index(drop=True)
    train_df = train_df.astype(
        {"fingerprint_similarity": "float32"}
    )  # model outputs float32 tensor

    if val_ds_path is None:
        # Split train_df into train_df and val_df using pandas
        val_df = train_df.sample(frac=0.15, random_state=sample_seed)
        train_df = train_df.drop(val_df.index)
        val_df = val_df.reset_index(drop=True)
        logging.info("Split train_df into training and validation datasets.")
    else:
        logging.info(f"Loading {val_ds_path} dataset")
        val_df = pd.read_parquet(
            val_ds_path,
            columns=["smiles_a", "smiles_b", "fingerprint_similarity"],
        )
        if num_of_rows_to_validate_on is not None:
            val_df = val_df.sample(
                n=num_of_rows_to_validate_on, replace=False, random_state=sample_seed
            ).reset_index(drop=True)
    # validation uses int8 tensors but keep it as a float for now
    val_df = val_df.astype({"fingerprint_similarity": "float16"})

    logging.info(f"Creating {dataset_path} train dataset")
    train_examples = [
        InputExample(
            texts=[row.smiles_a, row.smiles_b], label=row.fingerprint_similarity
        )
        for _, row in train_df.iterrows()
    ]
    del train_df
    gc.collect()

    logging.info(f"Train samples: {len(train_examples)}")
    logging.info(f"Validation samples: {len(val_df)}")
    return train_examples, val_df
