# Copyright 2025 Emmanuel Cortes. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections.abc import Callable

from datasets import Dataset, DatasetDict, Value, load_dataset

from chem_mrl.schemas import DatasetConfig

from .util import get_file_extension

logger = logging.getLogger(__name__)


def load_dataset_with_fallback(dataset_name: str, key: str, columns: list[str]) -> Dataset:
    """Try loading as HF dataset first, fallback to local file."""
    truncated_dataset_name = dataset_name
    if len(dataset_name) > 63:
        truncated_dataset_name = dataset_name[:30] + "..." + dataset_name[-30:]
    try:
        # Try loading as Hugging Face dataset
        logger.info(f"Attempting to load {truncated_dataset_name} as a Hugging Face dataset")
        dataset = load_dataset(dataset_name)
        assert isinstance(dataset, DatasetDict)
        ds = dataset[key]
        logger.info(f"Successfully loaded {truncated_dataset_name}[{key}] from Hugging Face")
    except Exception:
        # Fallback to local file loading
        logger.info(f"Failed to load {truncated_dataset_name} as a HF dataset, trying local file")
        file_type = get_file_extension(dataset_name)
        dataset = load_dataset(file_type, data_files=dataset_name)
        assert isinstance(dataset, DatasetDict)
        ds = dataset[key]
        logger.info(f"Successfully loaded {truncated_dataset_name} as a local {file_type} file")

    # Filter to only the columns we need if they exist
    available_columns = [col for col in columns if col in ds.column_names]
    if len(available_columns) != len(columns):
        missing_columns = set(columns) - set(available_columns)
        raise ValueError(f"Missing required columns: {missing_columns}")

    ds = ds.select_columns(columns)
    logger.info(f"{truncated_dataset_name}[{key}] contains {len(ds)} examples")

    return ds


def load_dataset_from_config(dataset_config: DatasetConfig, seed: int | None = None, is_classifier: bool = False):
    """
    Load and process a single dataset based on its configuration.

    Args:
        dataset_config: Configuration for the dataset

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_columns = [
        dataset_config.smiles_a_column_name,
        dataset_config.smiles_b_column_name,
        dataset_config.label_column_name,
    ]
    if is_classifier:
        data_columns = [
            dataset_config.smiles_a_column_name,
            dataset_config.label_column_name,
        ]

    def raw_to_expected_example_chem_mrl(batch):
        return {
            "smiles_a": batch[dataset_config.smiles_a_column_name],
            "smiles_b": batch[dataset_config.smiles_b_column_name],
            "label": batch[dataset_config.label_column_name],
        }

    def raw_to_expected_example_classifier(batch):
        return {
            "smiles_a": batch[dataset_config.smiles_a_column_name],
            "label": batch[dataset_config.label_column_name],
        }

    raw_to_expected_example: Callable = (
        raw_to_expected_example_classifier if is_classifier else raw_to_expected_example_chem_mrl
    )

    def process_ds(
        ds: Dataset,
        cast: str | None = None,
        sample_size: int | None = None,
    ):
        if sample_size is not None:
            ds = ds.shuffle(seed=seed).select(range(sample_size))
        if cast is not None:
            ds = ds.cast_column(dataset_config.label_column_name, Value(cast))
        ds = ds.map(raw_to_expected_example, batched=True, remove_columns=ds.column_names)
        return ds

    train_ds = None
    if dataset_config.train_dataset is not None:
        train_ds = process_ds(
            load_dataset_with_fallback(
                dataset_config.train_dataset.name,
                dataset_config.train_dataset.split_key,
                data_columns,
            ),
            cast="int64" if is_classifier else dataset_config.train_dataset.label_cast_type.value,
            sample_size=dataset_config.train_dataset.sample_size,
        )

    val_ds = None
    if dataset_config.val_dataset is not None:
        val_ds = process_ds(
            load_dataset_with_fallback(
                dataset_config.val_dataset.name,
                dataset_config.val_dataset.split_key,
                data_columns,
            ),
            cast="int64" if is_classifier else dataset_config.val_dataset.label_cast_type.value,
            sample_size=dataset_config.val_dataset.sample_size,
        )

    test_ds = None
    if dataset_config.test_dataset is not None:
        test_ds = process_ds(
            load_dataset_with_fallback(
                dataset_config.test_dataset.name,
                dataset_config.test_dataset.split_key,
                data_columns,
            ),
            cast="int64" if is_classifier else dataset_config.test_dataset.label_cast_type.value,
            sample_size=dataset_config.test_dataset.sample_size,
        )

    return train_ds, val_ds, test_ds
