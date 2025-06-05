import logging

from datasets import Dataset, DatasetDict, load_dataset

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
