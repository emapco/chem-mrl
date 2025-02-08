import logging
from typing import Callable

import pandas as pd
from sentence_transformers import InputExample
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PandasDataFrameDataset(Dataset):
    """
    PyTorch Dataset class for a Pandas DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str,
        smiles_a_column: str,
        smiles_b_column: str | None = None,
        generate_dataset_examples_at_init: bool = True,
        show_progress_bar: bool = False,
    ):
        self.__df = df
        self.__smiles_a_column = smiles_a_column
        self.__smiles_b_column = smiles_b_column
        self.__label_column = label_column
        self.__show_progress_bar = show_progress_bar
        # strategy pattern - determine which _get function to call at runtime
        self._get = self._set_get_method(self.__smiles_b_column, generate_dataset_examples_at_init)

    def __len__(self):
        return len(self.__df)

    def __getitem__(self, idx: int) -> InputExample:
        return self._get(self.__df.iloc[idx])

    def _get_smiles_pair_example(self, row):
        from sentence_transformers import InputExample  # reimport for pandarallel

        return InputExample(
            texts=[row[self.__smiles_a_column], row[self.__smiles_b_column]],
            label=row[self.__label_column],
        )

    def _get_single_smiles_example(self, row):
        from sentence_transformers import InputExample  # reimport for pandarallel

        return InputExample(texts=row[self.__smiles_a_column], label=row[self.__label_column])

    def _get_pregenerated(self, row) -> InputExample:
        return row["examples"]

    def _pregenerate_examples(self, apply: Callable):
        self.__df["examples"] = self.__df.parallel_apply(apply, axis=1)  # type: ignore

    def _set_get_method(self, smiles_b_column, generate_dataset_examples_at_init):
        if smiles_b_column is None:
            getter = self._get_single_smiles_example
        else:
            getter = self._get_smiles_pair_example

        if generate_dataset_examples_at_init:
            from pandarallel import pandarallel

            pandarallel.initialize(progress_bar=self.__show_progress_bar)
            logger.info("Pregenerate examples to match the expected type by sentence_transformers")
            self._pregenerate_examples(getter)
            getter = self._get_pregenerated

        return getter
