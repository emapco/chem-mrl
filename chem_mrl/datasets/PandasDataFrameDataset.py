import pandas as pd
from sentence_transformers import InputExample
from torch.utils.data import Dataset


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
    ):
        self._df = df
        self._smiles_a_column = smiles_a_column
        self._smiles_b_column = smiles_b_column
        self._label_column = label_column
        if smiles_b_column is None:
            self._get = self._get_single_smiles_example
        else:
            self._get = self._get_smiles_pair_example

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        return self._get(row)

    def _get_smiles_pair_example(self, row):
        return InputExample(
            texts=row[self._smiles_a_column, self._smiles_b_column],
            label=row[self._label_column],
        )

    def _get_single_smiles_example(self, row):
        return InputExample(
            texts=row[self._smiles_a_column], label=row[self._label_column]
        )
