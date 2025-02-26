import random
from abc import ABC, abstractmethod

import pandas as pd
from pandarallel import pandarallel

from .oracle import Oracle
from .utils import Utils


class BaseLibrary(ABC):
    _df_columns = ["smiles", "score"]

    @staticmethod
    def from_parquet(file_name, columns=None):
        if columns is None:
            columns = ["smiles", "score"]
        return pd.read_parquet(file_name, columns=columns)

    @staticmethod
    def from_csv(file_name):
        return pd.read_csv(file_name, index_col=0)

    @classmethod
    def fragment_molecules(cls, molecules: pd.Series, oracle: Oracle, num_workers=None):
        if num_workers is not None:
            pandarallel.initialize(progress_bar=True, nb_workers=num_workers)
        else:
            pandarallel.initialize(progress_bar=True)

        def fragment_molecule(smiles) -> pd.DataFrame | None:
            fragments: dict[str, float | None] = oracle.evaluate(Utils.cut(smiles))
            if len(fragments) == 0:
                return None
            return pd.DataFrame({"smiles": fragments.keys(), "score": fragments.values()})

        fragmented_molecules: pd.Series = molecules.parallel_apply(lambda x: fragment_molecule(x))
        fragmented_molecules.dropna(inplace=True)
        return pd.concat(fragmented_molecules.tolist(), ignore_index=True)

    @abstractmethod
    def export(self, num: int) -> list[str]:
        pass

    @abstractmethod
    def update(self, molecules: dict[str, float | None]):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.exported = []


class FragmentLibrary(BaseLibrary):
    def __init__(self, fragments: pd.DataFrame | None = None, max_fragments=1000):
        if fragments is None:
            self.fragments = pd.DataFrame(columns=self._df_columns)

        self.max_fragments = max_fragments
        self.top_n(self.max_fragments)

        self.molecules = pd.DataFrame(columns=self._df_columns)

    def top_n(self, n):
        if self.fragments.shape[0] > n:
            self.fragments = self.fragments.sort_values(
                "score", ascending=False, ignore_index=True
            ).head(n)

    def export(self, num=1):
        self.exported = []
        fragments = self.fragments["smiles"].to_list()

        for _ in range(num):
            num_try, max_try = 0, 100
            safe_text = None
            while num_try < max_try:
                frag1, frag2 = random.sample(fragments, 2)
                combined = Utils.attach(frag1, frag2)

                safe_text = None if combined is None else Utils.smiles2safe(combined)

                if safe_text is not None:
                    break

            assert safe_text is not None
            self.exported.append(safe_text)

        return self.exported

    def update(self, molecules):
        min_score = self.fragments["score"].min() if self.fragments.shape[0] > 0 else 0.0

        unique_molecules = set(self.molecules["smiles"].to_list())
        better_molecules = {
            k: v
            for k, v in molecules.items()
            if (v is not None) and (v > min_score) and (k not in unique_molecules)
        }

        if len(better_molecules) == 0:
            return

        new_molecules = []
        new_fragments = []

        for m, score in better_molecules.items():
            new_molecules.append([m, score])

            for frag in Utils.cut(m):
                new_fragments.append([frag, score])

        df_fragments = pd.DataFrame(new_fragments, columns=self.fragments.columns)

        if self.fragments.shape[0] > 0:
            self.fragments = pd.concat([self.fragments, df_fragments])
        else:
            self.fragments = df_fragments

        self.top_n(self.max_fragments)

        df_molecules = pd.DataFrame(new_molecules, columns=self.molecules.columns)

        if self.molecules.shape[0] > 0:
            self.molecules = pd.concat([self.molecules, df_molecules])
        else:
            self.molecules = df_molecules

        self.molecules = self.molecules.sort_values("score", ascending=False, ignore_index=True)


class MoleculeLibrary(BaseLibrary):
    def __init__(self, max_molecules=1000):
        self.molecules = pd.DataFrame(columns=self._df_columns)
        self.max_molecules = max_molecules
        self.top_n(max_molecules)

    def top_n(self, n):
        if self.molecules.shape[0] > n:
            self.molecules = self.molecules.sort_values(
                "score", ascending=False, ignore_index=True
            ).head(n)

    def export(self, num=1):
        self.exported = []

        molecules = self.molecules["smiles"].sample(num)

        for molecule in molecules:
            safe_text = Utils.smiles2safe(molecule)
            assert safe_text is not None
            self.exported.append(safe_text)

        return self.exported

    def update(self, molecules):
        min_score = self.molecules["score"].min() if self.molecules.shape[0] > 0 else 0.0

        unique_molecules = set(self.molecules["smiles"].to_list())
        better_molecules = {
            k: v
            for k, v in molecules.items()
            if (v is not None) and (v > min_score) and (k not in unique_molecules)
        }

        if len(better_molecules) == 0:
            return

        new_molecules = []

        for m, score in better_molecules.items():
            new_molecules.append([m, score])

        df_molecules = pd.DataFrame(new_molecules, columns=self.molecules.columns)

        if self.molecules.shape[0] > 0:
            self.molecules = pd.concat([self.molecules, df_molecules])
        else:
            self.molecules = df_molecules

        self.top_n(self.max_molecules)
