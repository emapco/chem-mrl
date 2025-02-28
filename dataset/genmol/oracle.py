from collections.abc import Iterable
from typing import Any, Callable

import numpy as np

from chem_mrl.molecular_fingerprinter import MorganFingerprinter

from .types import EvalDatasetDict, ScoreConfig


class Oracle:
    @classmethod
    def RDKitScore(
        cls, name: str, score_config: ScoreConfig | None = None
    ) -> Callable[[str], Any | None]:
        if name == "QED":
            from rdkit.Chem.QED import qed

            return cls._score_from_smiles(qed, config=score_config)
        elif name == "LogP":
            from rdkit.Chem.Crippen import MolLogP  # type: ignore

            return cls._score_from_smiles(MolLogP, config=score_config)
        elif name == "SA":
            from rdkit.Contrib.SA_Score import sascorer  # type: ignore

            return cls._score_from_smiles(sascorer.calculateScore, config=score_config)
        else:
            raise NotImplementedError(f"Unsupported score name: {name}")

    @staticmethod
    def _score_from_smiles(
        func: Callable,
        config: ScoreConfig | None = None,
    ):
        def compute_func_score(smiles: str) -> float | None:
            mol_obj = MorganFingerprinter.mol_from_smiles(smiles)
            if mol_obj is None:
                return None
            return func(mol_obj)

        def composite_score(score: float, similarity: float) -> float:
            assert config is not None
            score = np.clip(score / config.score_cutoff, 0.0, 1.0)
            similarity = np.clip(similarity / config.similarity_cutoff, 0.0, 1.0)
            return score + similarity

        def score(
            smiles: str,
            reference: str | None = None,
            reference_score: float | None = None,
        ):
            # pandarallel requires fingerprinter to be defined here
            # since rdkit objects are not serializable
            fingerprinter = MorganFingerprinter(fp_size=2048)
            base_score = compute_func_score(smiles)

            if config is None:
                return base_score
            if base_score is None or reference is None:
                return base_score

            smiles = fingerprinter.canonicalize_smiles(smiles) or ""
            reference = fingerprinter.canonicalize_smiles(reference) or ""
            similarity = fingerprinter.tanimoto_similarity(
                smiles, reference, config.fingerprint_type
            )
            score = (
                composite_score(base_score, similarity)
                if config.use_composite_score
                else base_score
            )

            response: EvalDatasetDict = {
                "smiles": smiles,
                "base_score": base_score,
                "score": score,
                "similarity": similarity,
                "reference_smiles": reference,
                "reference_score": reference_score,
            }
            return response

        return score

    def __init__(
        self,
        score: Callable,
    ):
        self.score = score

    def evaluate(
        self,
        molecules: Iterable[str],
    ) -> dict[str, float | None]:
        return {smiles: self.score(smiles) for smiles in molecules}

    def evaluate_for_dataset_gen(
        self,
        molecules: Iterable[str],
        reference: str,
        reference_score: float | None = None,
    ):
        scores: list[EvalDatasetDict | None] = [
            self.score(smiles, reference, reference_score) for smiles in molecules
        ]
        return [score for score in scores if score is not None]
