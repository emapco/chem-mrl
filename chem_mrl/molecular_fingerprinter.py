import contextlib
import os
from typing import Literal

import numpy as np
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import rdFingerprintGenerator, rdFMCS, rdMolChemicalFeatures
from rdkit.Chem.MolStandardize import rdMolStandardize


class MorganFingerprinter:
    """A class to generate and compare Morgan molecular fingerprints using RDKit."""

    feature_defs = rdMolChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )
    feature_families = feature_defs.GetFeatureFamilies()

    def __init__(self, radius: int = 2, fp_size: int = 4096) -> None:
        """
        Initialize the fingerprint generator with specified parameters.
        A radius of 2 corresponds to ECFP4/FCFP4 fingerprints.

        Args:
            radius: Radius for Morgan fingerprint generation
            fp_size: Size of the fingerprint bit vector
        """
        if radius < 1:
            raise ValueError("Radius must be a positive integer")
        if fp_size < 1:
            raise ValueError("Fingerprint size must be a positive integer")
        self._radius = radius
        self._fp_size = fp_size

        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self._radius,
            fpSize=self._fp_size,
            countSimulation=True,
            includeChirality=True,
        )
        # Default feature atom invariants:
        # https://www.rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        self.functional_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self._radius,
            fpSize=self._fp_size,
            countSimulation=True,
            includeChirality=True,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def fp_size(self) -> int:
        return self._fp_size

    @staticmethod
    def mol_from_smiles(smiles: str) -> Chem.Mol | None:
        """Create a molecule object from SMILES string with standardization fallback."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol = Chem.MolFromSmiles(rdMolStandardize.StandardizeSmiles(smiles))
            return mol
        except Exception:
            return None

    def get_fingerprint(self, smiles: str) -> DataStructs.ExplicitBitVect | None:
        """Generate Morgan fingerprint for a given SMILES string."""
        mol = self.mol_from_smiles(smiles)
        if mol is None:
            return None
        return self.morgan_generator.GetFingerprint(mol)

    def get_functional_fingerprint(self, smiles: str) -> DataStructs.ExplicitBitVect | None:
        """Generate functional Morgan fingerprint for a given SMILES string."""
        mol = self.mol_from_smiles(smiles)
        if mol is None:
            return None
        return self.functional_generator.GetFingerprint(mol)

    def get_fingerprint_numpy(self, smiles: str) -> np.ndarray | float:
        """Convert fingerprint to numpy array format."""
        mol = self.mol_from_smiles(smiles)
        if mol is None:
            return np.nan
        # GetFingerprintAsNumPy       - dtype: uint8  - shape: (fp_size,)
        # GetCountFingerprintsAsNumPy - dtype: uint32 - shape: (fp_size,)
        return self.morgan_generator.GetFingerprintAsNumPy(mol)

    def get_functional_fingerprint_numpy(self, smiles: str) -> np.ndarray | float:
        """Convert fingerprint to numpy array format."""
        mol = self.mol_from_smiles(smiles)
        if mol is None:
            return np.nan
        return self.functional_generator.GetFingerprintAsNumPy(mol)

    def tanimoto_similarity(
        self, smiles_a, smiles_b, fingerprint_type: Literal["morgan", "functional"] = "morgan"
    ) -> float:
        """
        Compute Tanimoto similarity between two molecules.

        Args:
            smiles_a: SMILES string for the first molecule
            smiles_b: SMILES string for the second molecule
            fingerprint_type: Type of fingerprint to use ('morgan' or 'functional')

        Returns:
                Tanimoto similarity score between the two molecules.
                Returns NaN if either fingerprint is None.
        """
        get_fp = (
            self.get_functional_fingerprint
            if fingerprint_type == "functional"
            else self.get_fingerprint
        )

        fp1 = get_fp(smiles_a)
        if fp1 is None:
            return np.nan

        fp2 = get_fp(smiles_b)
        if fp2 is None:
            return np.nan

        return DataStructs.TanimotoSimilarity(fp1, fp2)

    @classmethod
    def canonicalize_smiles(cls, smiles: str) -> str | None:
        """
        Get canonical SMILES string from a given SMILES string.
        """
        mol = cls.mol_from_smiles(smiles)
        if mol is None:
            return None
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles

    @classmethod
    def get_smiles_feature_family_set(cls, smiles: str):
        fam_feat_set = set()
        mol = cls.mol_from_smiles(smiles)
        if mol is None:
            return fam_feat_set

        with contextlib.suppress(IndexError):
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                rawFeats = cls.feature_defs.GetMolFeature(mol, idx)
                fam_feat_set.update(rawFeats.GetFamily().split(", "))
                if len(fam_feat_set) == len(cls.feature_families):
                    return fam_feat_set
        return fam_feat_set

    @classmethod
    def get_strict_common_smiles(
        cls,
        smiles_a: str,
        smiles_b: str,
    ) -> str | None:
        return cls.find_MCS_from_smiles(
            smiles_a=smiles_a,
            smiles_b=smiles_b,
            maximizeBonds=True,
            timeout=7,
            completeRingsOnly=True,
        )

    @classmethod
    def get_common_smiles(cls, smiles_a: str, smiles_b: str) -> str | None:
        return cls.find_MCS_from_smiles(
            smiles_a=smiles_a,
            smiles_b=smiles_b,
            maximizeBonds=False,
            timeout=10,
            completeRingsOnly=False,
        )

    @classmethod
    def find_MCS_from_smiles(cls, smiles_a, smiles_b, **mcs_kwargs) -> str | None:
        try:
            res = rdFMCS.FindMCS(
                [
                    cls.mol_from_smiles(smiles_a),
                    cls.mol_from_smiles(smiles_b),
                ],
                **mcs_kwargs,
            )
            if res.smartsString == "" or res.canceled:
                return None
            smiles = Chem.MolToSmiles(Chem.MolFromSmarts(res.smartsString), canonical=True)
        except Exception:
            return None

        try:
            return rdMolStandardize.StandardizeSmiles(smiles)
        except Exception:
            return smiles
