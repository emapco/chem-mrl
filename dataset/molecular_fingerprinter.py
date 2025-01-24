from abc import ABC, abstractmethod

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


class MolecularFingerprinter(ABC):
    """Abstract base class for molecular fingerprint generation and comparison."""

    @abstractmethod
    def get_fingerprint(self, smiles: str) -> DataStructs.UIntSparseIntVect | None:
        """Generate fingerprint for a given SMILES string."""
        pass

    @abstractmethod
    def get_functional_fingerprint(
        self, smiles: str
    ) -> DataStructs.UIntSparseIntVect | None:
        """Generate functional fingerprint for a given SMILES string."""
        pass

    @abstractmethod
    def compute_similarity(
        self, row: dict[str, str], fingerprint_type: str = "morgan"
    ) -> float:
        """Compute similarity between two molecules."""
        pass


class MorganFingerprinter:
    """A class to generate and compare Morgan molecular fingerprints using RDKit."""

    def __init__(self, radius: int = 2, fp_size: int = 8192) -> None:
        """
        Initialize the fingerprint generator with specified parameters.
        A radius of 2 corresponds to ECFP4/FCFP4 fingerprints.

        Args:
            radius: Radius for Morgan fingerprint generation
            fp_size: Size of the fingerprint bit vector
        """
        self._radius = radius
        self._fp_size = fp_size

        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self._radius, fpSize=self._fp_size
        )
        # Default feature atom invariants:
        # https://www.rdkit.org/docs/GettingStartedInPython.html#feature-definitions-used-in-the-morgan-fingerprints
        self.functional_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self._radius,
            fpSize=self._fp_size,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def fp_size(self) -> int:
        return self._fp_size

    def _create_mol_from_smiles(self, smiles: str) -> Chem.Mol | None:
        """Create a molecule object from SMILES string with standardization fallback."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol = Chem.MolFromSmiles(rdMolStandardize.StandardizeSmiles(smiles))
            return mol
        except Exception:
            return None

    def get_fingerprint(self, smiles: str) -> DataStructs.UIntSparseIntVect | None:
        """Generate Morgan fingerprint for a given SMILES string."""
        mol = self._create_mol_from_smiles(smiles)
        if mol is None:
            return None
        return self.morgan_generator.GetCountFingerprint(mol)

    def get_functional_fingerprint(
        self, smiles: str
    ) -> DataStructs.UIntSparseIntVect | None:
        """Generate functional Morgan fingerprint for a given SMILES string."""
        mol = self._create_mol_from_smiles(smiles)
        if mol is None:
            return None
        return self.functional_generator.GetCountFingerprint(mol)

    def get_fingerprint_numpy(self, smiles: str) -> np.ndarray | float:
        """Convert fingerprint to numpy array format."""
        mol = self._create_mol_from_smiles(smiles)
        if mol is None:
            return np.nan
        # dtype: uint32  -  shape: (self._fp_size,)
        return self.morgan_generator.GetCountFingerprintAsNumPy(mol)

    def get_functional_fingerprint_numpy(self, smiles: str) -> np.ndarray | float:
        """Convert fingerprint to numpy array format."""
        mol = self._create_mol_from_smiles(smiles)
        if mol is None:
            return np.nan
        # dtype: uint32  -  shape: (self._fp_size,)
        return self.functional_generator.GetCountFingerprintAsNumPy(mol)

    def compute_similarity(
        self, smiles_a, smiles_b, fingerprint_type: str = "morgan"
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
            self.get_fingerprint
            if fingerprint_type == "morgan"
            else self.get_functional_fingerprint
        )

        fp1 = get_fp(smiles_a)
        fp2 = get_fp(smiles_b)

        if fp1 is None or fp2 is None:
            return np.nan

        return DataStructs.TanimotoSimilarity(fp1, fp2)
