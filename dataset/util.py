import importlib.util
import logging
import sys
from typing import overload

from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles

if importlib.util.find_spec("safe") is not None:
    import safe  # don't import in containers which don't require the safe library


@overload
def init_logging(name: str) -> logging.Logger: ...


@overload
def init_logging() -> None: ...


def init_logging(name: str | None = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    RDLogger.DisableLog("rdApp.*")  # type: ignore - DisableLog is an exported function

    if name is not None:
        return logging.getLogger(name)


def canonicalize_smiles(smiles: str) -> str | None:
    mol = MolFromSmiles(smiles)
    if mol is None:
        try:
            mol = MolFromSmiles(StandardizeSmiles(smiles))  # may raise error
            if mol is None:
                return None
        except Exception:
            return None
    smiles = MolToSmiles(mol, canonical=True)
    return smiles


def parallel_canonicalize_smiles(smiles: str) -> str | None:
    # reimport for pandarallel parallel_apply
    from rdkit.Chem import MolFromSmiles, MolToSmiles  # noqa: F401
    from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles  # noqa: F401

    return canonicalize_smiles(smiles)


def parallel_safe_encode(smiles):
    smiles = parallel_canonicalize_smiles(smiles)
    if smiles is None:
        return None
    try:
        return safe.encode(smiles, ignore_stereo=True)  # type: ignore
    except (safe.SAFEEncodeError, safe.SAFEFragmentationError):  # type: ignore
        return None
