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
