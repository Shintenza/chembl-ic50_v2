"""
SMILES standardisation and molecular validity utilities.

All thresholds are read from ``config.CLEANING`` so they remain in one
place throughout the project.
"""

from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level RDKit utilities (constructed once for performance)
# ---------------------------------------------------------------------------

_REMOVER = SaltRemover.SaltRemover()
_UNCHARGER = rdMolStandardize.Uncharger()
_LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()


def standardize_smiles(smiles: str) -> str | None:
    """Return a canonical, standardised SMILES string, or ``None`` on failure.

    Pipeline applied in order:
    1. Parse with RDKit — return ``None`` if unparseable.
    2. Select largest fragment (removes salts/counterions).
    3. Neutralise formal charges where chemically reasonable.
    4. Generate canonical SMILES.

    Parameters
    ----------
    smiles:
        Raw SMILES string as stored in ChEMBL.

    Returns
    -------
    str | None
        Standardised canonical SMILES, or ``None`` if standardisation fails.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        # Keep the largest organic fragment (handles salts like "CC.Cl")
        mol = _LARGEST_FRAG.choose(mol)
        if mol is None:
            return None

        # Neutralise charges (e.g. carboxylate → carboxylic acid)
        mol = _UNCHARGER.uncharge(mol)
        if mol is None:
            return None

        canonical = Chem.MolToSmiles(mol, canonical=True)
        if not canonical:
            return None

        return canonical

    except Exception as exc:
        logger.debug("standardize_smiles failed for %r: %s", smiles, exc)
        return None


def is_valid_molecule(smiles: str) -> bool:
    """Return ``True`` if the molecule passes all physicochemical filters.

    Filters applied (thresholds from ``config.CLEANING``):
    - Heavy atom count in [MIN_ATOMS, MAX_ATOMS]
    - Molecular weight in [MIN_MW, MAX_MW]

    Parameters
    ----------
    smiles:
        A *standardised* canonical SMILES (output of :func:`standardize_smiles`).

    Returns
    -------
    bool
        ``True`` if the molecule passes all filters, ``False`` otherwise.
    """
    if not smiles:
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        num_heavy = mol.GetNumHeavyAtoms()
        if not (config.CLEANING["MIN_ATOMS"] <= num_heavy <= config.CLEANING["MAX_ATOMS"]):
            return False

        mw = Descriptors.ExactMolWt(mol)
        if not (config.CLEANING["MIN_MW"] <= mw <= config.CLEANING["MAX_MW"]):
            return False

        return True

    except Exception as exc:
        logger.debug("is_valid_molecule failed for %r: %s", smiles, exc)
        return False
