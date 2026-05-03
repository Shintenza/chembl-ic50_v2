import logging

from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

SALT_REMOVER = SaltRemover.SaltRemover()
UNCHARGER = rdMolStandardize.Uncharger()
LARGEST_FRAG_CHOOSER = rdMolStandardize.LargestFragmentChooser()


def standardize_smiles(smiles: str) -> str | None:
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        mol = LARGEST_FRAG_CHOOSER.choose(mol)
        if mol is None:
            return None

        mol = UNCHARGER.uncharge(mol)
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
    if len(smiles) == 0:
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        num_heavy = mol.GetNumHeavyAtoms()
        if not (
            config.CLEANING["MIN_ATOMS"] <= num_heavy <= config.CLEANING["MAX_ATOMS"]
        ):
            return False

        mw = Descriptors.ExactMolWt(mol)
        if not (config.CLEANING["MIN_MW"] <= mw <= config.CLEANING["MAX_MW"]):
            return False

        return True

    except Exception as exc:
        logger.debug("is_valid_molecule failed for %r: %s", smiles, exc)
        return False
