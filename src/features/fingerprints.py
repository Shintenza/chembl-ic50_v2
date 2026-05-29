from functools import lru_cache

from rdkit.Chem.AllChem import FingerprintGenerator64

from rdkit import Chem
import numpy as np

import torch
from torch import from_numpy, tensor, float32, long


@lru_cache(maxsize=1)
def _default_generator() -> FingerprintGenerator64:
    from rdkit.Chem import rdFingerprintGenerator
    from config import FINGERPRINT

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT["RADIUS"],
        fpSize=FINGERPRINT["N_BITS"],
    )


def smiles_to_morgan(generator: FingerprintGenerator64, smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fp = generator.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def smiles_to_fingerprint(smiles: str) -> torch.Tensor | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = _default_generator().GetFingerprint(mol)
    return from_numpy(np.array(fp, dtype=np.float32))


def prepare_tensor(
    generator: FingerprintGenerator64,
    smiles: str,
    pic50: float,
    id: int,
):
    fp = smiles_to_morgan(generator, smiles)
    if fp is None:
        return None

    x = from_numpy(fp)
    y = tensor(pic50, dtype=float32)
    idx = tensor(id, dtype=long)

    return (x, y, idx)
