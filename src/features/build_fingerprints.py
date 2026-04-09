from functools import partial
from rdkit.Chem.AllChem import FingerprintGenerator64
import json
import logging
from pathlib import Path
from rdkit.Chem import rdFingerprintGenerator

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from .build_features import build_features

logger = logging.getLogger(__name__)


def smiles_to_morgan(
    generator: FingerprintGenerator64, smiles: str
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = generator.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def prepare_tensor(
    generator: FingerprintGenerator64,
    smiles: str,
    pic50: float,
    id: int,
):
    fp = smiles_to_morgan(generator, smiles)
    if fp is None:
        return None

    x = torch.from_numpy(fp)
    y = torch.tensor(pic50, dtype=torch.float32)
    idx = torch.tensor(id, dtype=torch.long)

    return (x, y, idx)


def build_fingerprints(
    cleaned_dir: Path,
    output_dir: Path,
    radius: int,
    n_bits: int,
    chunk_size: int,
) -> int:
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    builder_fn = partial(prepare_tensor, mfpgen)

    return build_features(
        cleaned_dir=cleaned_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        builder_fn=builder_fn,
    )
