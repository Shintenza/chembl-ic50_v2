from rdkit.Chem.AllChem import FingerprintGenerator64

from rdkit import Chem
import numpy as np

from torch import from_numpy, tensor, float32, long


def smiles_to_morgan(
    generator: FingerprintGenerator64, smiles: str
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
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

    x = from_numpy(fp)
    y = tensor(pic50, dtype=float32)
    idx = tensor(id, dtype=long)

    return (x, y, idx)
