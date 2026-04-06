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

logger = logging.getLogger(__name__)


def smiles_to_morgan(
    generator: FingerprintGenerator64, smiles: str
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = generator.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def build_fingerprints(
    cleaned_dir: Path,
    output_dir: Path,
    radius: int,
    n_bits: int,
    chunk_size: int,
) -> int:
    """Build Morgan fingerprint chunks from cleaned parquets into a flat directory.

    Reads every ``batch_*.parquet`` in *cleaned_dir*, converts each SMILES to a
    Morgan fingerprint, and buffers rows until *chunk_size* is reached — then
    flushes a ``chunk_NNNN.pt`` file containing a ``(X, y, activity_ids)`` tuple:

    - ``X``: ``Tensor[N, n_bits]`` — Morgan bit vectors (float32)
    - ``y``: ``Tensor[N]``          — pIC50 labels (float32)
    - ``activity_ids``: ``Tensor[N]`` — ChEMBL activity IDs (int64)

    Parameters
    ----------
    cleaned_dir:
        Directory containing cleaned ``batch_*.parquet`` files.
    output_dir:
        Flat output directory.  ``chunk_*.pt`` and ``metadata.json`` are
        written here directly (no train/val/test subdirs).
    radius:
        Morgan fingerprint radius.
    n_bits:
        Fingerprint bit vector length.
    chunk_size:
        Maximum rows per chunk file.

    Returns
    -------
    int
        Total number of fingerprints written.
    """
    cleaned_dir = Path(cleaned_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(cleaned_dir.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet files found in {cleaned_dir}")

    # Each row: (fp_array, label, activity_id)
    buffer: list[tuple[np.ndarray, float, int]] = []
    chunk_count: int = 0
    total: int = 0
    skipped: int = 0

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for parquet_path in tqdm(parquet_files, desc="Building fingerprints", unit="file"):
        df = pd.read_parquet(parquet_path, engine="pyarrow")

        for _, row in tqdm(df.iterrows(), total=len(df), leave=False, unit="mol"):
            fp = smiles_to_morgan(mfpgen, row["std_smiles"])
            if fp is None:
                skipped += 1
                continue

            buffer.append((fp, float(row["pchembl_value"]), int(row["activity_id"])))
            total += 1

            if len(buffer) >= chunk_size:
                _flush(buffer, output_dir, chunk_count)
                chunk_count += 1
                buffer = []

    if buffer:
        _flush(buffer, output_dir, chunk_count)

    meta = {"total_samples": total}
    (output_dir / "metadata.json").write_text(json.dumps(meta))
    logger.info("Total fingerprints written: %d", total)

    if skipped:
        logger.warning("Skipped %d rows (invalid SMILES)", skipped)

    return total


def _flush(rows: list[tuple[np.ndarray, float, int]], out_dir: Path, n: int) -> None:
    fps = torch.from_numpy(np.stack([r[0] for r in rows]))
    ys = torch.from_numpy(np.array([r[1] for r in rows], dtype=np.float32))
    ids = torch.tensor([r[2] for r in rows], dtype=torch.long)
    torch.save((fps, ys, ids), out_dir / f"chunk_{n:04d}.pt")
