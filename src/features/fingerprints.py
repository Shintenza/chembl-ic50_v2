"""Morgan fingerprint generation and fingerprint dataset builder."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

logger = logging.getLogger(__name__)


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    radius:
        Morgan algorithm radius (number of hops).
    n_bits:
        Length of the bit vector.

    Returns
    -------
    np.ndarray of shape ``(n_bits,)`` and dtype ``float32``, or ``None``
    if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def build_fingerprints_by_split(
    cleaned_dir: Path,
    split_map_path: Path,
    output_base: Path,
    radius: int,
    n_bits: int,
    chunk_size: int,
) -> dict[str, int]:
    """Build Morgan fingerprint chunks from cleaned parquets, split into train/val/test.

    Reads every ``batch_*.parquet`` in *cleaned_dir*, converts each SMILES to a
    Morgan fingerprint, looks up its split, and buffers rows until *chunk_size*
    is reached — then flushes a ``chunk_NNNN.pt`` file containing a
    ``(X, y)`` tuple (``X: Tensor[N, n_bits]``, ``y: Tensor[N]``).

    Parameters
    ----------
    cleaned_dir:
        Directory containing cleaned ``batch_*.parquet`` files.
    split_map_path:
        Split-map Parquet file (columns: activity_id, split).
    output_base:
        Root output directory. ``train/``, ``val/``, ``test/`` subdirs are
        created here.
    radius:
        Morgan fingerprint radius.
    n_bits:
        Fingerprint bit vector length.
    chunk_size:
        Maximum rows per chunk file.

    Returns
    -------
    dict[str, int]
        Total sample counts per split: ``{'train': N, 'val': N, 'test': N}``.
    """
    cleaned_dir    = Path(cleaned_dir)
    split_map_path = Path(split_map_path)
    output_base    = Path(output_base)

    split_map = pd.read_parquet(split_map_path, engine="pyarrow")
    activity_to_split: dict[int, str] = dict(
        zip(split_map["activity_id"].tolist(), split_map["split"].tolist())
    )

    splits = ("train", "val", "test")
    for s in splits:
        (output_base / s).mkdir(parents=True, exist_ok=True)

    # Buffers: lists of (fp_array, label) rows
    buffers:      dict[str, list[tuple[np.ndarray, float]]] = {s: [] for s in splits}
    chunk_counts: dict[str, int]                            = {s: 0  for s in splits}
    total_counts: dict[str, int]                            = {s: 0  for s in splits}
    skipped = 0

    parquet_files = sorted(cleaned_dir.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet files found in {cleaned_dir}")

    for parquet_path in tqdm(parquet_files, desc="Building fingerprints", unit="file"):
        df = pd.read_parquet(parquet_path, engine="pyarrow")

        for _, row in tqdm(df.iterrows(), total=len(df), leave=False, unit="mol"):
            split = activity_to_split.get(int(row["activity_id"]))
            if split is None:
                continue

            fp = smiles_to_morgan(row["std_smiles"], radius=radius, n_bits=n_bits)
            if fp is None:
                skipped += 1
                continue

            buffers[split].append((fp, float(row["pchembl_value"])))
            total_counts[split] += 1

            if len(buffers[split]) >= chunk_size:
                _flush(buffers[split], output_base / split, chunk_counts[split])
                chunk_counts[split] += 1
                buffers[split] = []

    # Flush remaining rows
    for split, buf in buffers.items():
        if buf:
            _flush(buf, output_base / split, chunk_counts[split])

    # Write metadata
    for split in splits:
        meta = {"total_samples": total_counts[split]}
        (output_base / split / "metadata.json").write_text(json.dumps(meta))
        logger.info("%s: %d samples", split, total_counts[split])

    if skipped:
        logger.warning("Skipped %d rows (invalid SMILES)", skipped)

    return total_counts


def _flush(rows: list[tuple[np.ndarray, float]], split_dir: Path, n: int) -> None:
    fps    = np.stack([r[0] for r in rows])
    labels = np.array([r[1] for r in rows], dtype=np.float32)
    chunk  = (torch.from_numpy(fps), torch.from_numpy(labels))
    torch.save(chunk, split_dir / f"chunk_{n:04d}.pt")
