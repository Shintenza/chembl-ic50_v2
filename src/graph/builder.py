"""
Molecular graph builder.

Converts cleaned SMILES into PyTorch Geometric Data objects and writes them
into a single flat directory of chunk files (split-agnostic).  Each Data
object already carries an ``activity_id`` attribute so that training-time
datasets can filter by any split map without recomputing graphs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.atom_features import atom_features
from src.features.bond_features import bond_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def mol_to_graph(
    smiles: str,
    y_value: float,
    activity_id: int,
    target_id: str | None = None,
) -> Data | None:
    """Convert a SMILES string to a PyG Data object.

    Returns None if the molecule is invalid or has no bonds.
    """
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetBonds():
        return None

    atom_feat_list = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.stack(atom_feat_list), dtype=torch.float)

    edge_indices: list[tuple[int, int]] = []
    edge_feat_list: list[np.ndarray] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices.extend([(i, j), (j, i)])
        edge_feat_list.extend([feat, feat])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(np.stack(edge_feat_list), dtype=torch.float)
    y          = torch.tensor([y_value], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.activity_id = activity_id
    if target_id is not None:
        data.target_id = target_id

    return data


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_graphs(
    cleaned_dir: Path,
    output_dir: Path,
    chunk_size: int,
) -> int:
    """Build graphs from cleaned parquets and write flat chunk files.

    Reads every ``batch_*.parquet`` in *cleaned_dir*, converts each row to a
    PyG Data object, and flushes them into ``output_dir/chunk_NNNN.pt`` files
    when the buffer reaches *chunk_size*.  Each Data object carries an
    ``activity_id`` attribute so downstream datasets can apply any split map
    at load time without recomputing graphs.

    Parameters
    ----------
    cleaned_dir:
        Directory containing cleaned ``batch_*.parquet`` files.
    output_dir:
        Flat output directory.  ``chunk_*.pt`` and ``metadata.json`` are
        written here directly (no train/val/test subdirs).
    chunk_size:
        Maximum graphs per chunk file.

    Returns
    -------
    int
        Total number of graphs written.
    """
    cleaned_dir = Path(cleaned_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(cleaned_dir.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet files found in {cleaned_dir}")

    buffer:      list[Data] = []
    chunk_count: int        = 0
    total:       int        = 0

    for parquet_path in tqdm(parquet_files, desc="Building graphs", unit="file"):
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        has_target = "target_chembl_id" in df.columns

        for _, row in tqdm(df.iterrows(), total=len(df), leave=False, unit="mol"):
            g = mol_to_graph(
                smiles=row["std_smiles"],
                y_value=float(row["pchembl_value"]),
                activity_id=int(row["activity_id"]),
                target_id=row["target_chembl_id"] if has_target else None,
            )
            if g is None:
                continue

            buffer.append(g)
            total += 1

            if len(buffer) >= chunk_size:
                _flush(buffer, output_dir, chunk_count)
                chunk_count += 1
                buffer = []

    if buffer:
        _flush(buffer, output_dir, chunk_count)

    meta = {"total_graphs": total}
    (output_dir / "metadata.json").write_text(json.dumps(meta))
    logger.info("Total graphs written: %d", total)

    return total


def _flush(graphs: list[Data], out_dir: Path, n: int) -> None:
    torch.save(graphs, out_dir / f"chunk_{n:04d}.pt")
