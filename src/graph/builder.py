"""
Molecular graph builder.

Converts cleaned SMILES into PyTorch Geometric Data objects and writes them
directly into per-split chunk files (train/val/test) in a single pass.
No intermediate files are created.
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


def build_graphs_by_split(
    cleaned_dir: Path,
    split_map_path: Path,
    output_base: Path,
    chunk_size: int,
) -> dict[str, int]:
    """Build graphs from cleaned parquets and write directly to split dirs.

    Reads every ``batch_*.parquet`` in *cleaned_dir*, converts each row to a
    PyG Data object, looks up its split from the split map, and appends it to
    the appropriate output buffer.  Buffers are flushed to
    ``output_base/{split}/chunk_NNNN.pt`` files when they reach *chunk_size*.

    Nothing is written to a temporary location — graphs land in their final
    split directory immediately.

    Parameters
    ----------
    cleaned_dir:
        Directory containing cleaned ``batch_*.parquet`` files.
    split_map_path:
        Split-map Parquet file (columns: activity_id, split).
    output_base:
        Root output directory.  train/, val/, test/ subdirs are created here.
    chunk_size:
        Maximum graphs per chunk file.

    Returns
    -------
    dict[str, int]
        Total graph counts per split: ``{'train': N, 'val': N, 'test': N}``.
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

    buffers:      dict[str, list[Data]] = {s: [] for s in splits}
    chunk_counts: dict[str, int]        = {s: 0  for s in splits}
    total_counts: dict[str, int]        = {s: 0  for s in splits}

    parquet_files = sorted(cleaned_dir.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet files found in {cleaned_dir}")

    for parquet_path in tqdm(parquet_files, desc="Building graphs", unit="file"):
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        has_target = "target_chembl_id" in df.columns

        for _, row in tqdm(df.iterrows(), total=len(df), leave=False, unit="mol"):
            split = activity_to_split.get(int(row["activity_id"]))
            if split is None:
                continue

            g = mol_to_graph(
                smiles=row["std_smiles"],
                y_value=float(row["pchembl_value"]),
                activity_id=int(row["activity_id"]),
                target_id=row["target_chembl_id"] if has_target else None,
            )
            if g is None:
                continue

            buffers[split].append(g)
            total_counts[split] += 1

            if len(buffers[split]) >= chunk_size:
                _flush(buffers[split], output_base / split, chunk_counts[split])
                chunk_counts[split] += 1
                buffers[split] = []

    # Flush remaining graphs
    for split, buf in buffers.items():
        if buf:
            _flush(buf, output_base / split, chunk_counts[split])

    # Write metadata so SplitGraphDataset knows total counts without scanning files
    for split in splits:
        meta = {"total_graphs": total_counts[split]}
        (output_base / split / "metadata.json").write_text(json.dumps(meta))
        logger.info("%s: %d graphs", split, total_counts[split])

    return total_counts


def _flush(graphs: list[Data], split_dir: Path, n: int) -> None:
    torch.save(graphs, split_dir / f"chunk_{n:04d}.pt")
