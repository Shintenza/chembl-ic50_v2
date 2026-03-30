"""
Bemis-Murcko scaffold-based dataset splitting.

Molecules are grouped by their Murcko scaffold.  Scaffold groups are
sorted largest-first and greedily assigned to train, then val, then test,
which ensures that molecules with the same scaffold never appear in
multiple splits (chemical generalisation test).

The split map is intentionally decoupled from the graph files — it stores
only lightweight metadata (activity_id, SMILES, target, label, split) and
can be regenerated or modified independently of the expensive graph-building
step.  ``activity_id`` is the ChEMBL primary key from the ``activities``
table and serves as the stable join key between the split map and the chunk
index.
"""

from __future__ import annotations

import logging
import random
import sys
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.enums import Split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scaffold generation
# ---------------------------------------------------------------------------

_NO_SCAFFOLD = "NO_SCAFFOLD"


def generate_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES for *smiles*.

    Parameters
    ----------
    smiles:
        A standardised canonical SMILES string.

    Returns
    -------
    str
        Canonical scaffold SMILES, or ``'NO_SCAFFOLD'`` if the molecule
        has no ring system or RDKit fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _NO_SCAFFOLD
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        if not scaffold:
            return _NO_SCAFFOLD
        return scaffold
    except Exception as exc:
        logger.debug("generate_scaffold failed for %r: %s", smiles, exc)
        return _NO_SCAFFOLD


# ---------------------------------------------------------------------------
# Splitting algorithm
# ---------------------------------------------------------------------------


def scaffold_split(
    smiles_list: list[str],
    frac_train: float,
    frac_val: float,
    frac_test: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split *smiles_list* indices by scaffold into train / val / test sets.

    Algorithm:
    1. Compute scaffold for every molecule.
    2. Group molecule indices by scaffold.
    3. Shuffle scaffold groups with *seed* for reproducibility.
    4. Sort groups largest-first (greedy) then assign to splits in order:
       train → val → test by cumulative-size thresholds.

    Parameters
    ----------
    smiles_list:
        List of standardised canonical SMILES strings.
    frac_train:
        Fraction of molecules assigned to the training set.
    frac_val:
        Fraction of molecules assigned to the validation set.
    frac_test:
        Fraction of molecules assigned to the test set.
    seed:
        Random seed used to shuffle scaffold groups of equal size.

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        ``(train_idx, val_idx, test_idx)`` — lists of integer indices into
        *smiles_list*.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, (
        "Fractions must sum to 1."
    )

    n = len(smiles_list)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)

    # Build scaffold → [indices] mapping
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(tqdm(smiles_list, desc="Generating scaffolds", leave=False)):
        scaffold = generate_scaffold(smi)
        scaffold_to_indices[scaffold].append(idx)

    # Sort groups: largest first; within equal size, shuffle deterministically
    rng = random.Random(seed)
    scaffold_groups = list(scaffold_to_indices.values())
    scaffold_groups.sort(key=lambda g: (-len(g), rng.random()))

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for group in scaffold_groups:
        if len(train_idx) < n_train:
            train_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    logger.info(
        "scaffold_split: train=%d, val=%d, test=%d (total=%d)",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        len(train_idx) + len(val_idx) + len(test_idx),
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Split-map builder
# ---------------------------------------------------------------------------


def build_split_map(
    cleaned_dir: Path,
    frac_train: float,
    frac_val: float,
    frac_test: float,
    seed: int,
) -> pd.DataFrame:
    """Collect all cleaned molecules, run scaffold split, return a split map.

    Uses ``activity_id`` (ChEMBL's primary key from the ``activities`` table)
    as the stable join key.

    Parameters
    ----------
    cleaned_dir:
        Directory containing cleaned ``batch_*.parquet`` files.
    frac_train, frac_val, frac_test:
        Fractional sizes for each split (must sum to 1).
    seed:
        Random seed for scaffold-group shuffling.

    Returns
    -------
    pd.DataFrame
        Columns: ``activity_id``, ``split``.
        ``split`` values are one of ``'train'``, ``'val'``, ``'test'``.
    """
    cleaned_dir = Path(cleaned_dir)
    parquet_files = sorted(cleaned_dir.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet files found in {cleaned_dir}")

    chunks: list[pd.DataFrame] = []
    for pq_path in tqdm(parquet_files, desc="Loading cleaned data", unit="file"):
        df = pd.read_parquet(pq_path, engine="pyarrow", columns=["activity_id", "std_smiles"])
        chunks.append(df)

    all_df = pd.concat(chunks, ignore_index=True)
    smiles_list = all_df["std_smiles"].tolist()

    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list=smiles_list,
        frac_train=frac_train,
        frac_val=frac_val,
        frac_test=frac_test,
        seed=seed,
    )

    split_labels = pd.array([""] * len(all_df), dtype="object")
    split_labels[train_idx] = Split.TRAIN
    split_labels[val_idx]   = Split.VAL
    split_labels[test_idx]  = Split.TEST

    return pd.DataFrame({
        "activity_id": all_df["activity_id"],
        "split": split_labels,
    })
