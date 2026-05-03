from .utils import validate_splits
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


def generate_scaffold(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if not scaffold:
            return None
        return scaffold
    except Exception as exc:
        logger.debug("generate_scaffold failed for %r: %s", smiles, exc)
        return None


def scaffold_split(
    smiles_list: list[str],
    frac_train: float,
    frac_val: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    n = len(smiles_list)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)

    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)

    for idx, smi in enumerate(
        tqdm(smiles_list, desc="Generating scaffolds", leave=False)
    ):
        scaffold = generate_scaffold(smi)
        if scaffold is None:
            continue

        scaffold_to_indices[scaffold].append(idx)

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


@validate_splits
def build_split_map(
    cleaned_dir: Path,
    frac_train: float,
    frac_val: float,
    seed: int,
) -> pd.DataFrame:
    cleaned_dir = Path(cleaned_dir)
    parquet_files = sorted(cleaned_dir.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No cleaned parquet files found in {cleaned_dir}")

    df = pd.concat(
        (
            pd.read_parquet(
                pq_path, engine="pyarrow", columns=["activity_id", "std_smiles"]
            )
            for pq_path in parquet_files
        ),
        ignore_index=True,
    )

    smiles_list = df["std_smiles"].tolist()

    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list=smiles_list,
        frac_train=frac_train,
        frac_val=frac_val,
        seed=seed,
    )

    split_labels = pd.array([""] * len(df), dtype="object")
    split_labels[train_idx] = Split.TRAIN
    split_labels[val_idx] = Split.VAL
    split_labels[test_idx] = Split.TEST

    return pd.DataFrame(
        {
            "activity_id": df["activity_id"],
            "split": split_labels,
        }
    )
