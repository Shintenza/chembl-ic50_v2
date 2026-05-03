import logging
import sys
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from .smiles_utils import standardize_smiles, is_valid_molecule

logger = logging.getLogger(__name__)


def clean_batch(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path, engine="pyarrow")
    input_rows: int = len(df)

    df["std_smiles"] = df["canonical_smiles"].map(standardize_smiles)
    n_after_standardise = df["std_smiles"].notna().sum()
    dropped_invalid_smiles = input_rows - n_after_standardise
    df = df[df["std_smiles"].notna()].copy()

    valid_mask = df["std_smiles"].map(is_valid_molecule)
    n_after_validity = valid_mask.sum()
    dropped_invalid_mol = n_after_standardise - n_after_validity
    df = df[valid_mask].copy()

    pchembl_mask = df["pchembl_value"].between(
        config.CLEANING["MIN_PCHEMBL"],
        config.CLEANING["MAX_PCHEMBL"],
        inclusive="both",
    )
    n_after_pchembl = pchembl_mask.sum()
    dropped_pchembl_range = n_after_validity - n_after_pchembl
    df = df[pchembl_mask].copy()

    df.to_parquet(output_path, index=False, engine="pyarrow")

    stats = {
        "input_rows": input_rows,
        "output_rows": len(df),
        "dropped_invalid_smiles": int(dropped_invalid_smiles),
        "dropped_invalid_mol": int(dropped_invalid_mol),
        "dropped_pchembl_range": int(dropped_pchembl_range),
    }
    logger.debug(
        "clean_batch %s → %s | stats=%s", input_path.name, output_path.name, stats
    )
    return stats


def clean_all_batches(raw_dir: Path, cleaned_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(raw_dir.glob("batch_*.parquet"))
    if not batch_files:
        logger.warning("No batch files found in %s", raw_dir)
        return pd.DataFrame()

    records: list[dict] = []

    for batch_path in tqdm(batch_files, desc="Cleaning batches", unit="file"):
        out_path = cleaned_dir / batch_path.name
        stats = clean_batch(batch_path, out_path)
        stats["batch_file"] = batch_path.name
        records.append(stats)

    summary_df = pd.DataFrame(records)[
        [
            "batch_file",
            "input_rows",
            "output_rows",
            "dropped_invalid_smiles",
            "dropped_invalid_mol",
            "dropped_pchembl_range",
        ]
    ]
    return summary_df
