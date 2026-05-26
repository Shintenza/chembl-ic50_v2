import logging
from src.cleaning import standardize_smiles, is_valid_molecule
import pandas as pd
import duckdb
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def get_smiles_median_mapping(
    batch_files_location: Path,
    smiles_col: str,
    target_col: str,
    conflict_threshold: float,
) -> dict[str, float]:
    query = f"""
        SELECT {smiles_col}, median({target_col}) as global_median
        FROM '{batch_files_location}/*.parquet'
        GROUP BY {smiles_col}
        HAVING (max({target_col}) - min({target_col})) <= {conflict_threshold}
    """
    df = duckdb.query(query).to_df()
    return dict(zip(df[smiles_col], df["global_median"]))


def clean_batch(
    input_path: Path, output_path: Path, smiles_median_mapping: dict[str, float]
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path, engine="pyarrow")
    input_rows: int = len(df)

    df["pchembl_value"] = df["canonical_smiles"].map(smiles_median_mapping)
    df = df[df["pchembl_value"].notna()].copy()
    n_after_mapping = len(df)
    dropped_missing_mapping = input_rows - n_after_mapping

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
        "dropped_inconsitent_results": dropped_missing_mapping,
        "dropped_invalid_smiles": dropped_invalid_smiles,
        "dropped_invalid_mol": dropped_invalid_mol,
        "dropped_pchembl_range": dropped_pchembl_range,
    }
    logger.debug(
        "clean_batch %s → %s | stats=%s", input_path.name, output_path.name, stats
    )
    return stats
