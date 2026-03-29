#!/usr/bin/env python3
"""
06_build_fingerprints.py — Convert cleaned SMILES to Morgan fingerprint chunks.

Usage
-----
    python scripts/06_build_fingerprints.py
    python scripts/06_build_fingerprints.py --split-map scaffold_split.parquet

Reads cleaned Parquet files and writes fingerprint chunks directly into
data/fingerprints/<split_name>/train/, val/, test/.

Each chunk file is a ``(X, y)`` tuple saved with ``torch.save``:
    - ``X``: ``Tensor[N, N_BITS]`` — Morgan bit vectors (float32)
    - ``y``: ``Tensor[N]``          — pIC50 labels (float32)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.features.fingerprints import build_fingerprints_by_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Morgan fingerprint chunks into train/val/test dirs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-map",
        type=str,
        default="scaffold_split.parquet",
        metavar="FILENAME",
        help="Filename of the split map inside SPLITS_DIR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_map_path = config.PATHS["SPLITS_DIR"] / args.split_map
    if not split_map_path.exists():
        raise FileNotFoundError(
            f"Split map not found: {split_map_path}\nRun 03_split.py first."
        )

    output_base = config.PATHS["FINGERPRINTS_DIR"] / split_map_path.stem

    logger.info("=== Build Morgan Fingerprints ===")
    logger.info("Cleaned dir : %s", config.PATHS["CLEANED_DIR"])
    logger.info("Output dir  : %s", output_base)
    logger.info("Split map   : %s", split_map_path)
    logger.info("Radius      : %d", config.FINGERPRINT["RADIUS"])
    logger.info("Bits        : %d", config.FINGERPRINT["N_BITS"])
    logger.info("Chunk size  : %d", config.FINGERPRINT["CHUNK_SIZE"])

    counts = build_fingerprints_by_split(
        cleaned_dir=config.PATHS["CLEANED_DIR"],
        split_map_path=split_map_path,
        output_base=output_base,
        radius=config.FINGERPRINT["RADIUS"],
        n_bits=config.FINGERPRINT["N_BITS"],
        chunk_size=config.FINGERPRINT["CHUNK_SIZE"],
    )

    print()
    print("=" * 55)
    print("Done")
    print("=" * 55)
    for split, n in counts.items():
        print(f"  {split:<8}: {n:>8,} samples → {output_base / split}")
    print("=" * 55)


if __name__ == "__main__":
    main()
