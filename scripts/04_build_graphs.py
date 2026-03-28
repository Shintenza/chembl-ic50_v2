#!/usr/bin/env python3
"""
04_build_graphs.py — Convert cleaned SMILES to PyG graphs, split on the fly.

Usage
-----
    python scripts/04_build_graphs.py
    python scripts/04_build_graphs.py --split-map my_split.parquet

Reads cleaned Parquet files and writes graphs directly into
data/graphs/<split_name>/train/, val/, test/ — no intermediate files.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.graph import build_graphs_by_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build molecular graphs directly into train/val/test dirs.",
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

    output_base = config.PATHS["GRAPHS_DIR"] / split_map_path.stem

    logger.info("=== Build Molecular Graphs ===")
    logger.info("Cleaned dir : %s", config.PATHS["CLEANED_DIR"])
    logger.info("Output dir  : %s", output_base)
    logger.info("Split map   : %s", split_map_path)
    logger.info("Chunk size  : %d", config.GRAPH["CHUNK_SIZE"])

    counts = build_graphs_by_split(
        cleaned_dir=config.PATHS["CLEANED_DIR"],
        split_map_path=split_map_path,
        output_base=output_base,
        chunk_size=config.GRAPH["CHUNK_SIZE"],
    )

    print()
    print("=" * 55)
    print("Done")
    print("=" * 55)
    for split, n in counts.items():
        print(f"  {split:<8}: {n:>8,} graphs → {output_base / split}")
    print("=" * 55)


if __name__ == "__main__":
    main()
