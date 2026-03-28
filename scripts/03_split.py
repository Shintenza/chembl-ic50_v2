#!/usr/bin/env python3
"""
03_split.py — Generate a Bemis-Murcko scaffold-based dataset split.

Usage
-----
    python scripts/03_split.py
    python scripts/03_split.py --frac-train 0.7 --frac-val 0.15 --frac-test 0.15 \\
                               --seed 123 --output-name my_split

The split map is saved as a Parquet file in ``config.PATHS['SPLITS_DIR']``.
Running this script multiple times with different arguments creates multiple
named split files; training script 05_train.py selects which one to use via
``--split-map``.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.splitting import build_split_map, save_split_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a scaffold-based train/val/test split map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frac-train",
        type=float,
        default=config.SPLIT["FRAC_TRAIN"],
        help="Fraction of molecules in the training set.",
    )
    parser.add_argument(
        "--frac-val",
        type=float,
        default=config.SPLIT["FRAC_VAL"],
        help="Fraction of molecules in the validation set.",
    )
    parser.add_argument(
        "--frac-test",
        type=float,
        default=config.SPLIT["FRAC_TEST"],
        help="Fraction of molecules in the test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SPLIT["SEED"],
        help="Random seed for scaffold-group shuffling.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="scaffold_split",
        help=(
            "Base name (without extension) for the output split-map file. "
            "Saved as SPLITS_DIR/<output-name>.parquet"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate fractions
    total = args.frac_train + args.frac_val + args.frac_test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0 (got {args.frac_train} + "
            f"{args.frac_val} + {args.frac_test} = {total:.6f})"
        )

    cleaned_dir: Path = config.PATHS["CLEANED_DIR"]
    splits_dir: Path = config.PATHS["SPLITS_DIR"]
    output_path: Path = splits_dir / f"{args.output_name}.parquet"

    logger.info("=== Scaffold Split ===")
    logger.info("Input directory : %s", cleaned_dir)
    logger.info("Output file     : %s", output_path)
    logger.info("Fractions       : train=%.3f  val=%.3f  test=%.3f",
                args.frac_train, args.frac_val, args.frac_test)
    logger.info("Seed            : %d", args.seed)

    split_df = build_split_map(
        cleaned_dir=cleaned_dir,
        frac_train=args.frac_train,
        frac_val=args.frac_val,
        frac_test=args.frac_test,
        seed=args.seed,
    )

    save_split_map(split_df, output_path)

    # Summary
    counts = split_df["split"].value_counts()
    total_n = len(split_df)
    print()
    print("=" * 50)
    print("Split Summary")
    print("=" * 50)
    for split_name in ("train", "val", "test"):
        n = counts.get(split_name, 0)
        pct = 100.0 * n / max(total_n, 1)
        print(f"  {split_name:<8}: {n:>8,}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<8}: {total_n:>8,}")
    print("=" * 50)
    print(f"Saved → {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
